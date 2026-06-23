from collections import Counter
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

from src.datasets import build_dataloaders
from src.models import build_model
from src.utils.metrics import count_parameters, compute_flops
from src.utils.plot_confusion import plot_confusion
from src.utils.plot_roc import plot_roc_curves
from src.utils.train_eval import calculate_roc_auc, check_label_leakage, evaluate, save_logs, train_one_epoch
from src.utils.tsne import extract_features, plot_tsne


def _build_optimizer(model, train_cfg: Dict):
    optimizer_name = str(train_cfg.get("optimizer", "adamw")).lower()
    lr = float(train_cfg.get("lr", 1e-4))
    wd = float(train_cfg.get("weight_decay", 1e-4))

    if optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(train_cfg.get("momentum", 0.9)),
            weight_decay=wd,
        )
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)


def _build_scheduler(optimizer, scheduler_cfg: Dict):
    """Build a learning rate scheduler from config.

    Supported types:
        - plateau: ReduceLROnPlateau (patience=5, factor=0.5, mode='max')
        - step: StepLR
        - cosine: CosineAnnealingLR
        - none / missing: no scheduler
    """
    if not scheduler_cfg or str(scheduler_cfg.get("type", "none")).lower() == "none":
        return None, None  # (scheduler, scheduler_kind)

    scheduler_type = str(scheduler_cfg.get("type", "plateau")).lower()

    if scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=str(scheduler_cfg.get("mode", "max")),
            factor=float(scheduler_cfg.get("factor", 0.5)),
            patience=int(scheduler_cfg.get("patience", 5)),
            min_lr=float(scheduler_cfg.get("min_lr", 1e-6)),
        )
        return scheduler, "plateau"

    if scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(scheduler_cfg.get("step_size", 20)),
            gamma=float(scheduler_cfg.get("gamma", 0.1)),
        )
        return scheduler, "epoch"

    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(scheduler_cfg.get("t_max", 10)),
            eta_min=float(scheduler_cfg.get("eta_min", 1e-6)),
        )
        return scheduler, "epoch"

    raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


def _build_loss(train_cfg: Dict, dataset, num_classes: int, device: torch.device):
    if not bool(train_cfg.get("class_weighting", True)):
        return nn.CrossEntropyLoss()

    targets = [label for _, label in dataset.samples]
    counter = Counter(targets)
    num_samples = len(targets)
    weights = [num_samples / (num_classes * max(counter[i], 1)) for i in range(num_classes)]
    class_weights = torch.tensor(weights, dtype=torch.float, device=device)
    return nn.CrossEntropyLoss(weight=class_weights)


def train_and_evaluate(config: Dict, run_dir: Path, device: torch.device, seed: Optional[int] = None):
    loaders, class_names = build_dataloaders(config, device=device, seed=seed)
    num_classes = len(class_names)
    model = build_model(config).to(device)

    # --- cuDNN / CUDA performance tuning ---
    # set_seed() disables these for reproducibility; re-enable for training speed.
    # - benchmark=True: cuDNN auto-tuner finds the fastest conv algo for this input size
    # - TF32 matmul: leverages Ampere+ tensor cores for float32 matmul (~2× faster)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    # --- torch.compile (PyTorch 2.0+) ---
    _compile_enabled = bool(config.get("train", {}).get("compile", True))
    if _compile_enabled and hasattr(torch, "compile") and device.type == "cuda":
        try:
            # "default" mode gives ~30-50% speedup with no CUDA-graph issues.
            # "reduce-overhead" / "max-autotune" offer more speed but require
            # CUDA-graph-compatible models and careful tensor lifetime mgmt.
            compile_mode = str(config.get("train", {}).get("compile_mode", "default"))
            model = torch.compile(model, mode=compile_mode)
            print(f"[torch.compile] Model compiled (mode={compile_mode})")
        except Exception as exc:
            print(f"[torch.compile] Compilation failed ({exc}); falling back to eager mode.")

    # --- Model complexity metrics ---
    n_params = count_parameters(model, trainable_only=True)
    img_size = int(config.get("dataset", {}).get("img_size", 224))
    in_ch = int(config.get("model", {}).get("in_channels", 3))
    try:
        n_flops = compute_flops(model, input_shape=(1, in_ch, img_size, img_size), device=device)
    except Exception:
        n_flops = 0
    model_name = str(config.get("model", {}).get("name", "unknown"))
    print(f"{model_name} | Params: {n_params / 1e6:.2f}M | FLOPs: {n_flops / 1e6:.2f}M")

    train_cfg = config.get("train", {})
    optimizer = _build_optimizer(model, train_cfg)
    scheduler, scheduler_kind = _build_scheduler(optimizer, config.get("scheduler", {}))
    criterion = _build_loss(train_cfg, loaders["train"].dataset, num_classes, device)

    # --- Early stopping config ---
    early_stop_cfg = train_cfg.get("early_stopping", {})
    early_stop_enabled = bool(early_stop_cfg.get("enabled", True))
    early_stop_patience = int(early_stop_cfg.get("patience", 15))
    early_stop_metric = str(early_stop_cfg.get("metric", "val_f1"))  # "val_f1" or "val_loss"

    best_val_f1 = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    early_stop_counter = 0

    logs = []
    epochs = int(train_cfg.get("epochs", 30))
    log_path = run_dir / "logs" / "train_log.csv"
    best_ckpt_path = run_dir / "checkpoints" / "best.pt"
    last_ckpt_path = run_dir / "checkpoints" / "last.pt"
    confusion_path = run_dir / "figures" / "confusion_matrix.png"
    tsne_path = run_dir / "figures" / "tsne.png"
    roc_path = run_dir / "figures" / "roc_curve.png"

    try:
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_one_epoch(model, loaders["train"], optimizer, criterion, device)
            val_loss, val_metrics, _, _, val_y_score = evaluate(model, loaders["val"], criterion, device)
            val_acc, prec, rec, f1, gmean, bal_acc, kappa = val_metrics
            val_auc = calculate_roc_auc(
                [label for _, label in loaders["val"].dataset.samples],
                val_y_score,
                num_classes,
            )

            # --- Scheduler step ---
            if scheduler is not None:
                if scheduler_kind == "plateau":
                    # ReduceLROnPlateau steps on the monitored metric
                    monitor_val = f1 if early_stop_metric == "val_f1" else -val_loss
                    scheduler.step(monitor_val)
                else:
                    scheduler.step()

            # --- Early stopping check ---
            if early_stop_enabled:
                if early_stop_metric == "val_loss":
                    is_better = val_loss < best_val_loss
                    best_val_loss = min(val_loss, best_val_loss)
                else:
                    is_better = f1 > best_val_f1

                if is_better:
                    best_val_f1 = max(f1, best_val_f1)
                    best_val_loss = min(val_loss, best_val_loss)
                    best_epoch = epoch
                    early_stop_counter = 0
                    torch.save(model.state_dict(), best_ckpt_path)
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= early_stop_patience:
                        print(
                            f"[Early Stop] No improvement in {early_stop_metric} "
                            f"for {early_stop_patience} epochs. Stopping at epoch {epoch}."
                        )
                        break
            else:
                # Without early stopping, save best model by F1
                if f1 > best_val_f1:
                    best_val_f1 = f1
                    best_epoch = epoch
                    torch.save(model.state_dict(), best_ckpt_path)

            logs.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                    "gmean": gmean,
                    "val_bal_acc": bal_acc,
                    "val_kappa": kappa,
                    "val_auc": val_auc,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )

            # Persist logs every epoch so long runs keep progress on disk.
            save_logs(logs, path=str(log_path))

            if epoch % int(train_cfg.get("print_freq", 1)) == 0:
                print(
                    f"[Epoch {epoch:03d}/{epochs:03d}] "
                    f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} "
                    f"val_f1={f1:.4f} val_auc={val_auc:.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}"
                )
    finally:
        torch.save(model.state_dict(), last_ckpt_path)
        if logs:
            save_logs(logs, path=str(log_path))

    # --- Restore best checkpoint for final test ---
    if best_ckpt_path.exists():
        print(f"Loading best checkpoint from epoch {best_epoch} (val_f1={best_val_f1:.4f})")
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))

    test_loss, test_metrics, y_true, y_pred, y_score = evaluate(model, loaders["test"], criterion, device)
    test_acc, test_prec, test_rec, test_f1, test_gmean, test_bal_acc, test_kappa = test_metrics
    test_auc = calculate_roc_auc(y_true, y_score, num_classes)

    # ── Label-shuffling leakage check ──
    leakage = check_label_leakage(y_true, y_pred, num_classes)
    if leakage["is_suspicious"]:
        print("\n" + "!" * 60)
        print(leakage["warning"])
        print("!" * 60 + "\n")
    else:
        print(f"[OK] Shuffled-label check passed "
              f"(shuffled acc = {leakage['shuffled_acc']:.4f}, "
              f"chance = {leakage['chance_level']:.4f})")

    results = {
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_precision": test_prec,
        "test_recall": test_rec,
        "test_f1": test_f1,
        "test_gmean": test_gmean,
        "test_bal_acc": test_bal_acc,
        "test_kappa": test_kappa,
        "test_auc": test_auc,
        "shuffled_acc": leakage["shuffled_acc"],
        "chance_level": leakage["chance_level"],
        "leakage_suspicious": leakage["is_suspicious"],
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "params": n_params,
        "flops": n_flops,
        "model_name": model_name,
    }

    vis_cfg = config.get("visualization", {})
    if bool(vis_cfg.get("confusion_matrix", True)):
        plot_confusion(
            y_true,
            y_pred,
            class_names,
            save_path=str(confusion_path),
        )

    if bool(vis_cfg.get("roc_curve", True)):
        plot_roc_curves(
            y_true,
            y_score,
            class_names,
            save_path=str(roc_path),
        )

    if bool(vis_cfg.get("tsne", False)):
        features, labels = extract_features(model, loaders["test"], device)
        plot_tsne(
            features,
            labels,
            class_names,
            title="t-SNE Visualization on Test Set",
            save_path=str(tsne_path),
        )

    return results

