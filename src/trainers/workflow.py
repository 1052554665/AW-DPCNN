from collections import Counter
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

from src.datasets import build_dataloaders
from src.models import build_model
from src.utils.plot_confusion import plot_confusion
from src.utils.train_eval import evaluate, save_logs, train_one_epoch
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
    if not scheduler_cfg or str(scheduler_cfg.get("type", "none")).lower() == "none":
        return None

    scheduler_type = str(scheduler_cfg.get("type", "step")).lower()
    if scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(scheduler_cfg.get("step_size", 20)),
            gamma=float(scheduler_cfg.get("gamma", 0.1)),
        )
    if scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(scheduler_cfg.get("t_max", 10)),
            eta_min=float(scheduler_cfg.get("eta_min", 1e-6)),
        )
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


def train_and_evaluate(config: Dict, run_dir: Path, device: torch.device):
    loaders, class_names = build_dataloaders(config, device=device)
    model = build_model(config).to(device)

    train_cfg = config.get("train", {})
    optimizer = _build_optimizer(model, train_cfg)
    scheduler = _build_scheduler(optimizer, config.get("scheduler", {}))
    criterion = _build_loss(train_cfg, loaders["train"].dataset, len(class_names), device)

    best_val_f1 = -1.0
    logs = []
    epochs = int(train_cfg.get("epochs", 30))

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, loaders["train"], optimizer, criterion, device)
        val_loss, val_metrics, _, _ = evaluate(model, loaders["val"], criterion, device)
        val_acc, prec, rec, f1, gmean, bal_acc, kappa = val_metrics

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
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        if scheduler is not None:
            scheduler.step()

        if f1 > best_val_f1:
            best_val_f1 = f1
            torch.save(model.state_dict(), run_dir / "checkpoints" / "best.pt")

        if epoch % int(train_cfg.get("print_freq", 1)) == 0:
            print(
                f"[Epoch {epoch:03d}/{epochs:03d}] "
                f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} "
                f"val_f1={f1:.4f}"
            )

    torch.save(model.state_dict(), run_dir / "checkpoints" / "last.pt")
    save_logs(logs, path=str(run_dir / "logs" / "train_log.csv"))

    model.load_state_dict(torch.load(run_dir / "checkpoints" / "best.pt", map_location=device))
    test_loss, test_metrics, y_true, y_pred = evaluate(model, loaders["test"], criterion, device)
    test_acc, test_prec, test_rec, test_f1, test_gmean, test_bal_acc, test_kappa = test_metrics

    results = {
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_precision": test_prec,
        "test_recall": test_rec,
        "test_f1": test_f1,
        "test_gmean": test_gmean,
        "test_bal_acc": test_bal_acc,
        "test_kappa": test_kappa,
    }

    vis_cfg = config.get("visualization", {})
    if bool(vis_cfg.get("confusion_matrix", True)):
        plot_confusion(
            y_true,
            y_pred,
            class_names,
            save_path=str(run_dir / "figures" / "confusion_matrix.png"),
        )

    if bool(vis_cfg.get("tsne", False)):
        features, labels = extract_features(model, loaders["test"], device)
        plot_tsne(
            features,
            labels,
            class_names,
            title="t-SNE Visualization on Test Set",
            save_path=str(run_dir / "figures" / "tsne.png"),
        )

    return results

