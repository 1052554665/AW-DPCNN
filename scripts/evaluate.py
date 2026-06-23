import argparse
from pathlib import Path

import torch
import torch.nn as nn

from src.datasets import build_dataloaders
from src.models import build_model
from src.utils.config import load_yaml
from src.utils.metrics import count_parameters, compute_flops
from src.utils.plot_confusion import plot_confusion
from src.utils.plot_roc import plot_roc_curves
from src.utils.train_eval import calculate_roc_auc, check_label_leakage, evaluate
from src.utils.tsne import extract_features, plot_tsne


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint with a resolved config.")
    parser.add_argument("--config", required=True, help="Path to resolved/base YAML config")
    parser.add_argument("--checkpoint", required=True, help="Path to model .pt checkpoint")
    parser.add_argument("--output-dir", required=True, help="Where to save evaluation artifacts")
    parser.add_argument("--device", default="", help="cuda/cpu override")
    return parser.parse_args()


def resolve_device(config_device: str, cli_device: str) -> torch.device:
    requested = cli_device.strip() or str(config_device).strip()
    if not requested or requested.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device, but CUDA is not available.")
    return torch.device(requested)


def main():
    args = parse_args()
    config = load_yaml(args.config)

    device = resolve_device(config.get("device", "auto"), args.device)

    model = build_model(config).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    # --- Model complexity ---
    n_params = count_parameters(model, trainable_only=True)
    img_size = int(config.get("dataset", {}).get("img_size", 224))
    in_ch = int(config.get("model", {}).get("in_channels", 3))
    try:
        n_flops = compute_flops(model, input_shape=(1, in_ch, img_size, img_size), device=device)
    except Exception:
        n_flops = 0
    model_name = str(config.get("model", {}).get("name", "unknown"))
    print(f"{model_name} | Params: {n_params / 1e6:.2f}M | FLOPs: {n_flops / 1e6:.2f}M")

    loaders, class_names = build_dataloaders(config)
    num_classes = len(class_names)
    criterion = nn.CrossEntropyLoss()
    test_loss, metrics, y_true, y_pred, y_score = evaluate(model, loaders["test"], criterion, device)
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion(y_true, y_pred, class_names, save_path=str(output_dir / "confusion_matrix.png"))

    if bool(config.get("visualization", {}).get("roc_curve", True)):
        plot_roc_curves(y_true, y_score, class_names, save_path=str(output_dir / "roc_curve.png"))

    if bool(config.get("visualization", {}).get("tsne", False)):
        features, labels = extract_features(model, loaders["test"], device)
        plot_tsne(features, labels, class_names, save_path=str(output_dir / "tsne.png"))

    keys = ["acc", "precision", "recall", "f1", "gmean", "bal_acc", "kappa", "auc"]
    values = list(metrics) + [test_auc]
    print(f"test_loss={test_loss:.6f}")
    for key, value in zip(keys, values):
        print(f"{key}={value:.6f}")
    print(f"params={n_params}  flops={n_flops}")


if __name__ == "__main__":
    main()

