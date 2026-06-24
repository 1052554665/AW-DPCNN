#!/usr/bin/env python3
"""
Noise Robustness Evaluation
============================
Evaluate trained models under additive Gaussian noise at multiple SNR
levels.  Produces accuracy‑vs‑SNR curves and a summary CSV.

Noise is injected in **pixel space** (before normalisation) to simulate
acoustic measurement noise propagating through the fused representation.

Usage::

    # Single model
    python scripts/noise_robustness.py \\
        --config configs/default.yaml \\
        --exp-config experiments/exp1/MSCA_VGG16.yaml \\
        --checkpoint PATH/TO/best.pt

    # Batch: evaluate all models in an experiment directory
    python scripts/noise_robustness.py \
        --config configs/default.yaml \
        --exp-dir experiments/exp1 \
        --auto-checkpoint  --snr 5 10 15 20 25 30 # picks best.pt from the latest run of each config

    # Custom SNR range
    python scripts/noise_robustness.py \\
        --config configs/default.yaml \\
        --exp-config experiments/exp1/vgg16.yaml \\
        --checkpoint .../best.pt \\
        --snr -10 -5 0 5 10 15 20
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# ── Project imports ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models import build_model
from src.utils.config import load_config, load_yaml
from src.utils.train_eval import evaluate


# ═══════════════════════════════════════════════════════════════════════
#  Noise injection transform
# ═══════════════════════════════════════════════════════════════════════

class AddGaussianNoise:
    """Add Gaussian noise to a tensor in [0, 1] to achieve a target SNR (dB).

    SNR = 10·log₁₀(P_signal / P_noise), where P_signal = mean(x²).

    Parameters
    ----------
    snr_db : float or None
        Target SNR in dB.  ``None`` → no noise (clean).
    """

    def __init__(self, snr_db: Optional[float]):
        self.snr_db = snr_db

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.snr_db is None:
            return x
        # x in [0, 1] after ToTensor
        signal_power = x.pow(2).mean()
        if signal_power == 0:
            return x
        snr_linear = 10 ** (self.snr_db / 10.0)
        noise_power = signal_power / snr_linear
        noise = torch.randn_like(x) * noise_power.sqrt()
        return (x + noise).clamp(0.0, 1.0)

    def __repr__(self):
        return f"AddGaussianNoise(SNR={self.snr_db} dB)"


# ═══════════════════════════════════════════════════════════════════════
#  Dataloader builder with configurable noise
# ═══════════════════════════════════════════════════════════════════════

def build_noisy_loader(
    config: Dict,
    snr_db: Optional[float],
    batch_size: int = 64,
    num_workers: int = 8,
) -> DataLoader:
    """Build a test DataLoader that injects Gaussian noise at *snr_db*."""
    data_cfg = config["dataset"]
    root_dir = Path(data_cfg["root_dir"]).expanduser().resolve()
    test_dir = root_dir / data_cfg.get("test_split", "test")
    img_size = int(data_cfg.get("img_size", 224))
    mean = data_cfg.get("normalize_mean", [0.485, 0.456, 0.406])
    std = data_cfg.get("normalize_std", [0.229, 0.224, 0.225])

    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),                     # → [0, 1]
        AddGaussianNoise(snr_db),                   # ← noise here
        transforms.Normalize(mean=mean, std=std),   # → ImageNet stats
    ])

    ds = datasets.ImageFolder(str(test_dir), transform=tf)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )


# ═══════════════════════════════════════════════════════════════════════
#  Single‑model evaluation across SNRs
# ═══════════════════════════════════════════════════════════════════════

def evaluate_model_noise(
    config: Dict,
    checkpoint_path: str,
    snr_values: List[Optional[float]],
    device: torch.device,
    num_workers: int = 8,
) -> Dict[str, list]:
    """Evaluate a trained model at multiple SNR levels.

    Returns dict with keys: snr, accuracy, f1, auc, loss
    """
    model = build_model(config).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    if any(k.startswith("_orig_mod.") for k in state.keys()):
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    results = {"snr": [], "accuracy": [], "f1": [], "auc": [], "loss": []}

    for snr in tqdm(snr_values, desc="SNR levels", ncols=80):
        loader = build_noisy_loader(config, snr,
                                     batch_size=64, num_workers=num_workers)
        loss, metrics, y_true, y_pred, y_score = evaluate(
            model, loader, criterion, device,
        )
        acc, prec, rec, f1, gmean, bal_acc, kappa = metrics

        # Compute AUC
        from src.utils.train_eval import calculate_roc_auc
        num_classes = len(loader.dataset.classes)
        auc_val = calculate_roc_auc(y_true, y_score, num_classes)

        label = "clean" if snr is None else f"{snr} dB"
        results["snr"].append(label)
        results["accuracy"].append(round(acc * 100, 2))
        results["f1"].append(round(f1 * 100, 2))
        results["auc"].append(round(auc_val * 100, 2))
        results["loss"].append(round(loss, 4))

    return results


# ═══════════════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════════════

def plot_noise_curves(
    all_results: Dict[str, Dict[str, list]],
    save_path: str,
):
    """Plot accuracy‑vs‑SNR for multiple models on one figure."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "stix",
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 15,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "figure.dpi": 300,
        "axes.linewidth": 0.8,
    })

    fig, ax = plt.subplots(figsize=(8, 5.5))

    markers = ["o", "s", "D", "^", "v", "p", "*", "h"]
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(all_results), 3)))

    for idx, (model_name, res) in enumerate(all_results.items()):
        # Convert SNR labels to numeric for plotting
        snr_num = []
        acc_vals = []
        for snr_str, acc in zip(res["snr"], res["accuracy"]):
            if snr_str == "clean":
                snr_num.append(25)  # plot clean at SNR=25 for visual separation
            else:
                snr_num.append(float(snr_str.replace(" dB", "")))
            acc_vals.append(acc)

        # Sort by SNR
        order = np.argsort(snr_num)
        snr_num = [snr_num[i] for i in order]
        acc_vals = [acc_vals[i] for i in order]

        ax.plot(
            snr_num, acc_vals,
            marker=markers[idx % len(markers)],
            color=colors[idx % len(colors)],
            lw=1.8, markersize=6,
            label=model_name,
        )

    # Format x-axis: show "Clean" label, then SNR values
    all_snr_labels = sorted(set(
        float(r.replace(" dB", "")) if r != "clean" else 99
        for res in all_results.values() for r in res["snr"]
    ))
    tick_positions = []
    tick_labels = []
    for s in all_snr_labels:
        if s == 99:
            tick_positions.append(25)
            tick_labels.append("Clean")
        else:
            tick_positions.append(s)
            tick_labels.append(f"{int(s)}")

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Noise Robustness Comparison")
    ax.legend(loc="lower left", framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")

    # Add shaded region for "high noise"
    ax.axvspan(-6, 5, alpha=0.05, color="red", label="_nolegend_")

    plt.tight_layout()
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=300, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Noise robustness evaluation")
    p.add_argument("--config", default="configs/default.yaml",
                   help="Base config YAML")
    p.add_argument("--exp-config", default="",
                   help="Experiment override YAML (single model mode)")
    p.add_argument("--checkpoint", default="",
                   help="Path to best.pt (single model mode)")
    p.add_argument("--exp-dir", default="",
                   help="Batch mode: directory with experiment YAMLs")
    p.add_argument("--auto-checkpoint", action="store_true",
                   help="Auto-find best.pt in latest run subdirectory")
    p.add_argument("--snr", type=float, nargs="+",
                   default=[-5, 0, 5, 10, 15, 20],
                   help="SNR levels in dB (default: -5 0 5 10 15 20)")
    p.add_argument("--output-dir", default="experiments/noise_robustness",
                   help="Output directory for results and plots")
    p.add_argument("--device", default="",
                   help="cuda / cpu override")
    p.add_argument("--workers", type=int, default=8)
    return p


def resolve_device(config_device: str, cli_device: str) -> torch.device:
    requested = cli_device.strip() or str(config_device).strip()
    if not requested or requested.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    return torch.device(requested)


def find_checkpoint(run_root: Path) -> Optional[Path]:
    """Find best.pt anywhere under *run_root* (searches up to 3 levels deep).

    Handles nested structures like::

        run_root/
          exp1_MSCA_VGG16/
            trial_seed42/
              checkpoints/
                best.pt

    Returns None if no checkpoint is found.
    """
    if not run_root.exists():
        return None

    # Direct path
    direct = run_root / "checkpoints" / "best.pt"
    if direct.exists():
        return direct

    # Recursive search (max 4 levels to avoid deep filesystem walks)
    for depth in range(1, 5):
        pattern = "/".join(["*"] * depth)
        for ckpt in sorted(run_root.glob(f"{pattern}/checkpoints/best.pt"), reverse=True):
            return ckpt

    return None


def main():
    args = build_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    snr_values: List[Optional[float]] = [None] + list(args.snr)  # None = clean

    # ── Resolve device ──
    base_config = load_yaml(args.config)
    device = resolve_device(base_config.get("device", "auto"), args.device)
    print(f"Device: {device}")

    # ── Resolve models to evaluate ──
    tasks: List[Tuple[str, Dict, str]] = []  # (name, config, ckpt_path)

    if args.exp_config:
        # Single model mode
        config = load_config(args.config, args.exp_config)
        name = config.get("experiment_name", config["model"]["name"])
        model_name = config["model"]["name"]

        if args.checkpoint:
            ckpt_path = args.checkpoint
        elif args.auto_checkpoint:
            run_root = Path(config.get("output", {}).get(
                "root_dir", f"experiments/exp1/{model_name}"))
            ckpt = find_checkpoint(run_root)
            if ckpt is None:
                print(f"[ERROR] No checkpoint found in {run_root}")
                sys.exit(1)
            ckpt_path = str(ckpt)
        else:
            print("[ERROR] Specify --checkpoint or --auto-checkpoint")
            sys.exit(1)

        tasks.append((name, config, ckpt_path))

    elif args.exp_dir:
        # Batch mode
        exp_path = Path(args.exp_dir)
        for yaml_file in sorted(exp_path.glob("*.yaml")):
            config = load_config(args.config, str(yaml_file))
            name = config.get("experiment_name", config["model"]["name"])
            model_name = config["model"]["name"]

            if args.auto_checkpoint:
                run_root = Path(config.get("output", {}).get(
                    "root_dir", f"experiments/exp1/{model_name}"))
                ckpt = find_checkpoint(run_root)
                if ckpt is None:
                    print(f"[SKIP] {name}: no checkpoint found in {run_root}")
                    continue
                tasks.append((name, config, str(ckpt)))
            else:
                print(f"[SKIP] {name}: use --checkpoint or --auto-checkpoint")
    else:
        print("[ERROR] Specify --exp-config + --checkpoint, or --exp-dir")
        sys.exit(1)

    if not tasks:
        print("[ERROR] No models to evaluate")
        sys.exit(1)

    print(f"Models to evaluate: {len(tasks)}")
    print(f"SNR levels: {['clean'] + [f'{s} dB' for s in args.snr]}")

    # ── Evaluate each model ──
    all_results: Dict[str, Dict[str, list]] = {}

    for model_label, config, ckpt_path in tasks:
        model_short = config["model"]["name"]
        print(f"\n{'='*60}\n  {model_label} ({model_short})\n{'='*60}")

        results = evaluate_model_noise(
            config, ckpt_path, snr_values, device, args.workers,
        )
        all_results[model_short] = results

        # Print summary table
        print(f"  {'SNR':>8s}  {'Acc %':>8s}  {'F1 %':>8s}  {'AUC %':>8s}")
        print(f"  {'-'*40}")
        for snr_s, acc, f1, auc in zip(
            results["snr"], results["accuracy"],
            results["f1"], results["auc"],
        ):
            print(f"  {snr_s:>8s}  {acc:8.2f}  {f1:8.2f}  {auc:8.2f}")

    # ── Save summary CSV ──
    csv_path = os.path.join(args.output_dir, "noise_robustness.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "snr", "accuracy", "f1", "auc", "loss"])
        for model_name, res in all_results.items():
            for snr_s, acc, f1, auc, loss in zip(
                res["snr"], res["accuracy"],
                res["f1"], res["auc"], res["loss"],
            ):
                writer.writerow([model_name, snr_s, acc, f1, auc, loss])
    print(f"\n📋 Summary CSV: {csv_path}")

    # ── Save JSON ──
    json_path = os.path.join(args.output_dir, "noise_robustness.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"📋 Summary JSON: {json_path}")

    # ── Plot ──
    plot_path = os.path.join(args.output_dir, "noise_robustness.png")
    plot_noise_curves(all_results, plot_path)
    print(f"📈 Plot: {plot_path}")


if __name__ == "__main__":
    main()
