#!/usr/bin/env python3
"""
Hyperparameter Sensitivity Analysis for AW-DPCNN
=================================================
Sweeps key PCNN hyperparameters and evaluates classification accuracy
on a fixed test set using a pre‑trained checkpoint.

For each parameter combination, raw test‑set windows are re‑fused
on‑the‑fly (Mel + GADF + AW‑DPCNN) with the specified parameters,
then passed through the frozen classifier.

Parameters swept
----------------
  γ  — contrast amplification factor   {1, 2, 4, 8, 10, 20}
  N  — PCNN iteration count            {5, 8, 10, 15, 20}
  α  — decay coefficient (α_L = α_T)   {0.0001, 0.001, 0.01}

Output
------
  experiments/hyperparameter_sensitivity/
      sensitivity_gamma.csv  /  sensitivity_N.csv  /  sensitivity_alpha.csv
      sensitivity_gamma.png  /  sensitivity_N.png  /  sensitivity_alpha.png
      sensitivity_summary.json

Usage::

    # Full sweep (requires a trained checkpoint)
    python scripts/hyperparameter_sensitivity.py \\
        --config configs/default.yaml \\
        --checkpoint PATH/TO/best.pt

    # With auto-checkpoint discovery
    python scripts/hyperparameter_sensitivity.py \\
        --config configs/default.yaml \\
        --exp-config experiments/exp1/MSCA_VGG16.yaml \\
        --auto-checkpoint
"""

import argparse
import csv
import json
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.io import wavfile
from tqdm import tqdm

warnings.filterwarnings("ignore", message=".*TripleDES.*")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_CWRU_dataset import (  # noqa: E402
    aw_dpcnn_fusion_color,
    generate_gadf_image,
    generate_mel_image,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models import build_model
from src.utils.config import load_config, load_yaml


# ═══════════════════════════════════════════════════════════════════
#  Sweep definitions
# ═══════════════════════════════════════════════════════════════════

GAMMA_VALUES = [1, 2, 4, 8, 10, 20]
N_VALUES = [5, 8, 10, 15, 20]
ALPHA_VALUES = [0.0001, 0.001, 0.01]

# Default fixed parameters (from paper)
DEFAULT_GAMMA = 4.0
DEFAULT_N = 20
DEFAULT_ALPHA = 0.001

IMG_SIZE = 224
CMAP = cv2.COLORMAP_VIRIDIS
BATCH_SIZE = 64


# ═══════════════════════════════════════════════════════════════════
#  Raw window extractor  (avoids storing intermediate images)
# ═══════════════════════════════════════════════════════════════════

def _load_test_windows(raw_dir: str, metadata_path: Optional[str],
                       win_len: int, hop_len: int,
                       max_windows: Optional[int] = None
                       ) -> Tuple[List[Tuple[np.ndarray, int, int]], List[str]]:
    """Extract raw windows belonging to the test split.

    If *metadata_path* is provided (a CSV with columns source_file, split),
    only windows from test‑split source files are loaded.  Otherwise all
    .wav files under *raw_dir* are used.

    Returns (windows, class_names).
    """
    raw_root = Path(raw_dir)

    # ── Determine which source files belong to test ──
    test_sources: Optional[set] = None
    class_names: List[str] = []
    if metadata_path and Path(metadata_path).exists():
        import csv
        test_sources = set()
        source_to_class = {}
        with open(metadata_path, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("split", "").lower() == "test":
                    test_sources.add(row["source_file"])
                    source_to_class[row["source_file"]] = row.get("class_label", "")
        if not test_sources:
            print("[WARN] metadata.csv has no test-split rows; using all files")
            test_sources = None
        else:
            # Build class_names from metadata
            class_names = sorted(set(source_to_class.values()))

    # ── Build class index ──
    if not class_names:
        class_names = sorted(
            d.name for d in raw_root.iterdir()
            if d.is_dir() and not d.name.startswith(".") and
            not d.name.startswith("train") and not d.name.startswith("val") and
            not d.name.startswith("test")
        )
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    # ── Scan for .wav files ──
    windows = []
    for cls_dir in sorted(raw_root.iterdir()):
        if not cls_dir.is_dir() or cls_dir.name.startswith("."):
            continue
        if cls_dir.name.startswith("train") or cls_dir.name.startswith("val") or \
           cls_dir.name.startswith("test"):
            continue

        cls_name = cls_dir.name
        if cls_name not in class_to_idx:
            continue
        label_idx = class_to_idx[cls_name]

        for wav_path in sorted(cls_dir.glob("*.wav")):
            fname = wav_path.name
            if test_sources is not None and fname not in test_sources:
                continue  # skip files not in test split

            sr, data = wavfile.read(str(wav_path))
            if data.ndim > 1:
                data = data.mean(axis=1)
            data = data.astype(np.float32)

            for start in range(0, len(data) - win_len + 1, hop_len):
                window = data[start:start + win_len]
                windows.append((window, sr, label_idx))
                if max_windows and len(windows) >= max_windows:
                    return windows, class_names

    return windows, class_names


def _fuse_window(window: np.ndarray, sr: int,
                 gamma: float, n_iter: int, alpha: float,
                 cmap: int = cv2.COLORMAP_VIRIDIS) -> np.ndarray:
    """Generate Mel + GADF + AW‑DPCNN fusion for one window."""
    mel = generate_mel_image(
        window, sr, n_fft=2048, hop_length=256, n_mels=128,
        fmax=8000, img_size=IMG_SIZE, cmap=cmap,
    )
    gadf = generate_gadf_image(window, img_size=IMG_SIZE, cmap=cmap)
    fused = aw_dpcnn_fusion_color(mel, gadf,
                                   n_iter=n_iter, gamma=gamma,
                                   alpha_L=alpha, alpha_T=alpha)

    import torchvision.transforms as T
    tf = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return tf(fused)


# ═══════════════════════════════════════════════════════════════════
#  Evaluation
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def _evaluate_fused(model, fused_tensors, labels, device):
    """Evaluate model on a list of pre‑fused tensors."""
    model.eval()
    correct, total = 0, 0
    # Process in batches
    for i in range(0, len(fused_tensors), BATCH_SIZE):
        batch = torch.stack(fused_tensors[i:i + BATCH_SIZE]).to(device)
        y = torch.tensor(labels[i:i + BATCH_SIZE], device=device)
        out = model(batch)
        preds = out.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / total


def _sweep_parameter(
    windows, labels, sr, model, device,
    param_name: str, param_values: list,
    fixed_gamma: float, fixed_n: int, fixed_alpha: float,
    max_samples: int = 500,
) -> Dict[str, list]:
    """Sweep one parameter, keeping others fixed.

    Returns dict with keys: param, accuracy.
    """
    # Subsample for speed
    indices = np.random.RandomState(42).choice(
        len(windows), min(len(windows), max_samples), replace=False,
    )
    accuracies = []

    for pv in tqdm(param_values, desc=f"  {param_name}", ncols=70):
        gamma = pv if param_name == "gamma" else fixed_gamma
        n_iter = pv if param_name == "N" else fixed_n
        alpha = pv if param_name == "alpha_LT" else fixed_alpha

        fused = []
        sub_labels = []
        for idx in indices:
            win, _, lbl = windows[idx]
            try:
                ft = _fuse_window(win, sr, gamma=gamma, n_iter=n_iter, alpha=alpha)
                fused.append(ft)
                sub_labels.append(lbl)
            except Exception:
                continue

        if fused:
            acc = _evaluate_fused(model, fused, sub_labels, device)
        else:
            acc = 0.0
        accuracies.append(round(acc * 100, 2))

    return {"param": param_values, "accuracy": accuracies}


# ═══════════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════════

def _plot_sensitivity(param_name: str, param_values: list,
                      accuracies: list, save_path: str):
    """Plot accuracy vs. parameter value."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "stix",
        "font.size": 13, "axes.labelsize": 14, "axes.titlesize": 15,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "figure.dpi": 300, "axes.linewidth": 0.8,
    })

    labels = {"gamma": "γ (contrast amplification)", "N": "N (PCNN iterations)",
              "alpha_LT": "α_L = α_T (decay coefficient)"}

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(param_values, accuracies, marker="o", lw=2, markersize=8, color="steelblue")
    ax.set_xlabel(labels.get(param_name, param_name))
    ax.set_ylabel("Accuracy (%)")
    ax.grid(True, alpha=0.3, linestyle="--")

    # Highlight best
    best_idx = np.argmax(accuracies)
    ax.scatter([param_values[best_idx]], [accuracies[best_idx]],
               color="darkred", s=120, zorder=5,
               label=f"Best: {accuracies[best_idx]:.1f}%")

    # Mark default
    defaults = {"gamma": DEFAULT_GAMMA, "N": DEFAULT_N, "alpha_LT": DEFAULT_ALPHA}
    default_val = defaults.get(param_name)
    if default_val is not None and default_val in param_values:
        idx = param_values.index(default_val)
        ax.scatter([default_val], [accuracies[idx]],
                   color="orange", s=100, zorder=5,
                   marker="s", label=f"Default: {accuracies[idx]:.1f}%")

    ax.legend()
    plt.tight_layout()
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=300, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════

def find_checkpoint(run_root: Path, model_name: str) -> Optional[Path]:
    if not run_root.exists():
        return None
    direct = run_root / "checkpoints" / "best.pt"
    if direct.exists():
        return direct
    try:
        subdirs = sorted([d for d in run_root.iterdir() if d.is_dir()], reverse=True)
    except (FileNotFoundError, PermissionError):
        return None
    for d in subdirs:
        ckpt = d / "checkpoints" / "best.pt"
        if ckpt.exists():
            return ckpt
    return None


def main():
    parser = argparse.ArgumentParser(description="AW-DPCNN hyperparameter sensitivity")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--exp-config", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--auto-checkpoint", action="store_true")
    parser.add_argument("--data-dir", default="raw-data/transformer-five",
                        help="Raw .wav source directory (class sub‑folders)")
    parser.add_argument("--metadata", default="datasets/transformer-five/metadata.csv",
                        help="metadata.csv for identifying test‑split source files")
    parser.add_argument("--win-len", type=int, default=8192)
    parser.add_argument("--hop-len", type=int, default=4096)
    parser.add_argument("--max-samples", type=int, default=500,
                        help="Max test windows per sweep (default: 500)")
    parser.add_argument("--output-dir", default="experiments/hyperparameter_sensitivity")
    parser.add_argument("--device", default="")
    parser.add_argument("--sweep", default="all",
                        help="Parameter to sweep: gamma, N, alpha_LT, all")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Resolve device ──
    base_config = load_yaml(args.config)
    requested = args.device.strip() or str(base_config.get("device", "auto")).strip()
    if not requested or requested.lower() == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested)
    print(f"Device: {device}")

    # ── Resolve config & checkpoint ──
    if args.exp_config:
        config = load_config(args.config, args.exp_config)
    else:
        config = base_config
    config["model"]["num_classes"] = 5  # transformer-five has 5 classes

    if args.checkpoint:
        ckpt_path = args.checkpoint
    elif args.auto_checkpoint and args.exp_config:
        model_name = config["model"]["name"]
        run_root = Path(config.get("output", {}).get(
            "root_dir", f"experiments/experiment_result/exp1/{model_name}"))
        ckpt = find_checkpoint(run_root, model_name)
        if ckpt is None:
            print(f"[ERROR] No checkpoint in {run_root}")
            sys.exit(1)
        ckpt_path = str(ckpt)
    else:
        print("[ERROR] Specify --checkpoint or --exp-config --auto-checkpoint")
        sys.exit(1)

    # ── Load model ──
    model = build_model(config).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"Model: {config['model']['name']}  Checkpoint: {ckpt_path}")

    # ── Load test windows ──
    print(f"Loading test windows from {args.data_dir}...")
    windows, class_names = _load_test_windows(
        args.data_dir, args.metadata, args.win_len, args.hop_len,
        max_windows=args.max_samples * 2,
    )
    labels = [lbl for _, _, lbl in windows]
    sr = windows[0][1] if windows else 44100
    print(f"Loaded {len(windows)} test windows across {len(class_names)} classes")

    # ── Sweep ──
    all_results = {}

    sweeps = {
        "gamma": (GAMMA_VALUES, DEFAULT_GAMMA, DEFAULT_N, DEFAULT_ALPHA),
        "N": (N_VALUES, DEFAULT_GAMMA, DEFAULT_N, DEFAULT_ALPHA),
        "alpha_LT": (ALPHA_VALUES, DEFAULT_GAMMA, DEFAULT_N, DEFAULT_ALPHA),
    }

    to_sweep = list(sweeps) if args.sweep == "all" else [args.sweep]

    for param_name in to_sweep:
        print(f"\n{'='*60}\n  Sweeping {param_name}\n{'='*60}")
        param_values, fg, fn, fa = sweeps[param_name]

        # Use correct defaults for non-swept params
        if param_name == "gamma":
            fg = None  # will be overridden
        elif param_name == "N":
            fn = None
        elif param_name == "alpha_LT":
            fa = None

        res = _sweep_parameter(
            windows, labels, sr, model, device,
            param_name, param_values,
            fixed_gamma=DEFAULT_GAMMA if fg is None else fg,
            fixed_n=DEFAULT_N if fn is None else fn,
            fixed_alpha=DEFAULT_ALPHA if fa is None else fa,
            max_samples=args.max_samples,
        )
        all_results[param_name] = res

        # Print table
        print(f"  {'Value':>12s}  {'Acc %':>8s}")
        print(f"  {'-'*20}")
        for pv, acc in zip(res["param"], res["accuracy"]):
            mark = " ← default" if (
                (param_name == "gamma" and pv == DEFAULT_GAMMA) or
                (param_name == "N" and pv == DEFAULT_N) or
                (param_name == "alpha_LT" and abs(pv - DEFAULT_ALPHA) < 1e-8)
            ) else ""
            print(f"  {pv:>12}  {acc:8.2f}{mark}")

        # Plot
        plot_path = os.path.join(args.output_dir, f"sensitivity_{param_name}.png")
        _plot_sensitivity(param_name, res["param"], res["accuracy"], plot_path)
        print(f"  Plot: {plot_path}")

    # ── Save CSV ──
    csv_path = os.path.join(args.output_dir, "sensitivity_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["parameter", "value", "accuracy"])
        for param_name, res in all_results.items():
            for pv, acc in zip(res["param"], res["accuracy"]):
                writer.writerow([param_name, pv, acc])
    print(f"\n📋 CSV: {csv_path}")

    # ── Save JSON ──
    json_path = os.path.join(args.output_dir, "sensitivity_summary.json")
    json_serializable = {}
    for k, v in all_results.items():
        json_serializable[k] = {
            "param": [str(x) if isinstance(x, float) else x for x in v["param"]],
            "accuracy": v["accuracy"],
        }
    with open(json_path, "w") as f:
        json.dump(json_serializable, f, indent=2)
    print(f"📋 JSON: {json_path}")


if __name__ == "__main__":
    main()
