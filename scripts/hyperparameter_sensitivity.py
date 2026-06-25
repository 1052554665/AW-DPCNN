#!/usr/bin/env python3
"""
Hyperparameter Sensitivity Analysis for AW-DPCNN
=================================================
Sweeps key PCNN hyperparameters and evaluates classification accuracy
using a pre‑trained checkpoint on the test split of ``datasets/cwru_de``.

Raw .mat signals from the CWRU dataset are re‑fused on‑the‑fly (Mel +
GADF + AW‑DPCNN) with the specified parameters, then passed through the
frozen classifier.

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

    # Auto‑discover checkpoint from exp config
    python scripts/hyperparameter_sensitivity.py \
        --exp-config experiments/exp1/MSCA_VGG16.yaml \
        --auto-checkpoint

    # Explicit checkpoint
    python scripts/hyperparameter_sensitivity.py \
        --exp-config experiments/exp1/MSCA_VGG16.yaml \
        --checkpoint PATH/TO/best.pt

    # Sweep a single parameter
    python scripts/hyperparameter_sensitivity.py \
        --exp-config experiments/exp1/MSCA_VGG16.yaml \
        --auto-checkpoint --sweep gamma
"""

import argparse
import csv
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import loadmat
from tqdm import tqdm

warnings.filterwarnings("ignore", message=".*TripleDES.*")

# ── Project imports ──
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_PROJECT_ROOT))

from build_cwru_dataset import (  # noqa: E402
    aw_dpcnn_fusion_color,
    generate_gadf_image,
    generate_mel_image,
)
from src.models import build_model  # noqa: E402
from src.utils.config import load_config, load_yaml  # noqa: E402


# ═══════════════════════════════════════════════════════════════════
#  Sweep definitions
# ═══════════════════════════════════════════════════════════════════

GAMMA_VALUES   = [1, 2, 4, 8, 10, 20]
N_VALUES       = [5, 8, 10, 15, 20]
ALPHA_VALUES   = [0.0001, 0.001, 0.01]

DEFAULT_GAMMA  = 4.0
DEFAULT_N      = 20
DEFAULT_ALPHA  = 0.001

IMG_SIZE       = 224
BATCH_SIZE     = 64


# ═══════════════════════════════════════════════════════════════════
#  CWRU .mat file helpers
# ═══════════════════════════════════════════════════════════════════

CWRU_CLASS_MAP = {
    "BF007": ("12k_Drive_End_Bearing_Fault_Data/B/007",       "DE_time"),
    "BF014": ("12k_Drive_End_Bearing_Fault_Data/B/014",       "DE_time"),
    "BF021": ("12k_Drive_End_Bearing_Fault_Data/B/021",       "DE_time"),
    "IF007": ("12k_Drive_End_Bearing_Fault_Data/IR/007",      "DE_time"),
    "IF014": ("12k_Drive_End_Bearing_Fault_Data/IR/014",      "DE_time"),
    "IF021": ("12k_Drive_End_Bearing_Fault_Data/IR/021",      "DE_time"),
    "OF007": ("12k_Drive_End_Bearing_Fault_Data/OR/007/@6",   "DE_time"),
    "OF014": ("12k_Drive_End_Bearing_Fault_Data/OR/014/@6",   "DE_time"),
    "OF021": ("12k_Drive_End_Bearing_Fault_Data/OR/021/@6",   "DE_time"),
    "Normal": ("Normal",                                       "DE_time"),
}

CWRU_ROOT = Path("raw-data/CWRU-dataset")

# Mel params for 12 kHz CWRU
CWRU_SR       = 12000
CWRU_N_FFT    = 1024
CWRU_HOP_LEN  = 256
CWRU_N_MELS   = 128
CWRU_FMAX     = 6000


def _load_mat_signal(mat_path: str, sensor_key: str) -> np.ndarray:
    """Extract a sensor signal from a CWRU .mat file."""
    mat = loadmat(mat_path)
    for key in mat.keys():
        if sensor_key in key:
            sig = mat[key].squeeze().astype(np.float32)
            if sig.ndim != 1:
                sig = sig.ravel()
            return sig
    raise ValueError(f"No '{sensor_key}' found in {mat_path}")


def _collect_cwru_test_windows(
    dataset_dir: str,
    win_len: int = 2048,
    hop_len: int = 1024,
    max_windows: int = 500,
) -> tuple:
    """Collect raw CWRU windows for the *test* split.

    Reads the test‑split file list from ``datasets/cwru_de/test/``
    (ImageFolder structure), maps each PNG back to its source .mat file,
    and segments the raw signal into windows.

    Returns (windows, class_names).
    """
    test_root = Path(dataset_dir) / "test"
    if not test_root.exists():
        raise FileNotFoundError(f"Test split not found: {test_root}")

    class_names = sorted(
        d.name for d in test_root.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    print(f"  Classes: {class_names}")

    # Build set of source .mat stems from test PNG filenames
    # PNG: "<stem>_<idx>.png" → stem
    source_files: Dict[str, List[str]] = {}
    for cls_name in class_names:
        cls_dir = test_root / cls_name
        mat_stems = set()
        for p in sorted(cls_dir.glob("*.png")):
            parts = p.stem.rsplit("_", 1)
            stem = parts[0] if (len(parts) == 2 and parts[1].isdigit()) else p.stem
            mat_stems.add(stem)
        source_files[cls_name] = sorted(mat_stems)

    total_sources = sum(len(v) for v in source_files.values())
    print(f"  Unique source files: {total_sources}")

    # Load raw signals and segment into windows
    windows = []
    for cls_name in class_names:
        label_idx = class_to_idx[cls_name]
        cwru_info = CWRU_CLASS_MAP.get(cls_name)
        if cwru_info is None:
            print(f"  [WARN] No CWRU mapping for '{cls_name}'")
            continue
        sub_path, sensor_key = cwru_info

        mat_dir = CWRU_ROOT / sub_path
        if not mat_dir.exists():
            print(f"  [WARN] Directory not found: {mat_dir}")
            continue

        for stem in source_files[cls_name]:
            mat_path = mat_dir / f"{stem}.mat"
            if not mat_path.exists():
                continue
            try:
                signal = _load_mat_signal(str(mat_path), sensor_key)
            except Exception as exc:
                print(f"  [WARN] {mat_path}: {exc}")
                continue

            if win_len <= 0 or win_len >= len(signal):
                windows.append((signal, CWRU_SR, label_idx))
            else:
                for start in range(0, len(signal) - win_len + 1, hop_len):
                    windows.append((signal[start:start + win_len],
                                    CWRU_SR, label_idx))

            if max_windows and len(windows) >= max_windows:
                return windows, class_names

    return windows, class_names


# ═══════════════════════════════════════════════════════════════════
#  On‑the‑fly fusion
# ═══════════════════════════════════════════════════════════════════

def _fuse_window(window: np.ndarray, sr: int,
                 gamma: float, n_iter: int, alpha: float) -> torch.Tensor:
    """Mel + GADF + AW‑DPCNN → normalised tensor."""
    mel = generate_mel_image(
        window, sr,
        n_fft=CWRU_N_FFT, hop_length=CWRU_HOP_LEN,
        n_mels=CWRU_N_MELS, fmax=CWRU_FMAX,
        img_size=IMG_SIZE, cmap=cv2.COLORMAP_VIRIDIS,
    )
    gadf = generate_gadf_image(window, img_size=IMG_SIZE,
                                cmap=cv2.COLORMAP_VIRIDIS)
    fused = aw_dpcnn_fusion_color(
        mel, gadf, n_iter=n_iter, gamma=gamma,
        alpha_L=alpha, alpha_T=alpha,
    )

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
def _evaluate_fused(model, fused_tensors: list, labels: list,
                    device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for i in range(0, len(fused_tensors), BATCH_SIZE):
        batch = torch.stack(fused_tensors[i:i + BATCH_SIZE]).to(device)
        y = torch.tensor(labels[i:i + BATCH_SIZE], device=device)
        correct += (model(batch).argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return correct / total if total > 0 else 0.0


def _sweep_parameter(
    windows, labels, sr, model, device,
    param_name: str, param_values: list,
    fixed_gamma: float, fixed_n: int, fixed_alpha: float,
    max_samples: int = 500,
) -> Dict[str, list]:
    rng = np.random.RandomState(42)
    n_sample = min(len(windows), max_samples)
    indices = rng.choice(len(windows), n_sample, replace=False)

    accuracies = []
    for pv in tqdm(param_values, desc=f"  {param_name}", ncols=70):
        gamma  = pv if param_name == "gamma"    else fixed_gamma
        n_iter = pv if param_name == "N"        else fixed_n
        alpha  = pv if param_name == "alpha_LT" else fixed_alpha

        fused, sub_labels = [], []
        for idx in indices:
            win, _, lbl = windows[idx]
            try:
                fused.append(_fuse_window(win, sr, gamma=gamma,
                                           n_iter=n_iter, alpha=alpha))
                sub_labels.append(lbl)
            except Exception:
                continue

        acc = _evaluate_fused(model, fused, sub_labels, device) if fused else 0.0
        accuracies.append(round(acc * 100, 2))

    return {"param": param_values, "accuracy": accuracies}


# ═══════════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════════

def _plot_sensitivity(param_name: str, param_values: list,
                      accuracies: list, save_path: str):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "stix",
        "font.size": 13, "axes.labelsize": 14, "axes.titlesize": 15,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "figure.dpi": 300, "axes.linewidth": 0.8,
    })
    labels = {
        "gamma": r"$\gamma$ (contrast amplification)",
        "N": r"$N$ (PCNN iterations)",
        "alpha_LT": r"$\alpha_L = \alpha_T$ (decay coefficient)",
    }
    defaults = {"gamma": DEFAULT_GAMMA, "N": DEFAULT_N, "alpha_LT": DEFAULT_ALPHA}

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(param_values, accuracies, marker="o", lw=2, markersize=8,
            color="steelblue")
    ax.set_xlabel(labels.get(param_name, param_name))
    ax.set_ylabel("Accuracy (%)")
    ax.grid(True, alpha=0.3, linestyle="--")

    best_idx = np.argmax(accuracies)
    ax.scatter([param_values[best_idx]], [accuracies[best_idx]],
               color="darkred", s=120, zorder=5,
               label=f"Best: {accuracies[best_idx]:.1f}%")

    default_val = defaults.get(param_name)
    if default_val is not None and default_val in param_values:
        idx = param_values.index(default_val)
        ax.scatter([default_val], [accuracies[idx]],
                   color="orange", s=100, zorder=5,
                   marker="s", label=f"Default: {accuracies[idx]:.1f}%")

    ax.legend()
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════
#  Checkpoint discovery
# ═══════════════════════════════════════════════════════════════════

def find_checkpoint(run_root: Path) -> Optional[Path]:
    """Recursively search for best.pt."""
    if not run_root.exists():
        return None
    direct = run_root / "checkpoints" / "best.pt"
    if direct.exists():
        return direct
    for pattern in ["**/checkpoints/best.pt", "**/best.pt"]:
        candidates = list(run_root.glob(pattern))
        if candidates:
            return sorted(candidates, key=lambda p: p.stat().st_mtime,
                          reverse=True)[0]
    return None


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="AW-DPCNN hyperparameter sensitivity (CWRU dataset)",
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--exp-config", default="",
                        help="Experiment override YAML")
    parser.add_argument("--checkpoint", default="",
                        help="Explicit path to best.pt")
    parser.add_argument("--auto-checkpoint", action="store_true",
                        help="Auto‑discover checkpoint from exp‑config")
    parser.add_argument("--dataset-dir", default="datasets/cwru_de",
                        help="ImageFolder dataset root (for test split list)")
    parser.add_argument("--win-len", type=int, default=2048)
    parser.add_argument("--hop-len", type=int, default=1024)
    parser.add_argument("--max-samples", type=int, default=500,
                        help="Max test windows per sweep")
    parser.add_argument("--output-dir",
                        default="experiments/hyperparameter_sensitivity")
    parser.add_argument("--device", default="")
    parser.add_argument("--sweep", default="all",
                        help="Parameter to sweep: gamma, N, alpha_LT, all")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Device ──
    base_config = load_yaml(args.config)
    requested = (args.device.strip() or
                 str(base_config.get("device", "auto")).strip())
    if not requested or requested.lower() == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested)
    print(f"Device: {device}")

    # ── Config ──
    if args.exp_config:
        config = load_config(args.config, args.exp_config)
    else:
        config = base_config
    num_classes = int(config["model"].get("num_classes", 10))
    config["model"]["num_classes"] = num_classes

    # ── Checkpoint ──
    if args.checkpoint:
        ckpt_path = args.checkpoint
    elif args.auto_checkpoint and args.exp_config:
        run_root = Path(config.get("output", {}).get(
            "root_dir", "experiments/experiment_result/exp1"))
        ckpt = find_checkpoint(run_root)
        if ckpt is None:
            print(f"[ERROR] No checkpoint found in {run_root}")
            sys.exit(1)
        ckpt_path = str(ckpt)
    else:
        print("[ERROR] Specify --checkpoint or --exp-config --auto-checkpoint")
        sys.exit(1)

    # ── Load model ──
    model = build_model(config).to(device)
    state = torch.load(ckpt_path, map_location=device)
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    print(f"Model: {config['model']['name']}  |  "
          f"num_classes={num_classes}  |  ckpt={ckpt_path}")

    # ── Load CWRU test windows ──
    print(f"\nCollecting test windows from {args.dataset_dir} ...")
    windows, class_names = _collect_cwru_test_windows(
        args.dataset_dir, args.win_len, args.hop_len,
        max_windows=args.max_samples * 2,
    )
    labels = [lbl for _, _, lbl in windows]
    print(f"Loaded {len(windows)} test windows "
          f"across {len(class_names)} classes")

    if len(windows) == 0:
        print("[ERROR] No test windows found — check --dataset-dir")
        sys.exit(1)

    # ── Sweep ──
    all_results = {}
    sweeps = {
        "gamma":    (GAMMA_VALUES,   DEFAULT_GAMMA, DEFAULT_N, DEFAULT_ALPHA),
        "N":        (N_VALUES,       DEFAULT_GAMMA, DEFAULT_N, DEFAULT_ALPHA),
        "alpha_LT": (ALPHA_VALUES,   DEFAULT_GAMMA, DEFAULT_N, DEFAULT_ALPHA),
    }
    to_sweep = list(sweeps) if args.sweep == "all" else [args.sweep]

    for param_name in to_sweep:
        print(f"\n{'='*60}\n  Sweeping {param_name}\n{'='*60}")
        param_values, fg, fn, fa = sweeps[param_name]

        if param_name == "gamma":
            fg = None
        elif param_name == "N":
            fn = None
        elif param_name == "alpha_LT":
            fa = None

        res = _sweep_parameter(
            windows, labels, CWRU_SR, model, device,
            param_name, param_values,
            fixed_gamma=DEFAULT_GAMMA if fg is None else fg,
            fixed_n=DEFAULT_N if fn is None else fn,
            fixed_alpha=DEFAULT_ALPHA if fa is None else fa,
            max_samples=args.max_samples,
        )
        all_results[param_name] = res

        # Table
        print(f"  {'Value':>12s}  {'Acc %':>8s}")
        print(f"  {'-'*20}")
        for pv, acc in zip(res["param"], res["accuracy"]):
            is_default = (
                (param_name == "gamma" and pv == DEFAULT_GAMMA) or
                (param_name == "N" and pv == DEFAULT_N) or
                (param_name == "alpha_LT" and abs(pv - DEFAULT_ALPHA) < 1e-8)
            )
            print(f"  {pv:>12}  {acc:8.2f}{' ← default' if is_default else ''}")

        # Plot
        plot_path = os.path.join(args.output_dir,
                                 f"sensitivity_{param_name}.png")
        _plot_sensitivity(param_name, res["param"], res["accuracy"], plot_path)
        print(f"  Plot: {plot_path}")

    # ── Save ──
    csv_path = os.path.join(args.output_dir, "sensitivity_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["parameter", "value", "accuracy"])
        for param_name, res in all_results.items():
            for pv, acc in zip(res["param"], res["accuracy"]):
                writer.writerow([param_name, pv, acc])
    print(f"\nCSV: {csv_path}")

    json_path = os.path.join(args.output_dir, "sensitivity_summary.json")
    with open(json_path, "w") as f:
        json.dump({
            k: {"param": [str(x) if isinstance(x, float) else x
                          for x in v["param"]],
                "accuracy": v["accuracy"]}
            for k, v in all_results.items()
        }, f, indent=2)
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
