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

# 12k drive-end (default — same as before)
python scripts/hyperparameter_sensitivity.py \
    --exp-config experiments/exp1/MSCA_VGG16.yaml \
    --auto-checkpoint --trial trial_seed42 --dataset 12k_de --max-samples 500

# 12k fan-end
python scripts/hyperparameter_sensitivity.py \
    --exp-config experiments/exp1/MSCA_VGG16.yaml \
    --auto-checkpoint --trial trial_seed42 --dataset 12k_fe --max-samples 500

# 48k drive-end
python scripts/hyperparameter_sensitivity.py \
    --exp-config experiments/exp1/MSCA_VGG16.yaml \
    --auto-checkpoint --trial trial_seed42 --dataset 48k_de --max-samples 500

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
    generate_stft_image,
    TF_GENERATORS,
)
from src.models import build_model  # noqa: E402
from src.utils.config import load_config, load_yaml  # noqa: E402
from src.utils.dataset_registry import (  # noqa: E402
    DATASET_KEYS,
    get_dataset_config,
)


# ═══════════════════════════════════════════════════════════════════
#  Sweep definitions
# ═══════════════════════════════════════════════════════════════════

GAMMA_VALUES   = [1, 2, 4, 8, 10, 20]
N_VALUES       = [5, 8, 10, 15, 20, 25]
ALPHA_VALUES   = [0.0001, 0.001, 0.01, 0.1]

DEFAULT_GAMMA  = 10.0   # match paper default (γ=10 for full AW‑DPCNN)
DEFAULT_N      = 20
DEFAULT_ALPHA  = 0.001

IMG_SIZE       = 224


# ═══════════════════════════════════════════════════════════════════
#  CWRU .mat file helpers
# ═══════════════════════════════════════════════════════════════════

def _build_cwru_class_map(src_dir: str, sensor_key: str) -> dict:
    """Build a CWRU class→(subpath, sensor_key) map for a given dataset.

    Uses the positional ``@6`` suffix for outer-race faults (standard
    CWRU convention for the centred load-zone position).
    """
    return {
        "BF007": (f"{src_dir}/B/007",        sensor_key),
        "BF014": (f"{src_dir}/B/014",        sensor_key),
        "BF021": (f"{src_dir}/B/021",        sensor_key),
        "IF007": (f"{src_dir}/IR/007",       sensor_key),
        "IF014": (f"{src_dir}/IR/014",       sensor_key),
        "IF021": (f"{src_dir}/IR/021",       sensor_key),
        "OF007": (f"{src_dir}/OR/007/@6",    sensor_key),
        "OF014": (f"{src_dir}/OR/014/@6",    sensor_key),
        "OF021": (f"{src_dir}/OR/021/@6",    sensor_key),
        "Normal": ("Normal",                  sensor_key),
    }


# Default class map for 12k drive-end (backward‑compatible fallback)
CWRU_CLASS_MAP = _build_cwru_class_map(
    "12k_Drive_End_Bearing_Fault_Data", "DE_time")

CWRU_ROOT = Path("raw-data/CWRU-dataset")

# Mel / signal params for 12 kHz CWRU — overridable via CLI or --dataset
CWRU_SR       = 12000
CWRU_N_FFT    = 1024
CWRU_HOP_LEN  = 256
CWRU_N_MELS   = 128
CWRU_FMAX     = 6000
CWRU_WIN_LEN  = 2048
CWRU_SEG_HOP  = 1024


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
    class_map: dict = None,
) -> tuple:
    """Collect raw CWRU windows for the *test* split.

    Reads the test‑split file list from ``datasets/cwru_de/test/``
    (ImageFolder structure), maps each PNG back to its source .mat file,
    and segments the raw signal into windows.

    Returns (windows, class_names).
    """
    if class_map is None:
        class_map = CWRU_CLASS_MAP

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
        cwru_info = class_map.get(cls_name)
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

def _precompute_mel_gadf(windows: list, sr: int,
                          n_fft: int, hop_len: int, n_mels: int,
                          fmax: int, tf_method: str = "stft") -> list:
    """Pre‑compute TF + GADF BGR images for all windows (cached — independent of γ, N, α)."""
    import torchvision.transforms as T
    tf = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    pairs = []
    desc = f"  Pre‑computing {tf_method.upper()}+GADF"
    for win, _, _ in tqdm(windows, desc=desc, ncols=80):
        if tf_method == "stft":
            tf_img = generate_stft_image(
                win, sr, n_fft=n_fft, hop_length=hop_len,
                img_size=IMG_SIZE, cmap=cv2.COLORMAP_VIRIDIS,
            )
        else:
            tf_img = generate_mel_image(
                win, sr,
                n_fft=n_fft, hop_length=hop_len,
                n_mels=n_mels, fmax=fmax,
                img_size=IMG_SIZE, cmap=cv2.COLORMAP_VIRIDIS,
            )
        gadf = generate_gadf_image(win, img_size=IMG_SIZE,
                                   cmap=cv2.COLORMAP_VIRIDIS)
        # Store as float32 tensors (normalised) for faster re‑fusion
        tf_t  = tf(tf_img)
        gadf_t = tf(gadf)
        pairs.append((tf_t, gadf_t))
    return pairs


def _fuse_precomputed(mel_t: torch.Tensor, gadf_t: torch.Tensor,
                      gamma: float, n_iter: int, alpha: float) -> torch.Tensor:
    """Run AW‑DPCNN fusion on pre‑computed Mel+GADF tensors → normalised tensor."""
    import torchvision.transforms as T
    # Convert back to BGR uint8 for aw_dpcnn_fusion_color (OpenCV-based)
    denorm = T.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225],
    )
    mel_bgr  = (denorm(mel_t).permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    gadf_bgr = (denorm(gadf_t).permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)

    fused = aw_dpcnn_fusion_color(
        mel_bgr, gadf_bgr, n_iter=n_iter, gamma=gamma,
        alpha_L=alpha, alpha_T=alpha,
    )
    # Re-apply normalization for model input
    tf = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return tf(fused)


# ═══════════════════════════════════════════════════════════════════
#  Evaluation
# ═══════════════════════════════════════════════════════════════════

@torch.inference_mode()
def _evaluate_fused(model, fused_tensors: list, labels: list,
                    device: torch.device) -> tuple:
    """Return (accuracy, macro_recall)."""
    model.eval()
    all_preds, all_labels = [], []
    bs = 128  # larger batch for RTX 5090
    for i in range(0, len(fused_tensors), bs):
        batch = torch.stack(fused_tensors[i:i + bs]).to(device)
        y = torch.tensor(labels[i:i + bs], device=device)
        preds = model(batch).argmax(dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(y.cpu())
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    acc = (all_preds == all_labels).float().mean().item()

    # Macro-averaged recall
    classes = all_labels.unique()
    recalls = []
    for c in classes:
        tp = ((all_preds == c) & (all_labels == c)).sum().item()
        fn = ((all_preds != c) & (all_labels == c)).sum().item()
        recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    rec = float(np.mean(recalls)) if recalls else 0.0

    return acc, rec


def _sweep_parameter(
    mel_gadf_pairs: list, labels: list, model, device,
    param_name: str, param_values: list,
    fixed_gamma: float, fixed_n: int, fixed_alpha: float,
) -> Dict[str, list]:
    """Sweep one parameter using pre‑computed Mel+GADF pairs.

    Only AW‑DPCNN fusion is re‑run for each parameter value;
    Mel spectrograms and GADF images are computed once and cached.
    """
    accuracies = []
    recalls = []

    for pv in tqdm(param_values, desc=f"  {param_name}", ncols=70):
        gamma  = pv if param_name == "gamma"    else fixed_gamma
        n_iter = int(pv) if param_name == "N"   else fixed_n
        # alpha stored as str in param_values; convert for alpha_LT sweep
        if param_name == "alpha_LT":
            alpha = float(pv) if isinstance(pv, str) else pv
        else:
            alpha = fixed_alpha

        fused = []
        for mel_t, gadf_t in mel_gadf_pairs:
            try:
                fused.append(_fuse_precomputed(mel_t, gadf_t,
                                                gamma=gamma, n_iter=n_iter,
                                                alpha=alpha))
            except Exception:
                continue

        if fused:
            acc, rec = _evaluate_fused(model, fused, labels, device)
        else:
            acc, rec = 0.0, 0.0
        accuracies.append(round(acc * 100, 2))
        recalls.append(round(rec * 100, 2))

    return {"param": param_values, "accuracy": accuracies, "recall": recalls}


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

def find_checkpoint(run_root: Path, trial: str = "trial_seed42",
                    model_dir: Optional[str] = None,
                    dataset_key: Optional[str] = None) -> Optional[Path]:
    """Recursively search for best.pt, preferring the given trial.

    When ``model_dir`` is provided, the search is scoped to
    ``run_root / model_dir`` first to avoid picking up checkpoints
    from other models in the same experiment group.

    When ``dataset_key`` is also provided (e.g. ``"12k_de"``), the
    search is further narrowed to ``model_dir / dataset_key``.
    """
    if not run_root.exists():
        return None

    # Build a list of search roots: most specific first, then broad
    search_roots = [run_root]
    if model_dir:
        if dataset_key:
            scoped = run_root / model_dir / dataset_key
        else:
            scoped = run_root / model_dir
        if scoped.exists():
            search_roots.insert(0, scoped)

    for root in search_roots:
        # Try precise path: root / ** / trial / checkpoints / best.pt
        precise = list(root.glob(f"**/{trial}/checkpoints/best.pt"))
        if precise:
            return sorted(precise, key=lambda p: p.stat().st_mtime,
                          reverse=True)[0]

    for root in search_roots:
        # Try any trial: root / ** / trial_seed* / checkpoints / best.pt
        any_trial = list(root.glob("**/trial_seed*/checkpoints/best.pt"))
        if any_trial:
            return sorted(any_trial, key=lambda p: p.stat().st_mtime,
                          reverse=True)[0]

    for root in search_roots:
        # Last resort: any best.pt
        any_best = list(root.glob("**/best.pt"))
        if any_best:
            return sorted(any_best, key=lambda p: p.stat().st_mtime,
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
    parser.add_argument("--dataset", default="", choices=[""] + DATASET_KEYS,
                        help=f"Dataset key to auto‑configure paths & params "
                             f"{{{','.join(DATASET_KEYS)}}}")
    parser.add_argument("--dataset-dir", default="datasets/cwru_12k_de",
                        help="ImageFolder dataset root (for test split list). "
                             "Overridden when --dataset is set.")
    parser.add_argument("--win-len", type=int, default=CWRU_WIN_LEN)
    parser.add_argument("--hop-len", type=int, default=CWRU_SEG_HOP)
    parser.add_argument("--max-samples", type=int, default=500,
                        help="Max test windows per sweep")
    parser.add_argument("--output-dir",
                        default="experiments/experiment_result/hyperparameter_sensitivity")
    parser.add_argument("--device", default="")
    parser.add_argument("--sweep", default="all",
                        help="Parameter to sweep: gamma, N, alpha_LT, all")
    # Mel parameters — must match training dataset
    parser.add_argument("--mel-n-fft", type=int, default=CWRU_N_FFT,
                        help=f"Mel STFT window size (default: {CWRU_N_FFT})")
    parser.add_argument("--mel-hop-len", type=int, default=CWRU_HOP_LEN,
                        help=f"Mel STFT hop length (default: {CWRU_HOP_LEN})")
    parser.add_argument("--mel-n-mels", type=int, default=CWRU_N_MELS,
                        help=f"Mel filter bank size (default: {CWRU_N_MELS})")
    parser.add_argument("--mel-fmax", type=int, default=CWRU_FMAX,
                        help=f"Mel max frequency (default: {CWRU_FMAX})")
    parser.add_argument("--tf-method", default="stft", choices=["mel", "stft"],
                        help="Time‑frequency representation: mel or stft (default: stft)")
    # Trial seed for checkpoint discovery
    parser.add_argument("--trial", default="trial_seed42",
                        help="Trial name for checkpoint selection (default: trial_seed42)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Resolve dataset via registry (if --dataset is given) ──
    dataset_key = ""                 # stays empty when --dataset is not used
    cwru_class_map = CWRU_CLASS_MAP  # mutable: may be rebuilt below
    cwru_sr      = CWRU_SR
    cwru_n_fft   = CWRU_N_FFT
    cwru_hop_len = CWRU_HOP_LEN
    cwru_n_mels  = CWRU_N_MELS
    cwru_fmax    = CWRU_FMAX
    cwru_win_len = CWRU_WIN_LEN
    cwru_seg_hop = CWRU_SEG_HOP

    if args.dataset:
        dataset_key = args.dataset
        ds_cfg = get_dataset_config(dataset_key)
        # Override dataset directory & all signal / Mel params from registry
        args.dataset_dir = ds_cfg["root_dir"]
        cwru_sr      = int(ds_cfg.get("sr", CWRU_SR))
        cwru_n_fft   = int(ds_cfg.get("n_fft", CWRU_N_FFT))
        cwru_hop_len = int(ds_cfg.get("hop_len", CWRU_HOP_LEN))
        cwru_n_mels  = int(ds_cfg.get("n_mels", CWRU_N_MELS))
        cwru_fmax    = int(ds_cfg.get("fmax", CWRU_FMAX))
        cwru_win_len = int(ds_cfg.get("win_len", CWRU_WIN_LEN))
        cwru_seg_hop = int(ds_cfg.get("seg_hop", CWRU_SEG_HOP))
        # Rebuild class map with correct src_dir & sensor_key.
        # Strip CWRU_ROOT prefix because _collect_cwru_test_windows
        # prepends it again (CWRU_ROOT / sub_path).
        src_dir_rel = ds_cfg["src_dir"]
        root_str = str(CWRU_ROOT) + "/"
        if src_dir_rel.startswith(root_str):
            src_dir_rel = src_dir_rel[len(root_str):]
        cwru_class_map = _build_cwru_class_map(
            src_dir_rel, ds_cfg["sensor_key"])
        # CLI Mel-param overrides still take precedence when explicitly set
        if args.mel_n_fft != CWRU_N_FFT:
            cwru_n_fft = args.mel_n_fft
        if args.mel_hop_len != CWRU_HOP_LEN:
            cwru_hop_len = args.mel_hop_len
        if args.mel_n_mels != CWRU_N_MELS:
            cwru_n_mels = args.mel_n_mels
        if args.mel_fmax != CWRU_FMAX:
            cwru_fmax = args.mel_fmax
        if args.win_len != CWRU_WIN_LEN:
            cwru_win_len = args.win_len
        if args.hop_len != CWRU_SEG_HOP:
            cwru_seg_hop = args.hop_len
        print(f"[dataset] {dataset_key} → {args.dataset_dir}  "
              f"(sr={cwru_sr}, sensor={ds_cfg['sensor_key']})")

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
        model_dir = Path(args.exp_config).stem  # e.g. "MSCA_VGG16" from MSCA_VGG16.yaml
        ckpt = find_checkpoint(run_root, trial=args.trial,
                               model_dir=model_dir,
                               dataset_key=dataset_key or None)
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
        args.dataset_dir, cwru_win_len, cwru_seg_hop,
        max_windows=args.max_samples * 2,
        class_map=cwru_class_map,
    )
    labels = [lbl for _, _, lbl in windows]
    print(f"Loaded {len(windows)} test windows "
          f"across {len(class_names)} classes")

    if len(windows) == 0:
        print("[ERROR] No test windows found — check --dataset-dir")
        sys.exit(1)

    # ── Sub‑sample windows ──
    rng = np.random.RandomState(42)
    n_sample = min(len(windows), args.max_samples)
    indices = rng.choice(len(windows), n_sample, replace=False)
    sampled_windows = [windows[i] for i in indices]
    sampled_labels  = [labels[i] for i in indices]
    print(f"Using {n_sample} windows for sensitivity sweep")

    # ── Pre‑compute TF + GADF (once — independent of γ, N, α) ──
    print(f"\nPre‑computing TF ({args.tf_method}) and GADF images for all sampled windows ...")
    print(f"  TF params: n_fft={cwru_n_fft}, hop_len={cwru_hop_len}, "
          f"n_mels={cwru_n_mels}, fmax={cwru_fmax}")
    mel_gadf_pairs = _precompute_mel_gadf(
        sampled_windows, cwru_sr,
        n_fft=cwru_n_fft, hop_len=cwru_hop_len,
        n_mels=cwru_n_mels, fmax=cwru_fmax,
        tf_method=args.tf_method,
    )
    print(f"Cached {len(mel_gadf_pairs)} Mel+GADF pairs")

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
            mel_gadf_pairs, sampled_labels, model, device,
            param_name, param_values,
            fixed_gamma=DEFAULT_GAMMA if fg is None else fg,
            fixed_n=DEFAULT_N if fn is None else fn,
            fixed_alpha=DEFAULT_ALPHA if fa is None else fa,
        )
        all_results[param_name] = res

        # Table
        print(f"  {'Value':>12s}  {'Acc %':>8s}  {'Rec %':>8s}")
        print(f"  {'-'*30}")
        for pv, acc, rec in zip(res["param"], res["accuracy"], res["recall"]):
            is_default = (
                (param_name == "gamma" and pv == DEFAULT_GAMMA) or
                (param_name == "N" and pv == DEFAULT_N) or
                (param_name == "alpha_LT" and abs(pv - DEFAULT_ALPHA) < 1e-8)
            )
            print(f"  {pv:>12}  {acc:8.2f}  {rec:8.2f}{' ← default' if is_default else ''}")

        # Plot
        plot_path = os.path.join(args.output_dir,
                                 f"sensitivity_{param_name}.png")
        _plot_sensitivity(param_name, res["param"], res["accuracy"], plot_path)
        print(f"  Plot: {plot_path}")

    # ── Save ──
    csv_path = os.path.join(args.output_dir, "sensitivity_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["parameter", "value", "accuracy", "recall"])
        for param_name, res in all_results.items():
            for pv, acc, rec in zip(res["param"], res["accuracy"], res["recall"]):
                writer.writerow([param_name, pv, acc, rec])
    print(f"\nCSV: {csv_path}")

    json_path = os.path.join(args.output_dir, "sensitivity_summary.json")
    with open(json_path, "w") as f:
        json.dump({
            k: {"param": [str(x) if isinstance(x, float) else x
                          for x in v["param"]],
                "accuracy": v["accuracy"],
                "recall": v["recall"]}
            for k, v in all_results.items()
        }, f, indent=2)
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
