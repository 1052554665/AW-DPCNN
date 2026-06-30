#!/usr/bin/env python3
"""
Ablation Dataset Builder — CWRU 12k DE
=======================================
Builds five dataset variants for the unified ablation study (B0–B8).

All datasets share the **same file‑level train/val/test split** for fair
comparison.  Source: 12 kHz drive‑end CWRU bearing data (.mat files).

Datasets built
--------------
  B0 — stft_only        STFT spectrogram only (pseudo‑colour, no fusion)
  B1 — gadf_only        GADF image only (pseudo‑colour, no fusion)
  B2 — concat           STFT + GADF pixel‑wise average (naive fusion)
  B3 — awdpcnn_gamma1   AW‑DPCNN with γ=1 (equal‑weight PCNN fusion)
  B4 — awdpcnn_full     Full AW‑DPCNN with γ=10 (adaptive fusion, shared
                         by B5–B8 classifier‑level experiments)

Output structure::

    datasets/ablation/
        stft_only/        train/{BF007,...,Normal}/  val/  test/  metadata.csv
        gadf_only/        ...
        concat/           ...
        awdpcnn_gamma1/   ...
        awdpcnn_full/     ...

Usage::

    python scripts/build_ablation_datasets.py --workers 32
"""

import argparse
import csv
import os
import random
import sys
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
from build_cwru_dataset import (  # noqa: E402
    _gray_to_pseudo,
    aw_dpcnn_fusion_color,
    generate_gadf_image,
    generate_mel_image,
    generate_stft_image,
    TF_GENERATORS,
)

# ═══════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════

SRC_DIR    = "raw-data/CWRU-dataset/12k_Drive_End_Bearing_Fault_Data"
NORMAL_DIR = "raw-data/CWRU-dataset/Normal"
OUT_ROOT   = "datasets/ablation"
SENSOR_KEY = "DE_time"
SR         = 12000
FMAX       = 6000
N_FFT      = 1024
HOP_LEN    = 256
N_MELS     = 128
WIN_LEN    = 2048
HOP_WIN    = 1024
IMG_SIZE   = 224
N_ITER     = 20

CLASS_MAP = OrderedDict([
    (("B",  "007"), "BF007"), (("B",  "014"), "BF014"), (("B",  "021"), "BF021"),
    (("IR", "007"), "IF007"), (("IR", "014"), "IF014"), (("IR", "021"), "IF021"),
    (("OR", "007"), "OF007"), (("OR", "014"), "OF014"), (("OR", "021"), "OF021"),
])
NORMAL_CLASS = "Normal"


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════

def _load_mat_signal(path: str, key: str) -> np.ndarray:
    mat = loadmat(path)
    for k in mat.keys():
        if key in k:
            s = mat[k].squeeze().astype(np.float32)
            return s.ravel() if s.ndim != 1 else s
    raise ValueError(f"No '{key}' in {path}")


def _collect_mat_files() -> Dict[str, List[str]]:
    files: Dict[str, List[str]] = OrderedDict()
    src = Path(SRC_DIR)
    for (ftype, sev), cls in CLASS_MAP.items():
        if ftype == "OR":
            sub = src / ftype / sev / "@6"
            if not sub.exists():
                sub = src / ftype / sev
        else:
            sub = src / ftype / sev
        mats = sorted(sub.glob("*.mat")) if sub.exists() else []
        if mats:
            files[cls] = [str(p) for p in mats]
    normal = Path(NORMAL_DIR)
    if normal.exists():
        mats = sorted(normal.glob("*.mat"))
        if mats:
            files[NORMAL_CLASS] = [str(p) for p in mats]
    return files


def _file_level_split(
    files_by_class: Dict[str, List[str]],
    ratios: tuple = (0.6, 0.2, 0.2),
    seed: int = 42,
) -> tuple:
    rng = random.Random(seed)
    train_m, val_m, test_m = {}, {}, {}
    for cls, paths in files_by_class.items():
        paths = sorted(paths)
        rng.shuffle(paths)
        n = len(paths)
        n_tr = max(1, round(n * ratios[0]))
        n_vl = max(1, round(n * ratios[1]))
        if n_tr + n_vl >= n:
            n_tr = max(1, n - 2)
            n_vl = max(1, n - n_tr - 1)
        train_m[cls] = paths[:n_tr]
        val_m[cls]   = paths[n_tr:n_tr + n_vl]
        test_m[cls]  = paths[n_tr + n_vl:]
    return train_m, val_m, test_m


# ═══════════════════════════════════════════════════════════════════════
#  Per‑variant image builders
# ═══════════════════════════════════════════════════════════════════════

# Set via CLI; controls which time‑frequency representation is used
# by the fusion and concat variants.
TF_METHOD = "stft"  # default — overridden by --tf-method


def _make_tf(window: np.ndarray) -> np.ndarray:
    """Generate time‑frequency image using the selected method."""
    if TF_METHOD == "stft":
        return generate_stft_image(window, SR, n_fft=N_FFT,
                                    hop_length=HOP_LEN, img_size=IMG_SIZE,
                                    cmap=cv2.COLORMAP_VIRIDIS)
    return generate_mel_image(window, SR, n_fft=N_FFT, hop_length=HOP_LEN,
                               n_mels=N_MELS, fmax=FMAX, img_size=IMG_SIZE,
                               cmap=cv2.COLORMAP_VIRIDIS)


def _make_gadf(window: np.ndarray) -> np.ndarray:
    return generate_gadf_image(window, img_size=IMG_SIZE,
                                cmap=cv2.COLORMAP_VIRIDIS)

def _make_concat(window: np.ndarray) -> np.ndarray:
    tf_img = _make_tf(window).astype(np.float32)
    gadf = _make_gadf(window).astype(np.float32)
    return ((tf_img + gadf) / 2).astype(np.uint8)

def _make_awdpcnn(window: np.ndarray, gamma: float) -> np.ndarray:
    tf_img = _make_tf(window)
    gadf = _make_gadf(window)
    return aw_dpcnn_fusion_color(tf_img, gadf, n_iter=N_ITER, gamma=gamma)

VARIANTS = {
    "stft_only":      (f"B0  TF-only ({TF_METHOD})",      _make_tf,        {}),
    "gadf_only":      ("B1  GADF-only",                    _make_gadf,       {}),
    "concat":         (f"B2  Concat (avg, {TF_METHOD})",   _make_concat,     {}),
    "awdpcnn_gamma1": ("B3  AW-DPCNN γ=1",                  _make_awdpcnn,   {"gamma": 1.0}),
    "awdpcnn_full":   (f"B4  AW-DPCNN γ=10 ({TF_METHOD})", _make_awdpcnn,   {"gamma": 10.0}),
}


# ═══════════════════════════════════════════════════════════════════════
#  Worker (module‑level for multiprocessing)
# ═══════════════════════════════════════════════════════════════════════

def _process_one(args_tuple):
    """Generate one image.  args: (builder_fn, window, out_path, kwargs)."""
    fn, win, out, kwargs = args_tuple
    try:
        img = fn(win, **kwargs)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        cv2.imwrite(out, img)
        return 1
    except Exception as exc:
        print(f"[ERROR] {out}: {exc}")
        return 0


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Ablation dataset builder — CWRU 12k DE")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--tf-method", default="stft", choices=["mel", "stft"],
                        help="Time‑frequency representation: mel or stft (default: stft)")
    args = parser.parse_args()

    # ── Apply TF method globally (used by variant builders) ──
    global TF_METHOD
    TF_METHOD = args.tf_method

    # ── Collect & split ──
    files_by_class = _collect_mat_files()
    if not files_by_class:
        print("[ERROR] No .mat files found.")
        sys.exit(1)
    total = sum(len(v) for v in files_by_class.values())
    print(f"Source: {total} .mat files, {len(files_by_class)} classes")
    print(f"TF method: {TF_METHOD}")
    for cls, paths in files_by_class.items():
        print(f"  {cls:8s}: {len(paths)} files")

    train_m, val_m, test_m = _file_level_split(files_by_class)
    split_map = {"train": train_m, "val": val_m, "test": test_m}
    for sn, sm in split_map.items():
        print(f"  {sn}: {sum(len(v) for v in sm.values())} files")

    # ── Load signals ──
    print("\nLoading .mat signals...")
    path_to_signal: Dict[str, np.ndarray] = {}
    for cls, paths in files_by_class.items():
        for p in tqdm(paths, desc=f"  {cls}", ncols=80):
            try:
                path_to_signal[p] = _load_mat_signal(p, SENSOR_KEY)
            except Exception as exc:
                print(f"\n[WARN] {p}: {exc}")

    # ── Pre‑segment all windows once ──
    print("\nSegmenting windows...")
    all_windows: Dict[str, tuple] = {}
    for cls, paths in files_by_class.items():
        for p in paths:
            sig = path_to_signal.get(p)
            if sig is None:
                continue
            stem = Path(p).stem
            wins = []
            for start in range(0, len(sig) - WIN_LEN + 1, HOP_WIN):
                wins.append(sig[start:start + WIN_LEN])
            all_windows[p] = (wins, stem)

    # ── Build each variant ──
    for variant_name, (label, builder_fn, builder_kwargs) in VARIANTS.items():
        print(f"\n{'='*60}\n  {label}\n{'='*60}")
        out_dir = Path(OUT_ROOT) / variant_name
        tasks = []
        metadata_rows = []

        for split_name in ["train", "val", "test"]:
            split_cls = split_map[split_name]
            for cls, file_paths in sorted(split_cls.items()):
                out_cls_dir = out_dir / split_name / cls
                for fpath in file_paths:
                    entry = all_windows.get(fpath)
                    if entry is None:
                        continue
                    windows, stem = entry
                    for idx, win in enumerate(windows):
                        fname = f"{stem}_{idx:05d}.png"
                        out_path = out_cls_dir / fname
                        if not args.overwrite and out_path.exists():
                            continue
                        tasks.append((builder_fn, win, str(out_path), builder_kwargs))
                        metadata_rows.append({
                            "filename": fname, "class_label": cls,
                            "source_file": stem, "split": split_name,
                            "window_idx": idx,
                            "window_start_sample": idx * HOP_WIN,
                        })

        print(f"  Tasks: {len(tasks)}")
        if not tasks:
            continue

        csv_path = out_dir / "metadata.csv"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["filename", "class_label",
                                              "source_file", "split",
                                              "window_idx", "window_start_sample"])
            w.writeheader()
            w.writerows(metadata_rows)

        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            results = list(tqdm(
                ex.map(_process_one, tasks, chunksize=8),
                total=len(tasks), desc=f"  {variant_name}", ncols=80,
            ))
        print(f"  Done: {sum(results)}/{len(tasks)}")

    print(f"\n✓ All datasets built in {OUT_ROOT}/")


if __name__ == "__main__":
    main()
