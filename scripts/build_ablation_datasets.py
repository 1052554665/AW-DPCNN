#!/usr/bin/env python3
"""
Build ablation datasets for component decomposition experiments (B0–B3).

B0 — Mel‑only  (pseudo‑colour Mel spectrogram, no fusion)
B1 — GADF‑only (pseudo‑colour GADF image, no fusion)
B2 — Concat    (pixel‑wise average of Mel + GADF pseudo‑colour images)
B3 — AW‑DPCNN γ=1  (fixed‑weight PCNN fusion, no adaptive weighting)

B4+ use the existing full AW‑DPCNN dataset (datasets/transformer-five).

All datasets share the same file‑level split for fair comparison.

Revise the source directory and parameters if needed::

    SRC_DIR = "raw-data/transformer-five"
    WIN_LEN, HOP_LEN = 8192, 4096

Usage::

    python scripts/build_ablation_datasets.py --workers 16
"""

import argparse
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

warnings.filterwarnings("ignore", message=".*TripleDES.*")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_CWRU_dataset import (  # noqa: E402
    _file_level_split,
    _gray_to_pseudo,
    aw_dpcnn_fusion_color,
)
from representation_comparison import (  # noqa: E402
    generate_gadf,
    generate_mel,
)

OUTPUT_ROOT = "datasets/ablation"
SRC_DIR = "raw-data/transformer-five"

# Common image parameters
IMG_SIZE = 224
CMAP = cv2.COLORMAP_VIRIDIS
WIN_LEN, HOP_LEN = 8192, 4096


def _collect_files():
    files_by_class = {}
    for class_dir in sorted(Path(SRC_DIR).iterdir()):
        if not class_dir.is_dir() or class_dir.name.startswith("."):
            continue
        wavs = sorted(class_dir.glob("*.wav"))
        if wavs:
            files_by_class[class_dir.name] = [str(p) for p in wavs]
    return files_by_class


def _load_signals(files_by_class):
    path_to_signal = {}
    for paths in files_by_class.values():
        for p in paths:
            sr, data = wavfile.read(p)
            if data.ndim > 1:
                data = data.mean(axis=1)
            path_to_signal[p] = (sr, data.astype(np.float32))
    return path_to_signal


def _build_single_rep(rep_fn, name, split_map, path_to_signal,
                      sequence_length=None):
    """Build a single‑representation dataset (B0, B1)."""
    output_dir = os.path.join(OUTPUT_ROOT, name)
    tasks = []

    for split_name in ["train", "val", "test"]:
        for cls, file_paths in sorted(split_map[split_name].items()):
            out_cls_dir = os.path.join(output_dir, split_name, cls)
            for fpath in file_paths:
                sr_val, signal = path_to_signal[fpath]
                stem = Path(fpath).stem
                for start in range(0, len(signal) - WIN_LEN + 1, HOP_LEN):
                    window = signal[start:start + WIN_LEN]
                    tasks.append((rep_fn, window, sr_val, out_cls_dir,
                                  f"{stem}_{len(tasks):05d}.png", sequence_length))
    return tasks, output_dir


def _build_concat(split_map, path_to_signal, sequence_length=None):
    """Build Mel+GADF pixel‑average dataset (B2)."""
    output_dir = os.path.join(OUTPUT_ROOT, "concat")
    tasks = []

    for split_name in ["train", "val", "test"]:
        for cls, file_paths in sorted(split_map[split_name].items()):
            out_cls_dir = os.path.join(output_dir, split_name, cls)
            for fpath in file_paths:
                sr_val, signal = path_to_signal[fpath]
                stem = Path(fpath).stem
                for start in range(0, len(signal) - WIN_LEN + 1, HOP_LEN):
                    window = signal[start:start + WIN_LEN]
                    tasks.append((window, sr_val, out_cls_dir,
                                  f"{stem}_{len(tasks):05d}.png", sequence_length))
    return tasks, output_dir


def _build_awdpcnn_gamma1(split_map, path_to_signal, sequence_length=None):
    """Build AW‑DPCNN with γ=1 (B3)."""
    output_dir = os.path.join(OUTPUT_ROOT, "awdpcnn_gamma1")
    tasks = []

    for split_name in ["train", "val", "test"]:
        for cls, file_paths in sorted(split_map[split_name].items()):
            out_cls_dir = os.path.join(output_dir, split_name, cls)
            for fpath in file_paths:
                sr_val, signal = path_to_signal[fpath]
                stem = Path(fpath).stem
                for start in range(0, len(signal) - WIN_LEN + 1, HOP_LEN):
                    window = signal[start:start + WIN_LEN]
                    tasks.append((window, sr_val, out_cls_dir,
                                  f"{stem}_{len(tasks):05d}.png", sequence_length))
    return tasks, output_dir


def _process_single_rep(args):
    rep_fn, window, sr_val, out_cls_dir, fname, sequence_length = args
    try:
        # generate_mel takes (signal, sr, ...), generate_gadf takes (signal, ...)
        if rep_fn is generate_mel:
            img = rep_fn(window, sr_val, img_size=IMG_SIZE, cmap=CMAP)
        else:
            img = rep_fn(window, img_size=IMG_SIZE,
                         sequence_length=sequence_length)
        out_path = os.path.join(out_cls_dir, fname)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, img)
        return 1
    except Exception as e:
        print(f"[ERROR] {fname}: {e}")
        return 0


def _process_concat(args):
    window, sr_val, out_cls_dir, fname, sequence_length = args
    try:
        mel = generate_mel(window, sr_val, img_size=IMG_SIZE, cmap=CMAP)
        gadf = generate_gadf(window, img_size=IMG_SIZE,
                             sequence_length=sequence_length)
        avg = ((mel.astype(np.float32) + gadf.astype(np.float32)) / 2).astype(np.uint8)
        out_path = os.path.join(out_cls_dir, fname)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, avg)
        return 1
    except Exception as e:
        print(f"[ERROR] {fname}: {e}")
        return 0


def _process_awdpcnn_gamma1(args):
    window, sr_val, out_cls_dir, fname, sequence_length = args
    try:
        mel = generate_mel(window, sr_val, img_size=IMG_SIZE, cmap=CMAP)
        gadf = generate_gadf(window, img_size=IMG_SIZE,
                             sequence_length=sequence_length)
        fused = aw_dpcnn_fusion_color(mel, gadf, n_iter=20, gamma=1.0)
        out_path = os.path.join(out_cls_dir, fname)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, fused)
        return 1
    except Exception as e:
        print(f"[ERROR] {fname}: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--sequence-length", type=int, default=None,
                        help="Max time‑steps for GAF (resample if longer). "
                             "None = use img_size * 4.")
    args = parser.parse_args()

    files_by_class = _collect_files()
    path_to_signal = _load_signals(files_by_class)
    ratios = (0.6, 0.2, 0.2)
    train_map, val_map, test_map = _file_level_split(files_by_class, ratios, 42)
    split_map = {"train": train_map, "val": val_map, "test": test_map}

    print(f"Source: {sum(len(v) for v in files_by_class.values())} files")
    print(f"Split: train={sum(len(v) for v in train_map.values())}  "
          f"val={sum(len(v) for v in val_map.values())}  "
          f"test={sum(len(v) for v in test_map.values())}")

    # ── B0: Mel‑only ──
    print("\n[B0] Mel-only...")
    tasks, _ = _build_single_rep(generate_mel, "mel_only", split_map, path_to_signal,
                                 sequence_length=args.sequence_length)
    print(f"  Tasks: {len(tasks)}")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        results = list(tqdm(ex.map(_process_single_rep, tasks, chunksize=8),
                            total=len(tasks), desc="  B0 mel_only"))
    print(f"  Done: {sum(results)}/{len(tasks)}")

    # ── B1: GADF‑only ──
    print("\n[B1] GADF-only...")
    tasks, _ = _build_single_rep(generate_gadf, "gadf_only", split_map, path_to_signal,
                                 sequence_length=args.sequence_length)
    print(f"  Tasks: {len(tasks)}")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        results = list(tqdm(ex.map(_process_single_rep, tasks, chunksize=8),
                            total=len(tasks), desc="  B1 gadf_only"))
    print(f"  Done: {sum(results)}/{len(tasks)}")

    # ── B2: Concat ──
    print("\n[B2] Concat (Mel+GADF average)...")
    tasks, _ = _build_concat(split_map, path_to_signal,
                             sequence_length=args.sequence_length)
    print(f"  Tasks: {len(tasks)}")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        results = list(tqdm(ex.map(_process_concat, tasks, chunksize=8),
                            total=len(tasks), desc="  B2 concat"))
    print(f"  Done: {sum(results)}/{len(tasks)}")

    # ── B3: AW‑DPCNN γ=1 ──
    print("\n[B3] AW-DPCNN γ=1...")
    tasks, _ = _build_awdpcnn_gamma1(split_map, path_to_signal,
                                     sequence_length=args.sequence_length)
    print(f"  Tasks: {len(tasks)}")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        results = list(tqdm(ex.map(_process_awdpcnn_gamma1, tasks, chunksize=8),
                            total=len(tasks), desc="  B3 awdpcnn_g1"))
    print(f"  Done: {sum(results)}/{len(tasks)}")

    print(f"\nDone. Output: {OUTPUT_ROOT}/")
    print("  mel_only/  gadf_only/  concat/  awdpcnn_gamma1/")


if __name__ == "__main__":
    main()
