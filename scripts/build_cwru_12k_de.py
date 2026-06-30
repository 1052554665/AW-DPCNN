#!/usr/bin/env python3
"""
AW-DPCNN Dataset Builder — CWRU 12kHz Drive End (10‑class)
============================================================

Builds AW-DPCNN fused images from the **12 kHz drive-end** bearing fault
data of the CWRU dataset.  The output is a 10‑class ImageFolder‑compatible
directory.

Target classes
--------------
  BF007   BF014   BF021      (ball faults    — 0.007", 0.014", 0.021")
  IF007   IF014   IF021      (inner race     — 0.007", 0.014", 0.021")
  OF007   OF014   OF021      (outer race @6  — 0.007", 0.014", 0.021")
  Normal

Data source
-----------
  raw-data/CWRU-dataset/12k_Drive_End_Bearing_Fault_Data/
    B/{007,014,021}/           (ball fault .mat files)
    IR/{007,014,021}/          (inner race .mat files)
    OR/{007,014,021}/@6/       (outer race @ 6 o'clock)
  raw-data/CWRU-dataset/Normal/ (normal baseline .mat files)

Usage::

    # Full build with file‑level train/val/test split
    python scripts/build_cwru_12k_de.py --output-dir ./datasets/cwru_12k_de --file-split 60,20,20 --split-seed 42 --metadata --verify --workers 32

    # Dry‑run (preview split plan without generating images)
    python scripts/build_cwru_12k_de.py --dry-run

Notes
-----
- OR faults default to the **@6** (6 o'clock) load position, which is the
  standard benchmark setting.
- Fault size 0.028" is excluded (not in the 10 target classes).
"""

import argparse
import csv
import os
import random
import sys
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

warnings.filterwarnings("ignore", message=".*TripleDES.*")

import cv2
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

# ── Reuse AW-DPCNN pipeline from the unified builder ──
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_cwru_dataset import (  # noqa: E402
    _compute_js_divergence,
    _parse_cmap,
    aw_dpcnn_fusion_color,
    generate_gadf_image,
    generate_mel_image,
    generate_stft_image,
    TF_GENERATORS,
    process_one_window,
)


# ═══════════════════════════════════════════════════════════════════════
#  Configuration — 12 kHz Drive End
# ═══════════════════════════════════════════════════════════════════════

DATA_ROOT = "raw-data/CWRU-dataset"
FAULT_DIR = f"{DATA_ROOT}/12k_Drive_End_Bearing_Fault_Data"
NORMAL_DIR = f"{DATA_ROOT}/Normal"
SAMPLE_RATE = 12_000
MAT_KEY_PATTERN = "DE_time"       # extract DE_time from .mat
OR_POSITION = "@6"                # load position for outer‑race faults

# Target classes (ordered for consistent reporting)
ALL_CLASSES = [
    "BF007", "BF014", "BF021",
    "IF007", "IF014", "IF021",
    "OF007", "OF014", "OF021",
    "Normal",
]

# Map (fault_type, size_str) → class name
TYPE_SIZE_TO_CLASS = {
    ("B",  "007"): "BF007", ("B",  "014"): "BF014", ("B",  "021"): "BF021",
    ("IR", "007"): "IF007", ("IR", "014"): "IF014", ("IR", "021"): "IF021",
    ("OR", "007"): "OF007", ("OR", "014"): "OF014", ("OR", "021"): "OF021",
}

# Sizes to include
SIZES = ("007", "014", "021")
FAULT_TYPES = ("B", "IR", "OR")


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════

def _load_signal(mat_path: str, key_pattern: str = MAT_KEY_PATTERN) -> np.ndarray:
    """Load the target signal array from a CWRU .mat file."""
    mat = loadmat(mat_path)
    for key in mat.keys():
        if key_pattern in key and not key.startswith("__"):
            signal = mat[key].squeeze().astype(np.float32)
            if signal.ndim != 1:
                signal = signal.ravel()
            return signal
    raise ValueError(f"No key containing '{key_pattern}' found in {mat_path}")


def _collect_fault_files() -> dict:
    """Walk FAULT_DIR and return {class_name: [(mat_path, sr, signal), ...]}."""
    files_by_class: dict = defaultdict(list)
    fault_root = Path(FAULT_DIR)

    for fault_type in FAULT_TYPES:
        type_dir = fault_root / fault_type
        if not type_dir.is_dir():
            continue

        for size_str in SIZES:
            size_dir = type_dir / size_str
            if not size_dir.is_dir():
                continue

            cls_name = TYPE_SIZE_TO_CLASS.get((fault_type, size_str))
            if cls_name is None:
                continue

            # Resolve the innermost directory containing .mat files
            if fault_type == "OR":
                # OR: prefer OR_POSITION sub‑dir; fall back to size_dir itself
                mat_dir = size_dir / OR_POSITION
                if not mat_dir.is_dir():
                    mat_dir = size_dir
            else:
                mat_dir = size_dir

            mat_files = sorted(mat_dir.glob("*.mat"))
            for mp in mat_files:
                try:
                    sig = _load_signal(str(mp))
                except Exception as exc:
                    print(f"[WARN] {mp}: {exc}")
                    continue
                files_by_class[cls_name].append((str(mp), SAMPLE_RATE, sig))

    return dict(files_by_class)


def _collect_normal_files() -> list:
    """Collect normal baseline signals from NORMAL_DIR."""
    normal_root = Path(NORMAL_DIR)
    normal_files = []
    for mp in sorted(normal_root.glob("*.mat")):
        try:
            sig = _load_signal(str(mp))
        except Exception as exc:
            print(f"[WARN] {mp}: {exc}")
            continue
        normal_files.append((str(mp), SAMPLE_RATE, sig))
    return normal_files


# ═══════════════════════════════════════════════════════════════════════
#  File‑level split
# ═══════════════════════════════════════════════════════════════════════

def _file_level_split(
    files_by_class: dict,
    ratios: tuple,
    seed: int = 42,
) -> tuple:
    rng = random.Random(seed)
    train_map, val_map, test_map = {}, {}, {}
    r_train, r_val, r_test = ratios

    for cls, items in sorted(files_by_class.items()):
        files = sorted(items, key=lambda x: x[0])
        rng.shuffle(files)
        n = len(files)
        n_train = max(1, round(n * r_train))
        n_val = max(1, round(n * r_val))
        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val = max(1, n - n_train - 1)
        train_map[cls] = files[:n_train]
        val_map[cls] = files[n_train:n_train + n_val]
        test_map[cls] = files[n_train + n_val:]

    return train_map, val_map, test_map


# ═══════════════════════════════════════════════════════════════════════
#  Task collector
# ═══════════════════════════════════════════════════════════════════════

def _build_tasks_and_metadata(
    split_map: dict,
    output_dir: str,
    win_len: int,
    hop_len: int,
    img_size: int,
    n_iter: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    fmax: int,
    cmap: int,
    gaf_method: str,
    gamma: float,
    sequence_length: int,
    overwrite: bool,
    tf_method: str = "stft",
) -> tuple:
    tasks, metadata_rows = [], []

    for split_name in ("train", "val", "test"):
        for cls, items in sorted(split_map.get(split_name, {}).items()):
            out_cls_dir = os.path.join(output_dir, split_name, cls)
            for fpath, sr_val, signal in items:
                stem = Path(fpath).stem
                if win_len <= 0 or win_len >= len(signal):
                    out_path = os.path.join(out_cls_dir, f"{stem}.png")
                    if not overwrite and os.path.exists(out_path):
                        continue
                    tasks.append((
                        signal, sr_val, out_path, img_size, n_iter,
                        n_fft, hop_length, n_mels, fmax, cmap,
                        gaf_method, False, gamma, sequence_length,
                        tf_method,
                    ))
                    metadata_rows.append({
                        "filename": f"{stem}.png",
                        "class_label": cls,
                        "source_file": os.path.basename(fpath),
                        "split": split_name,
                        "window_idx": 0,
                        "window_start_sample": 0,
                    })
                else:
                    idx = 0
                    for start in range(0, len(signal) - win_len + 1, hop_len):
                        window = signal[start:start + win_len]
                        fname = f"{stem}_{idx:05d}.png"
                        out_path = os.path.join(out_cls_dir, fname)
                        if not overwrite and os.path.exists(out_path):
                            idx += 1
                            continue
                        tasks.append((
                            window, sr_val, out_path, img_size, n_iter,
                            n_fft, hop_length, n_mels, fmax, cmap,
                            gaf_method, False, gamma, sequence_length,
                            tf_method,
                        ))
                        metadata_rows.append({
                            "filename": fname,
                            "class_label": cls,
                            "source_file": os.path.basename(fpath),
                            "split": split_name,
                            "window_idx": idx,
                            "window_start_sample": start,
                        })
                        idx += 1

    return tasks, metadata_rows


def _save_metadata(csv_path: str, rows: list):
    fieldnames = [
        "filename", "class_label", "source_file", "split",
        "window_idx", "window_start_sample",
    ]
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AW-DPCNN CWRU 12kHz Drive End Dataset Builder (10‑class)")

    p.add_argument("--output-dir", default="datasets/cwru_12k_de",
                   help="Root dir for fused PNG images (ImageFolder layout)")
    p.add_argument("--sr", type=int, default=SAMPLE_RATE,
                   help=f"Sample rate override (default: {SAMPLE_RATE})")
    p.add_argument("--win-len", type=int, default=2048,
                   help="Window length in samples (0 = one image per file)")
    p.add_argument("--hop-len", type=int, default=1024,
                   help="Hop length between windows")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--cmap", default="viridis")
    p.add_argument("--n-fft", type=int, default=1024)
    p.add_argument("--hop-length", type=int, default=128)
    p.add_argument("--n-mels", type=int, default=128)
    p.add_argument("--fmax", type=int, default=6000)
    p.add_argument("--gaf-method", default="difference",
                   choices=["difference", "summation"])
    p.add_argument("--sequence-length", type=int, default=None,
                   help="Max time-steps for GAF")
    p.add_argument("--n-iter", type=int, default=20)
    p.add_argument("--gamma", type=float, default=4.0)
    p.add_argument("--tf-method", default="stft", choices=["mel", "stft"],
                   help="Time‑frequency representation: mel or stft (default: stft)")
    p.add_argument("--file-split", type=str, default="60,20,20",
                   help="Train/val/test ratios")
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--metadata", action="store_true", default=True)
    p.add_argument("--no-metadata", action="store_false", dest="metadata")
    p.add_argument("--verify", action="store_true", default=True)
    p.add_argument("--no-verify", action="store_false", dest="verify")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    args = build_parser().parse_args()
    cmap_code = _parse_cmap(args.cmap)

    parts = [float(x.strip()) for x in args.file_split.split(",")]
    if len(parts) != 3:
        print("[ERROR] --file-split requires three values, e.g. '60,20,20'")
        sys.exit(1)
    total_r = sum(parts)
    ratios = tuple(p / total_r for p in parts)

    # ── Header ──
    print("═" * 60)
    print("AW-DPCNN CWRU 12kHz Drive End Dataset Builder (10‑class)")
    print("═" * 60)
    print(f"  Fault dir      : {FAULT_DIR}")
    print(f"  Normal dir     : {NORMAL_DIR}")
    print(f"  Sample rate    : {SAMPLE_RATE} Hz")
    print(f"  Mat key        : {MAT_KEY_PATTERN}")
    print(f"  OR position    : {OR_POSITION}")
    print(f"  Window / Hop   : {args.win_len} / {args.hop_len}")
    print(f"  TF method      : {args.tf_method}")
    print(f"  File split     : {args.file_split}  (seed={args.split_seed})")
    print(f"  Workers        : {args.workers}")
    print(f"  Dry run        : {args.dry_run}")
    print("═" * 60)

    # ── Collect ──
    files_by_class = _collect_fault_files()
    normal_files = _collect_normal_files()
    if normal_files:
        files_by_class["Normal"] = normal_files

    total_files = sum(len(v) for v in files_by_class.values())
    print(f"\nCollected {total_files} .mat files → {len(files_by_class)} classes:")

    for cls in ALL_CLASSES:
        items = files_by_class.get(cls, [])
        n = len(items)
        marker = "✓" if n > 0 else "✗ — missing"
        print(f"  {marker} {cls:<8s} {n:3d} file(s)")

    missing = [c for c in ALL_CLASSES if c not in files_by_class]
    if missing:
        print(f"\n⚠  Missing classes: {', '.join(missing)}")

    if not files_by_class:
        print("\n[ERROR] No files found.")
        sys.exit(1)

    # ── Split ──
    train_map, val_map, test_map = _file_level_split(files_by_class, ratios, args.split_seed)
    split_map = {"train": train_map, "val": val_map, "test": test_map}

    print(f"\nFile‑level split (seed={args.split_seed}):")
    print(f"  {'Class':<8s} {'Total':>6s}  {'Train':>6s}  {'Val':>6s}  {'Test':>6s}")
    print(f"  {'-'*40}")
    t_f, v_f, te_f = 0, 0, 0
    for cls in ALL_CLASSES:
        if cls not in files_by_class:
            continue
        n = len(files_by_class[cls])
        nt = len(train_map.get(cls, []))
        nv = len(val_map.get(cls, []))
        nte = len(test_map.get(cls, []))
        t_f += nt; v_f += nv; te_f += nte
        warn = "  ⚠" if (nt == 0 or nv == 0 or nte == 0) else ""
        print(f"  {cls:<8s} {n:6d}  {nt:6d}  {nv:6d}  {nte:6d}{warn}")
    print(f"  {'-'*40}")
    print(f"  {'TOTAL':<8s} {total_files:6d}  {t_f:6d}  {v_f:6d}  {te_f:6d}")

    if args.dry_run:
        print("\n[Dry run complete — no images generated.]")
        return

    # ── Build tasks ──
    tasks, metadata_rows = _build_tasks_and_metadata(
        split_map=split_map,
        output_dir=args.output_dir,
        win_len=args.win_len,
        hop_len=args.hop_len,
        img_size=args.img_size,
        n_iter=args.n_iter,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        fmax=args.fmax,
        cmap=cmap_code,
        gaf_method=args.gaf_method,
        gamma=args.gamma,
        sequence_length=args.sequence_length,
        overwrite=args.overwrite,
        tf_method=args.tf_method,
    )

    print(f"\n[INFO] Total fusion tasks: {len(tasks)}")
    if not tasks:
        print("[INFO] Nothing to do — all images already exist (use --overwrite).")
        return

    # ── Generate ──
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(tqdm(
            executor.map(process_one_window, tasks, chunksize=8),
            total=len(tasks),
            desc="Building fused dataset",
            ncols=100,
        ))

    ok = sum(results)
    print(f"\n✅ Done — {ok}/{len(tasks)} images written to {args.output_dir}")

    # ── Metadata ──
    if args.metadata and metadata_rows:
        csv_path = os.path.join(args.output_dir, "metadata.csv")
        _save_metadata(csv_path, metadata_rows)
        print(f"📋 Metadata saved to {csv_path} ({len(metadata_rows)} rows)")

    # ── Verify ──
    if args.verify and metadata_rows:
        print("\n" + "─" * 60)
        print("Distribution verification")
        print("─" * 60)
        split_counts: dict = defaultdict(lambda: defaultdict(int))
        for row in metadata_rows:
            split_counts[row["split"]][row["class_label"]] += 1

        for sn in ("train", "val", "test"):
            counts = dict(sorted(split_counts[sn].items()))
            total = sum(counts.values())
            print(f"  {sn}: {total:6d} samples  {counts}")

        splits_present = [s for s in ("train", "val", "test") if s in split_counts]
        all_ok = True
        for i in range(len(splits_present)):
            for j in range(i + 1, len(splits_present)):
                sa, sb = splits_present[i], splits_present[j]
                js = _compute_js_divergence(split_counts[sa], split_counts[sb])
                status = "OK" if js < 0.05 else "WARN"
                if js >= 0.05:
                    all_ok = False
                print(f"  JS({sa}, {sb}) = {js:.6f}  [{status}]")
        print("─" * 60)


if __name__ == "__main__":
    main()
