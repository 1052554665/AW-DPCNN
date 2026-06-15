#!/usr/bin/env python3
"""
Build AW-DPCNN Fused Dataset — Transformer-Five (5‑class)
==========================================================

Reads **unsplit** .wav files from::

    raw-data/transformer-five/{ClassName}/*.wav

performs a **file‑level** stratified train/val/test split (avoids data
leakage between windows from the same recording), and produces AW‑DPCNN
fused images in::

    datasets/transformer-five/
        train/
            DCBias/  Harmonic/  Loosen/  Normal/  PartialDischarge/
        val/
            ...
        test/
            ...
        metadata.csv

The pipeline for each sliding window is:
  1. Mel spectrogram (pseudo‑colour via colormap)
  2. GADF image (pseudo‑colour via colormap)
  3. AW‑DPCNN fusion  (γ = 4, N = 20 — consistent with the paper)

**Critical**: The file‑level split ensures all windows from a single .wav
file are assigned to the same split — no data leakage.

Usage::

    # Default parameters (recommended)
    python scripts/build_transformer_five.py

    # Custom parameters
    python scripts/build_transformer_five.py \\
        --win-len 4096 --hop-len 1024 \\
        --n-fft 2048 --n-mels 128 --fmax 8000 \\
        --file-split 60,20,20 \\
        --workers 32 --metadata --verify

    # Dry-run (print plan only, no image generation)
    python scripts/build_transformer_five.py --dry-run
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

# ── Reuse the core pipeline from build_CWRU_dataset ──
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_CWRU_dataset import (                          # noqa: E402
    _compute_js_divergence,
    _file_level_split,
    _save_metadata,
    aw_dpcnn_fusion_color,
    generate_gadf_image,
    generate_mel_image,
    process_one_window,
)


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════

def _collect_raw_files(input_dir: str) -> dict:
    """Scan *input_dir* for class sub‑folders containing .wav files.

    Returns
    -------
    dict  {class_name: [(full_path, sr, signal_1d), ...]}
    """
    files_by_class: dict = {}
    root = Path(input_dir)

    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir() or class_dir.name.startswith("."):
            continue
        cls_name = class_dir.name
        items = []
        for wav_path in sorted(class_dir.glob("*.wav")):
            try:
                sr, data = wavfile.read(str(wav_path))
            except Exception as exc:
                print(f"[WARN] Cannot read {wav_path}: {exc}")
                continue
            if data.ndim > 1:
                data = data.mean(axis=1)
            data = data.astype(np.float32)
            items.append((str(wav_path), sr, data))
        if items:
            files_by_class[cls_name] = items
    return files_by_class


def _collect_window_tasks(
    files_by_class: dict,
    split_map: dict,           # {split_name: {cls: [file_path, ...]}}
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
    overwrite: bool,
):
    """Build task list + metadata rows from file‑level split maps.

    Each task is a tuple for ``process_one_window``.
    """
    tasks = []
    metadata_rows = []

    # Index: file_path -> (sr, signal)
    path_to_signal = {}
    for items in files_by_class.values():
        for fpath, sr_val, sig in items:
            path_to_signal[fpath] = (sr_val, sig)

    for split_name in ["train", "val", "test"]:
        split_cls = split_map.get(split_name, {})
        for cls, file_paths in split_cls.items():
            out_cls_dir = os.path.join(output_dir, split_name, cls)
            for fpath in file_paths:
                sr_val, signal = path_to_signal[fpath]
                stem = Path(fpath).stem

                # Generate windows
                if win_len <= 0 or win_len >= len(signal):
                    out_path = os.path.join(out_cls_dir, f"{stem}.png")
                    if not overwrite and os.path.exists(out_path):
                        continue
                    tasks.append((
                        signal, sr_val, out_path, img_size, n_iter,
                        n_fft, hop_length, n_mels, fmax, cmap,
                        gaf_method, False, gamma,
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
                    n_wins = len(range(0, len(signal) - win_len + 1, hop_len))
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
                            gaf_method, False, gamma,
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


def _parse_cmap(name: str) -> int:
    mapping = {
        "viridis": cv2.COLORMAP_VIRIDIS,
        "turbo": cv2.COLORMAP_TURBO,
        "jet": cv2.COLORMAP_JET,
        "plasma": cv2.COLORMAP_PLASMA,
        "inferno": cv2.COLORMAP_INFERNO,
        "magma": cv2.COLORMAP_MAGMA,
        "hot": cv2.COLORMAP_HOT,
        "cool": cv2.COLORMAP_COOL,
    }
    name_l = name.lower()
    if name_l in mapping:
        return mapping[name_l]
    raise ValueError(f"Unknown cmap '{name}'. Choices: {list(mapping.keys())}")


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build AW-DPCNN fused dataset for transformer-five (5-class)",
    )

    # I/O
    p.add_argument("--input-dir", default="raw-data/transformer-five",
                   help="Root dir of unsplit .wav class sub‑folders")
    p.add_argument("--output-dir", default="datasets/transformer-five",
                   help="Root dir for fused PNG images")
    p.add_argument("--sr", type=int, default=44100,
                   help="Sample rate (default: 44100 for transformer audio)")

    # Sliding window
    p.add_argument("--win-len", type=int, default=8192,
                   help="Window length in samples (0 = one image per file)")
    p.add_argument("--hop-len", type=int, default=4096,
                   help="Hop length between windows (samples)")

    # Image
    p.add_argument("--img-size", type=int, default=224,
                   help="Output image size (square)")
    p.add_argument("--cmap", default="viridis",
                   help="Colormap for Mel & GAF pseudo‑colour")

    # Mel parameters
    p.add_argument("--n-fft", type=int, default=2048)
    p.add_argument("--hop-length", type=int, default=256)
    p.add_argument("--n-mels", type=int, default=128)
    p.add_argument("--fmax", type=int, default=8000)

    # GADF
    p.add_argument("--gaf-method", default="difference",
                   choices=["difference", "summation"])

    # AW-DPCNN
    p.add_argument("--n-iter", type=int, default=20,
                   help="PCNN iterations (paper: N=20)")
    p.add_argument("--gamma", type=float, default=4.0,
                   help="Contrast amplification (paper: γ=4)")

    # File‑level split  (CRITICAL for leakage prevention)
    p.add_argument("--file-split", type=str, default="60,20,20",
                   help="File‑level train/val/test ratios, e.g. '60,20,20'")
    p.add_argument("--no-file-split", action="store_true",
                   help="Disable file‑level split (NOT recommended)")
    p.add_argument("--split-seed", type=int, default=42,
                   help="Random seed for split")

    # Metadata & verification
    p.add_argument("--metadata", action="store_true", default=True,
                   help="Generate metadata.csv (default: True)")
    p.add_argument("--no-metadata", action="store_false", dest="metadata",
                   help="Skip metadata.csv")
    p.add_argument("--verify", action="store_true", default=True,
                   help="Print class distribution & JS divergence (default: True)")
    p.add_argument("--no-verify", action="store_false", dest="verify",
                   help="Skip verification")

    # Execution
    p.add_argument("--workers", type=int, default=os.cpu_count() or 4,
                   help="Parallel workers")
    p.add_argument("--overwrite", action="store_true",
                   help="Re‑compute and overwrite existing outputs")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the split plan without generating images")

    return p


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    args = build_parser().parse_args()

    cmap_code = _parse_cmap(args.cmap)

    # ── Parse file‑split ──
    if args.no_file_split:
        file_split_ratios = ()
        print("[WARN] File‑level split DISABLED — possible data leakage!")
    else:
        parts = [float(x.strip()) for x in args.file_split.split(",")]
        if len(parts) != 3:
            print("[ERROR] --file-split requires three values, e.g. '60,20,20'")
            sys.exit(1)
        total = sum(parts)
        file_split_ratios = tuple(p / total for p in parts)

    # ── Header ──
    print("═" * 60)
    print("AW-DPCNN Transformer-Five Dataset Builder")
    print("═" * 60)
    print(f"  Input dir      : {args.input_dir}")
    print(f"  Output dir     : {args.output_dir}")
    print(f"  Sample rate    : {args.sr} Hz")
    print(f"  Window / Hop   : {args.win_len} / {args.hop_len}")
    print(f"  Image size     : {args.img_size}")
    print(f"  Colormap       : {args.cmap}")
    print(f"  PCNN iter / γ  : {args.n_iter} / {args.gamma}")
    print(f"  File split     : {args.file_split if file_split_ratios else 'DISABLED'}")
    print(f"  Split seed     : {args.split_seed}")
    print(f"  Metadata       : {args.metadata}")
    print(f"  Verify         : {args.verify}")
    print(f"  Workers        : {args.workers}")
    print(f"  Dry run        : {args.dry_run}")
    print("═" * 60)

    # ── Collect raw files ──
    files_by_class = _collect_raw_files(args.input_dir)
    if not files_by_class:
        print("[ERROR] No .wav files found.")
        sys.exit(1)

    total_files = sum(len(v) for v in files_by_class.values())
    print(f"\nFound {total_files} .wav files across {len(files_by_class)} classes:")
    for cls in sorted(files_by_class):
        print(f"  {cls:<20s} {len(files_by_class[cls]):4d} files")

    # ── File‑level split ──
    if file_split_ratios:
        cls_file_paths = {
            cls: [item[0] for item in items]
            for cls, items in files_by_class.items()
        }
        train_map, val_map, test_map = _file_level_split(
            cls_file_paths, file_split_ratios, args.split_seed,
        )
        split_map = {"train": train_map, "val": val_map, "test": test_map}

        # ── Print split plan ──
        print(f"\nFile‑level split (seed={args.split_seed}):")
        print(f"  {'Class':<20s} {'Total':>6s}  {'Train':>6s}  {'Val':>6s}  {'Test':>6s}")
        print(f"  {'-'*50}")
        for cls in sorted(files_by_class):
            n = len(files_by_class[cls])
            nt = len(train_map.get(cls, []))
            nv = len(val_map.get(cls, []))
            nte = len(test_map.get(cls, []))
            print(f"  {cls:<20s} {n:6d}  {nt:6d}  {nv:6d}  {nte:6d}")
        t_total = sum(len(v) for v in train_map.values())
        v_total = sum(len(v) for v in val_map.values())
        te_total = sum(len(v) for v in test_map.values())
        print(f"  {'TOTAL':<20s} {total_files:6d}  {t_total:6d}  {v_total:6d}  {te_total:6d}")

        # Quick leakage check on filenames
        train_files = {Path(p).name for paths in train_map.values() for p in paths}
        val_files = {Path(p).name for paths in val_map.values() for p in paths}
        test_files = {Path(p).name for paths in test_map.values() for p in paths}
        tv = train_files & val_files
        tt = train_files & test_files
        vt = val_files & test_files
        if tv or tt or vt:
            print(f"\n  [FAIL] File‑level LEAKAGE detected! train∩val={len(tv)} train∩test={len(tt)} val∩test={len(vt)}")
        else:
            print(f"\n  [OK] No file‑level leakage (all {total_files} files assigned to exactly one split)")
    else:
        split_map = {}
        print("\n[WARN] Skipping file‑level split. All windows from all files go into all splits.")

    if args.dry_run:
        print("\n[Dry run complete — no images generated.]")
        return

    # ── Collect window tasks ──
    tasks, metadata_rows = _collect_window_tasks(
        files_by_class=files_by_class,
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
        overwrite=args.overwrite,
    )

    print(f"\n[INFO] Total fusion tasks: {len(tasks)}")
    if not tasks:
        print("[INFO] Nothing to do — all images already exist (use --overwrite to re-generate).")
        return

    # ── Generate fused images (parallel) ──
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

    # ── Distribution verification ──
    if args.verify and metadata_rows:
        print("\n" + "─" * 60)
        print("Distribution verification")
        print("─" * 60)

        # Aggregate per-split class counts
        split_counts: dict = {}
        for row in metadata_rows:
            s = row["split"]
            c = row["class_label"]
            split_counts.setdefault(s, {}).setdefault(c, 0)
            split_counts[s][c] += 1

        for split_name in ["train", "val", "test"]:
            counts = split_counts.get(split_name, {})
            total = sum(counts.values())
            print(f"  {split_name}: {total} samples, "
                  f"classes: {dict(sorted(counts.items()))}")

        # JS divergence between splits
        splits_present = [s for s in ["train", "val", "test"] if s in split_counts]
        all_js_ok = True
        for i in range(len(splits_present)):
            for j in range(i + 1, len(splits_present)):
                sa, sb = splits_present[i], splits_present[j]
                js = _compute_js_divergence(split_counts[sa], split_counts[sb])
                status = "OK" if js < 0.01 else "WARN"
                if js >= 0.01:
                    all_js_ok = False
                print(f"  JS({sa}, {sb}) = {js:.6f}  [{status}]")

        if all_js_ok:
            print("\n  [OK] All JS divergences < 0.01 — no significant distribution shift.")
        else:
            print("\n  [WARN] JS divergence ≥ 0.01 detected — review the split.")

        print("─" * 60)


if __name__ == "__main__":
    main()
