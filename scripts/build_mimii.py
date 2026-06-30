#!/usr/bin/env python3
"""
Build AW-DPCNN Fused Dataset — MIMII (per noise level, 2‑class default)
=========================================================================

Reads raw .wav files from::

    raw-data/MIMII/{noise_level}/{noise_level}_{machine}/{machine}/id_{XX}/{normal|abnormal}/*.wav

Produces AW‑DPCNN fused images in ImageFolder layout.  By default uses a
**binary normal/abnormal** scheme (2 classes, universal "healthy vs faulty").

Noise levels
------------
MIMII provides three background‑noise SNR levels (identical recordings)::

    -6_dB   0_dB   6_dB

Use ``--noise-level`` to build a single‑level dataset or ``all`` for combined.

Class schemes
-------------
``binary`` (default)
    2 classes: ``normal``, ``abnormal`` — pooled across all 4 machine types.
    Best for cross‑domain generalisation (healthy/faulty is universal).

``machine_condition``
    8 classes: ``fan_normal``, ``fan_abnormal``, ``pump_normal``, ...

Usage::

    # Binary scheme, 0 dB only
    python scripts/build_mimii.py --noise-level 0_dB --output-dir datasets/mimii_0dB --workers 32

    # Binary scheme, 6 dB only
    python scripts/build_mimii.py --noise-level 6_dB --output-dir datasets/mimii_6dB

    # Binary scheme, -6 dB only
    python scripts/build_mimii.py --noise-level -6_dB --output-dir datasets/mimii_-6dB

    # All noise levels combined
    python scripts/build_mimii.py --noise-level all --output-dir datasets/MIMII

    # 8-class scheme (machine × condition)
    python scripts/build_mimii.py --noise-level 0_dB --class-scheme machine_condition

    # Dry-run
    python scripts/build_mimii.py --noise-level 0_dB --dry-run

control the image count:

    # Default: ~38 windows/file → ~680k images/noise-level
    python scripts/build_mimii.py --noise-level 0_dB

    # 10 windows per file → ~180k images  
    python scripts/build_mimii.py --noise-level 0_dB --max-windows-per-file 10

    # 5 windows per file → ~90k images
    python scripts/build_mimii.py --noise-level 0_dB --max-windows-per-file 5

    # 1 image per file → ~18k images (use --win-len 0)
    python scripts/build_mimii.py --noise-level 0_dB --win-len 0
    
    python scripts/build_mimii.py --noise-level='-6_dB' --output-dir datasets/mimii_-6dB --win-len 0

    # Combine with longer hop for fewer windows + cap
    python scripts/build_mimii.py --noise-level 0_dB --hop-len 16384 --max-windows-per-file 5
"""

import argparse
import csv
import os
import sys
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore", message=".*TripleDES.*")

import cv2
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_cwru_dataset import process_one_window  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════

MACHINE_TYPES = ["fan", "pump", "slider", "valve"]
CONDITIONS = ["normal", "abnormal"]
NOISE_LEVELS = ["-6_dB", "0_dB", "6_dB"]

# ── 2‑class (binary) scheme ──
BINARY_CLASSES = ["normal", "abnormal"]

# ── 8‑class (machine × condition) scheme ──
MACHINE_CONDITION_CLASSES = [
    "fan_normal", "fan_abnormal",
    "pump_normal", "pump_abnormal",
    "slider_normal", "slider_abnormal",
    "valve_normal", "valve_abnormal",
]


# ═══════════════════════════════════════════════════════════════════════
#  File collection (no audio loading — fast scan)
# ═══════════════════════════════════════════════════════════════════════

def _collect_files(
    input_dir: str,
    noise_level: str,
    class_scheme: str,
) -> Tuple[Dict[str, list], list]:
    """Scan for .wav files, filter by noise level, group by target class.

    Returns
    -------
    (files_by_class, skipped)
        files_by_class  – {target_class: [(path, noise_level, id_str, machine, condition), ...]}
        skipped         – list of (filename, reason)
    """
    files_by_class: dict = defaultdict(list)
    skipped: list = []

    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"[ERROR] Input directory not found: {input_path}")
        sys.exit(1)

    # Determine which noise-level directories to scan
    if noise_level == "all":
        scan_dirs = [input_path / nl for nl in NOISE_LEVELS]
    else:
        nl_dir = input_path / noise_level
        if not nl_dir.exists():
            print(f"[ERROR] Noise level directory not found: {nl_dir}")
            sys.exit(1)
        scan_dirs = [nl_dir]

    for scan_dir in scan_dirs:
        nl = scan_dir.name  # e.g. "0_dB"
        for wav_path in sorted(scan_dir.glob("**/*.wav")):
            parts = wav_path.parts
            try:
                condition = parts[-2]   # normal or abnormal
                id_str = parts[-3]      # id_00, id_02, ...
                machine = parts[-4]     # fan, pump, slider, valve
            except IndexError:
                skipped.append((wav_path.name, "unexpected directory depth"))
                continue

            if machine not in MACHINE_TYPES:
                skipped.append((wav_path.name, f"unknown machine '{machine}'"))
                continue
            if condition not in CONDITIONS:
                skipped.append((wav_path.name, f"unknown condition '{condition}'"))
                continue

            # Determine target class based on scheme
            if class_scheme == "binary":
                target_class = condition  # "normal" or "abnormal"
            else:
                target_class = f"{machine}_{condition}"

            files_by_class[target_class].append(
                (str(wav_path), nl, id_str, machine, condition)
            )

    return dict(files_by_class), skipped


# ═══════════════════════════════════════════════════════════════════════
#  File‑level split (id‑aware, no data leakage)
# ═══════════════════════════════════════════════════════════════════════

def _file_level_split(
    files_by_class: Dict[str, list],
    ratios: Tuple[float, float, float],
    seed: int = 42,
) -> Tuple[dict, dict, dict]:
    """Stratified split — same ``id_XX`` stays in one split.

    All MIMII files are 10 s @ 16 kHz, so file count is a fair proxy
    for workload.
    """
    r_train, r_val, r_test = ratios
    total_r = r_train + r_val + r_test
    r_train, r_val, r_test = r_train / total_r, r_val / total_r, r_test / total_r

    train_map, val_map, test_map = {}, {}, {}

    for cls, items in sorted(files_by_class.items()):
        # Group by id_str to keep same-ID files together
        id_groups: dict = defaultdict(list)
        for it in items:
            id_str = it[2]
            id_groups[id_str].append(it)

        # Weight groups by file count (uniform 10s duration)
        group_items = [(len(grp), grp) for grp in id_groups.values()]
        group_items.sort(key=lambda x: x[0], reverse=True)

        n_groups = len(group_items)
        n_train = max(1, round(n_groups * r_train))
        n_val = max(1, round(n_groups * r_val))
        n_test = max(1, n_groups - n_train - n_val)
        while n_train + n_val + n_test > n_groups:
            if n_test > 1:
                n_test -= 1
            elif n_val > 1:
                n_val -= 1
            else:
                n_train -= 1

        bins = [
            {"wins": 0, "items": [], "target": n_train, "name": "train"},
            {"wins": 0, "items": [], "target": n_val, "name": "val"},
            {"wins": 0, "items": [], "target": n_test, "name": "test"},
        ]

        for w, grp in group_items:
            candidates = [(b["wins"], i, b) for i, b in enumerate(bins)
                          if len(b["items"]) < b["target"]]
            if candidates:
                candidates.sort(key=lambda x: x[0])
                _, _, chosen = candidates[0]
            else:
                bins.sort(key=lambda b: b["wins"])
                chosen = bins[0]
            chosen["wins"] += w
            chosen["items"].extend(grp)

        train_map[cls] = bins[0]["items"]
        val_map[cls] = bins[1]["items"]
        test_map[cls] = bins[2]["items"]

    return train_map, val_map, test_map


def _compute_js_divergence(counts_a: dict, counts_b: dict) -> float:
    """Jensen–Shannon divergence between two class‑count dictionaries."""
    all_classes = sorted(set(counts_a) | set(counts_b))
    total_a = sum(counts_a.values()) or 1
    total_b = sum(counts_b.values()) or 1
    p = np.array([counts_a.get(c, 0) / total_a for c in all_classes])
    q = np.array([counts_b.get(c, 0) / total_b for c in all_classes])
    m = 0.5 * (p + q)

    def _kl(x, y):
        mask = (x > 0) & (y > 0)
        return np.sum(x[mask] * np.log(x[mask] / y[mask]))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _parse_cmap(name: str) -> int:
    mapping = {
        "viridis": cv2.COLORMAP_VIRIDIS, "turbo": cv2.COLORMAP_TURBO,
        "jet": cv2.COLORMAP_JET, "plasma": cv2.COLORMAP_PLASMA,
        "inferno": cv2.COLORMAP_INFERNO, "magma": cv2.COLORMAP_MAGMA,
        "hot": cv2.COLORMAP_HOT, "cool": cv2.COLORMAP_COOL,
    }
    name_l = name.lower()
    if name_l in mapping:
        return mapping[name_l]
    raise ValueError(f"Unknown cmap '{name}'. Choices: {list(mapping.keys())}")


# ═══════════════════════════════════════════════════════════════════════
#  Task collector — builds lightweight task tuples (NO audio loading)
# ═══════════════════════════════════════════════════════════════════════

def _collect_window_tasks(
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
    tf_method: str,
    overwrite: bool,
    max_windows: int = 0,
) -> Tuple[list, list]:
    """Build task list + metadata rows.  **Audio is NOT loaded here.**

    Each task is a lightweight tuple of (file_path, params…) so that the
    heavy I/O happens inside the worker process.
    """
    tasks = []
    metadata_rows = []

    for split_name in ["train", "val", "test"]:
        split_cls = split_map.get(split_name, {})
        for cls, items in sorted(split_cls.items()):
            out_cls_dir = os.path.join(output_dir, split_name, cls)
            for item in items:
                # item: (path, noise_level, id_str, machine, condition)
                fpath = item[0]
                noise_level = item[1]
                id_str = item[2]
                machine = item[3]
                condition = item[4]

                stem = Path(fpath).stem
                prefix = f"{noise_level}_{id_str}_{stem}"

                tasks.append((
                    fpath, out_cls_dir, prefix, img_size, n_iter,
                    n_fft, hop_length, n_mels, fmax, cmap,
                    gaf_method, gamma, sequence_length, tf_method,
                    win_len, hop_len, int(overwrite), max_windows,
                ))
                metadata_rows.append({
                    "filename_prefix": prefix,
                    "class_label": cls,
                    "source_file": os.path.basename(fpath),
                    "noise_level": noise_level,
                    "machine_id": id_str,
                    "machine": machine,
                    "condition_raw": condition,
                    "split": split_name,
                })

    return tasks, metadata_rows


# ═══════════════════════════════════════════════════════════════════════
#  Per‑file worker (audio loading + windowing + fusion — in subprocess)
# ═══════════════════════════════════════════════════════════════════════

def _process_one_file(task: tuple) -> int:
    """Load one audio file, slice into windows, fuse each, save to disk.

    This runs inside a ``ProcessPoolExecutor`` worker so that 54k files
    are loaded in parallel, not sequentially in the main thread.

    Returns the number of successfully written images.
    """
    (fpath, out_cls_dir, prefix, img_size, n_iter,
     n_fft, hop_length, n_mels, fmax, cmap,
     gaf_method, gamma, sequence_length, tf_method,
     win_len, hop_len, overwrite, max_windows) = task

    # ── Load audio ──
    try:
        sr_val, signal = wavfile.read(fpath)
    except Exception:
        return 0
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    signal = signal.astype(np.float32)

    # ── Build per‑window tasks ──
    window_tasks = []
    os.makedirs(out_cls_dir, exist_ok=True)

    if win_len <= 0 or win_len >= len(signal):
        out_path = os.path.join(out_cls_dir, f"{prefix}.png")
        if overwrite or not os.path.exists(out_path):
            window_tasks.append((
                signal, sr_val, out_path, img_size, n_iter,
                n_fft, hop_length, n_mels, fmax, cmap,
                gaf_method, False, gamma, sequence_length,
                tf_method,
            ))
    else:
        idx = 0
        for start in range(0, len(signal) - win_len + 1, hop_len):
            if max_windows > 0 and idx >= max_windows:
                break
            window = signal[start:start + win_len]
            fname = f"{prefix}_{idx:05d}.png"
            out_path = os.path.join(out_cls_dir, fname)
            if overwrite or not os.path.exists(out_path):
                window_tasks.append((
                    window, sr_val, out_path, img_size, n_iter,
                    n_fft, hop_length, n_mels, fmax, cmap,
                    gaf_method, False, gamma, sequence_length,
                    tf_method,
                ))
            idx += 1

    # ── Run AW‑DPCNN fusion for each window ──
    ok = 0
    for wt in window_tasks:
        ok += process_one_window(wt)
    return ok


def _save_metadata(csv_path: str, rows: list):
    fieldnames = [
        "filename", "class_label", "source_file", "noise_level",
        "machine_id", "machine", "condition_raw",
        "split", "window_idx", "window_start_sample",
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
        description="Build AW-DPCNN fused dataset for MIMII",
    )
    p.add_argument("--input-dir", default="raw-data/MIMII")
    p.add_argument("--output-dir", default="datasets/mimii_0dB",
                   help="Root dir for fused PNG images (ImageFolder layout)")
    p.add_argument("--sr", type=int, default=16000)

    # ── Noise level ──
    p.add_argument("--noise-level", default="0_dB",
                   choices=["-6_dB", "0_dB", "6_dB", "all"],
                   help="Which MIMII noise level to include (default: 0_dB)")

    # ── Class scheme ──
    p.add_argument("--class-scheme", default="binary",
                   choices=["binary", "machine_condition"],
                   help="binary = normal/abnormal (2-class); "
                        "machine_condition = machine×condition (8-class)")

    # Sliding window
    p.add_argument("--win-len", type=int, default=8192)
    p.add_argument("--hop-len", type=int, default=4096)
    p.add_argument("--max-windows-per-file", type=int, default=0,
                   help="Cap windows per audio file (0 = unlimited, default: 0)")

    # Image
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--cmap", default="viridis")

    # Spectrogram
    p.add_argument("--n-fft", type=int, default=1024)
    p.add_argument("--hop-length", type=int, default=256)
    p.add_argument("--n-mels", type=int, default=128)
    p.add_argument("--fmax", type=int, default=8000)

    # GADF
    p.add_argument("--gaf-method", default="difference",
                   choices=["difference", "summation"])
    p.add_argument("--sequence-length", type=int, default=None)

    # AW-DPCNN
    p.add_argument("--n-iter", type=int, default=20)
    p.add_argument("--gamma", type=float, default=4.0)

    # TF method
    p.add_argument("--tf-method", default="stft",
                   choices=["mel", "stft"])

    # File‑level split
    p.add_argument("--file-split", type=str, default="60,20,20")
    p.add_argument("--split-seed", type=int, default=42)

    # Metadata & verification
    p.add_argument("--metadata", action="store_true", default=True)
    p.add_argument("--no-metadata", action="store_false", dest="metadata")
    p.add_argument("--verify", action="store_true", default=True)
    p.add_argument("--no-verify", action="store_false", dest="verify")

    # Execution
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

    n_classes = 2 if args.class_scheme == "binary" else 8
    target_classes = BINARY_CLASSES if args.class_scheme == "binary" else MACHINE_CONDITION_CLASSES

    print("═" * 60)
    print(f"AW-DPCNN MIMII Dataset Builder ({n_classes}‑class, {args.class_scheme})")
    print("═" * 60)
    print(f"  Input dir      : {args.input_dir}")
    print(f"  Output dir     : {args.output_dir}")
    print(f"  Noise level    : {args.noise_level}")
    print(f"  Class scheme   : {args.class_scheme} ({n_classes} classes)")
    print(f"  Sample rate    : {args.sr} Hz")
    print(f"  Window / Hop   : {args.win_len} / {args.hop_len}"
          + (f"  (max {args.max_windows_per_file}/file)" if args.max_windows_per_file > 0 else ""))
    print(f"  File split     : {args.file_split}  (seed={args.split_seed})")
    print(f"  Workers        : {args.workers}")
    print(f"  TF method      : {args.tf_method}")
    print(f"  Dry run        : {args.dry_run}")
    print("═" * 60)

    # ── Collect files ──
    files_by_class, skipped = _collect_files(
        args.input_dir, args.noise_level, args.class_scheme,
    )

    total_files = sum(len(v) for v in files_by_class.values())
    n_found = len(files_by_class)

    print(f"\nSelected {total_files} .wav files → {n_found} classes:")
    for cls in target_classes:
        if cls in files_by_class:
            items = files_by_class[cls]
            machines = sorted({it[3] for it in items})
            print(f"  ✓ {cls:<20s} {len(items):5d} file(s)  (machines: {', '.join(machines)})")
        else:
            print(f"  ✗ {cls:<20s} — NOT in source data")

    if skipped:
        print(f"\nSkipped {len(skipped)} files (first 5):")
        for fname, reason in sorted(skipped)[:5]:
            print(f"  {fname:<40s} reason={reason}")

    if not files_by_class:
        print("\n[ERROR] No matching files found. Aborting.")
        sys.exit(1)

    # ── File‑level split ──
    train_map, val_map, test_map = _file_level_split(
        files_by_class, ratios, args.split_seed,
    )
    split_map = {"train": train_map, "val": val_map, "test": test_map}

    print(f"\nFile‑level split (seed={args.split_seed}):")
    print(f"  {'Class':<20s} {'Total':>6s}  {'Train':>6s}  {'Val':>6s}  {'Test':>6s}")
    print(f"  {'-'*52}")
    t_files = v_files = te_files = 0
    for cls in target_classes:
        if cls not in files_by_class:
            continue
        n = len(files_by_class[cls])
        nt = len(train_map.get(cls, []))
        nv = len(val_map.get(cls, []))
        nte = len(test_map.get(cls, []))
        t_files += nt; v_files += nv; te_files += nte
        warn = "  ⚠" if (nt == 0 or nv == 0 or nte == 0) else ""
        print(f"  {cls:<20s} {n:6d}  {nt:6d}  {nv:6d}  {nte:6d}{warn}")
    print(f"  {'-'*52}")
    print(f"  {'TOTAL':<20s} {total_files:6d}  {t_files:6d}  {v_files:6d}  {te_files:6d}")

    # Leakage check (full path — same filename in different noise levels ≠ leakage)
    train_names = {it[0] for items in train_map.values() for it in items}
    val_names = {it[0] for items in val_map.values() for it in items}
    test_names = {it[0] for items in test_map.values() for it in items}
    tv = train_names & val_names
    tt = train_names & test_names
    vt = val_names & test_names
    if tv or tt or vt:
        print(f"\n  [FAIL] File‑level LEAKAGE! train∩val={len(tv)} train∩test={len(tt)} val∩test={len(vt)}")
    else:
        print(f"\n  [OK] No file‑level leakage ({total_files} files each in exactly one split)")

    if args.dry_run:
        print("\n[Dry run complete — no images generated.]")
        return

    # ── Collect file‑level tasks (NO audio loaded yet) ──
    file_tasks, file_meta = _collect_window_tasks(
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
        tf_method=args.tf_method,
        overwrite=args.overwrite,
        max_windows=args.max_windows_per_file,
    )

    print(f"\n[INFO] Total files to process: {len(file_tasks)}")
    if not file_tasks:
        print("[INFO] Nothing to do — all images already exist (use --overwrite).")
        return

    # ── Generate (parallel — each worker loads its own audio) ──
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(tqdm(
            executor.map(_process_one_file, file_tasks, chunksize=4),
            total=len(file_tasks),
            desc="Building fused dataset",
            ncols=100,
        ))

    total_images = sum(results)
    print(f"\n✅ Done — {total_images} images written to {args.output_dir}")

    # ── Reconstruct window‑level metadata from output directory ──
    if args.metadata:
        metadata_rows = []
        for split_name in ["train", "val", "test"]:
            split_dir = os.path.join(args.output_dir, split_name)
            if not os.path.isdir(split_dir):
                continue
            for cls_dir in sorted(Path(split_dir).iterdir()):
                if not cls_dir.is_dir():
                    continue
                cls_name = cls_dir.name
                for png in sorted(cls_dir.glob("*.png")):
                    # Parse metadata from filename: noise_id_originalstem_windowidx.png
                    fname = png.name
                    parts = fname.rsplit("_", 1)
                    if len(parts) == 2 and parts[1].endswith(".png"):
                        window_part = parts[1].replace(".png", "")
                        try:
                            window_idx = int(window_part)
                        except ValueError:
                            window_idx = 0
                    else:
                        window_idx = 0

                    metadata_rows.append({
                        "filename": fname,
                        "class_label": cls_name,
                        "split": split_name,
                        "window_idx": window_idx,
                    })

        if metadata_rows:
            csv_path = os.path.join(args.output_dir, "metadata.csv")
            _save_metadata(csv_path, metadata_rows)
            print(f"📋 Metadata saved to {csv_path} ({len(metadata_rows)} rows)")

            # ── Verification ──
            if args.verify:
                print("\n" + "─" * 60)
                print("Distribution verification")
                print("─" * 60)

                split_counts: dict = defaultdict(lambda: defaultdict(int))
                for row in metadata_rows:
                    split_counts[row["split"]][row["class_label"]] += 1

                for split_name in ["train", "val", "test"]:
                    counts = dict(sorted(split_counts[split_name].items()))
                    total = sum(counts.values())
                    print(f"  {split_name}: {total:6d} samples  {counts}")

                splits_present = [s for s in ["train", "val", "test"] if s in split_counts]
                all_ok = True
                for i in range(len(splits_present)):
                    for j in range(i + 1, len(splits_present)):
                        sa, sb = splits_present[i], splits_present[j]
                        js = _compute_js_divergence(split_counts[sa], split_counts[sb])
                        status = "OK" if js < 0.01 else "WARN"
                        if js >= 0.01:
                            all_ok = False
                        print(f"  JS({sa}, {sb}) = {js:.6f}  [{status}]")

                if all_ok:
                    print("\n  [OK] All JS divergences < 0.01 — distribution consistent across splits.")
                else:
                    print("\n  [WARN] JS divergence ≥ 0.01 — review the split.")
                print("─" * 60)


if __name__ == "__main__":
    main()
