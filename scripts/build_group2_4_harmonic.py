#!/usr/bin/env python3
"""
Build AW-DPCNN Fused Dataset — Group 2‑4 Transformer Harmonic (9‑class)
=======================================================================

Reads **unsplit** .wav files from::

    raw-data/Group2_4（original）/G{group}_{condition}.wav

Selects the **9 harmonic‑only classes**, performs a **file‑level** stratified
train/val/test split, and produces AW‑DPCNN fused images in::

    datasets/Group2_4_harmonic/
        train/
            10pThirdHarmonic/  10pFifthHarmonic/  ...  30pSeventhHarmonic/
        val/
            ...
        test/
            ...
        metadata.csv

Source file naming convention
-----------------------------
``G{group}_{condition}.wav`` where `group ∈ {2, 3, 4}` and `condition`
maps to a target class:

===========================  ========================  ======================
Source condition              Target class              Group availability
===========================  ========================  ======================
10pThirdHarmonic              10pThirdHarmonic          G3, G4
10pFifthHarmonic              10pFifthHarmonic          G3, G4
10pSeventhHarmonic            10pSeventhHarmonic        G3, G4
20pThirdHarmonic              20pThirdHarmonic          G3, G4
20pFifthHarmonic              20pFifthHarmonic          G3, G4
20pSeventhHarmonic            20pSeventhHarmonic        G3, G4
30pThirdHarmonic              30pThirdHarmonic          G2, G3, G4
30pFifthHarmonic              30pFifthHarmonic          G2, G3, G4
30pSeventhHarmonic            30pSeventhHarmonic        G2, G3, G4
pureThirdHarmonic             —                         excluded
pureFifthHarmonic             —                         excluded
pureSeventhHarmonic           —                         excluded
10kvOverload                  —                         excluded
11kvOvervoltage               —                         excluded
NoLoad                        —                         excluded
===========================  ========================  ======================

Pipeline (per sliding window)
-----------------------------
  1. Mel spectrogram (pseudo‑colour)
  2. GADF image (pseudo‑colour)
  3. AW‑DPCNN fusion (γ = 4, N = 20)

!!! Check the source directory and parameters if needed

Usage::

    # Default parameters (recommended)
    python scripts/build_group2_4_harmonic.py

    # Dry-run (preview the split plan)
    python scripts/build_group2_4_harmonic.py --dry-run

    # Custom split ratio
    python scripts/build_group2_4_harmonic.py --file-split 50,25,25 --workers 16

    python scripts/build_group2_4_harmonic.py  --win-len 8192 --hop-len 8192 --n-fft 4096 --n-iter 10 --sequence-length 300 --gamma 10 --workers 32
"""

import argparse
import csv
import os
import re
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

# ── Reuse the core AW‑DPCNN pipeline ──
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_CWRU_dataset import (  # noqa: E402
    process_one_window,
)


# ═══════════════════════════════════════════════════════════════════════
#  Class mapping — source condition → target class name
# ═══════════════════════════════════════════════════════════════════════

CONDITION_TO_CLASS: Dict[str, str] = {
    "10pThirdHarmonic":     "10pThirdHarmonic",
    "10pFifthHarmonic":     "10pFifthHarmonic",
    "10pSeventhHarmonic":   "10pSeventhHarmonic",
    "20pThirdHarmonic":     "20pThirdHarmonic",
    "20pFifthHarmonic":     "20pFifthHarmonic",
    "20pSeventhHarmonic":   "20pSeventhHarmonic",
    "30pThirdHarmonic":     "30pThirdHarmonic",
    "30pFifthHarmonic":     "30pFifthHarmonic",
    "30pSeventhHarmonic":   "30pSeventhHarmonic",
}

# Conditions that are explicitly excluded (not in the 9 target classes)
EXCLUDED_CONDITIONS: set = {
    "pureThirdHarmonic", "pureFifthHarmonic", "pureSeventhHarmonic",
    "10kvOverload", "11kvOvervoltage", "NoLoad",
}

# Target classes (ordered for consistent reporting)
ALL_TARGET_CLASSES: List[str] = [
    "10pThirdHarmonic", "10pFifthHarmonic", "10pSeventhHarmonic",
    "20pThirdHarmonic", "20pFifthHarmonic", "20pSeventhHarmonic",
    "30pThirdHarmonic", "30pFifthHarmonic", "30pSeventhHarmonic",
]

FILE_PATTERN = re.compile(r"^G(\d+)_(.+)\.wav$")


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════

def _collect_and_filter(input_dir: str) -> Tuple[Dict[str, list], list, list]:
    """Scan *input_dir* for .wav files, filter to target classes.

    Returns
    -------
    (files_by_class, excluded_files, missing_classes)
        files_by_class  – {target_class: [(path, sr, signal, group, condition), ...]}
        excluded_files  – list of (filename, condition, reason)
        missing_classes – list of target classes not found in source data
    """
    files_by_class: dict = defaultdict(list)
    excluded: list = []
    seen_conditions: set = set()

    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"[ERROR] Input directory not found: {input_path}")
        sys.exit(1)

    for wav_path in sorted(input_path.glob("*.wav")):
        fname = wav_path.name
        m = FILE_PATTERN.match(fname)
        if not m:
            excluded.append((fname, "?", "filename pattern mismatch"))
            continue

        group = m.group(1)
        condition = m.group(2)
        seen_conditions.add(condition)

        if condition in EXCLUDED_CONDITIONS:
            excluded.append((fname, condition, "not in 9 harmonic target classes"))
            continue

        target_class = CONDITION_TO_CLASS.get(condition)
        if target_class is None:
            excluded.append((fname, condition, "no class mapping defined"))
            continue

        try:
            sr, data = wavfile.read(str(wav_path))
        except Exception as exc:
            print(f"[WARN] Cannot read {wav_path}: {exc}")
            continue

        if data.ndim > 1:
            data = data.mean(axis=1)
        data = data.astype(np.float32)

        files_by_class[target_class].append(
            (str(wav_path), sr, data, group, condition)
        )

    # Determine missing classes
    present = set(files_by_class.keys())
    missing = [c for c in ALL_TARGET_CLASSES if c not in present]

    return dict(files_by_class), excluded, missing


def _file_level_split_small(
    files_by_class: Dict[str, list],
    ratios: Tuple[float, float, float],
    seed: int = 42,
) -> Tuple[dict, dict, dict]:
    """Duration‑aware stratified file‑level split with chunking fallback.

    For classes with fewer than 3 source files, the longest file is
    split into contiguous chunks (treated as independent files for
    split purposes) so that every class can populate all three splits.
    All windows from a chunk go to the same split — no frame‑level
    leakage.
    """
    r_train, r_val, r_test = ratios
    total_r = r_train + r_val + r_test
    r_train, r_val, r_test = r_train / total_r, r_val / total_r, r_test / total_r

    # Window estimator
    def _n_windows(sig, win=8192, hop=4096):
        if len(sig) <= win:
            return 1
        return len(range(0, len(sig) - win + 1, hop))

    train_map, val_map, test_map = {}, {}, {}

    for cls, items in sorted(files_by_class.items()):
        n = len(items)

        # If fewer than 3 files, chunk the longest file to reach ≥ 3
        if n < 3:
            # Sort by signal length descending to find the longest
            items_sorted = sorted(items, key=lambda it: len(it[2]), reverse=True)
            longest = items_sorted[0]
            sig = longest[2]
            total_len = len(sig)
            needed = 3 - n  # how many additional chunks we need

            # Split the longest file into (needed + 1) equal chunks
            n_chunks = needed + 1
            chunk_len = total_len // n_chunks

            # Create chunk items — each inherits metadata from the original
            # but with modified signal and an augmented path stem
            chunks = []
            orig_path, sr, _, group, condition = longest
            for i in range(n_chunks):
                start = i * chunk_len
                end = total_len if i == n_chunks - 1 else (i + 1) * chunk_len
                chunk_sig = sig[start:end].copy()
                chunk_path = f"{orig_path}__chunk{i}"
                chunks.append((chunk_path, sr, chunk_sig, group, condition))

            # Replace items: keep shorter files as-is, replace longest with chunks
            new_items = items_sorted[1:] + chunks
            items = new_items
            n = len(items)  # should now be ≥ 3

        tagged = [(_n_windows(it[2]), i, it) for i, it in enumerate(items)]
        tagged.sort(key=lambda x: x[0], reverse=True)

        # Target file counts per split
        n_train = max(1, round(n * r_train))
        n_val = max(1, round(n * r_val))
        n_test = max(1, n - n_train - n_val)
        # Adjust if sum overflows
        while n_train + n_val + n_test > n:
            if n_test > 1:
                n_test -= 1
            elif n_val > 1:
                n_val -= 1
            else:
                n_train -= 1

        # Greedy balanced multi‑way partition
        bins = [
            {"wins": 0, "items": [], "target": n_train, "name": "train"},
            {"wins": 0, "items": [], "target": n_val, "name": "val"},
            {"wins": 0, "items": [], "target": n_test, "name": "test"},
        ]

        for w, orig_idx, it in tagged:
            candidates = [(b["wins"], i, b) for i, b in enumerate(bins)
                          if len(b["items"]) < b["target"]]
            if candidates:
                candidates.sort(key=lambda x: x[0])
                _, _, chosen = candidates[0]
            else:
                bins.sort(key=lambda b: b["wins"])
                chosen = bins[0]

            chosen["wins"] += w
            chosen["items"].append(it)

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
#  Task collector
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
    overwrite: bool,
    sequence_length: int = None,
) -> Tuple[list, list]:
    """Build task list + metadata rows from file‑level split maps."""
    tasks = []
    metadata_rows = []

    for split_name in ["train", "val", "test"]:
        split_cls = split_map.get(split_name, {})
        for cls, items in sorted(split_cls.items()):
            out_cls_dir = os.path.join(output_dir, split_name, cls)
            for (fpath, sr_val, signal, group, condition) in items:
                # Clean up chunk suffix for filenames
                name = Path(fpath).name
                if "__chunk" in name:
                    stem = name.replace(".wav__chunk", "_chunk")
                else:
                    stem = Path(fpath).stem

                if win_len <= 0 or win_len >= len(signal):
                    out_path = os.path.join(out_cls_dir, f"{stem}.png")
                    if not overwrite and os.path.exists(out_path):
                        continue
                    tasks.append((
                        signal, sr_val, out_path, img_size, n_iter,
                        n_fft, hop_length, n_mels, fmax, cmap,
                        gaf_method, False, gamma, sequence_length,
                    ))
                    metadata_rows.append({
                        "filename": f"{stem}.png",
                        "class_label": cls,
                        "source_file": os.path.basename(fpath),
                        "group": group,
                        "condition_raw": condition,
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
                        ))
                        metadata_rows.append({
                            "filename": fname,
                            "class_label": cls,
                            "source_file": os.path.basename(fpath),
                            "group": group,
                            "condition_raw": condition,
                            "split": split_name,
                            "window_idx": idx,
                            "window_start_sample": start,
                        })
                        idx += 1

    return tasks, metadata_rows


def _save_metadata(csv_path: str, rows: list):
    fieldnames = [
        "filename", "class_label", "source_file", "group",
        "condition_raw", "split", "window_idx", "window_start_sample",
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
        description="Build AW-DPCNN fused dataset for Group 2‑4 transformer (9 harmonic classes)",
    )
    p.add_argument("--input-dir", default="raw-data/Group2_4（original）",
                   help="Directory of unsplit .wav files")
    p.add_argument("--output-dir", default="datasets/Group2_4_harmonic",
                   help="Root dir for fused PNG images (ImageFolder layout)")
    p.add_argument("--sr", type=int, default=44100,
                   help="Sample rate (default: 44100)")

    # Sliding window
    p.add_argument("--win-len", type=int, default=8192,
                   help="Window length in samples (0 = one image per file)")
    p.add_argument("--hop-len", type=int, default=4096,
                   help="Hop length between windows")

    # Image
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--cmap", default="viridis")

    # Mel
    p.add_argument("--n-fft", type=int, default=2048)
    p.add_argument("--hop-length", type=int, default=256)
    p.add_argument("--n-mels", type=int, default=128)
    p.add_argument("--fmax", type=int, default=8000)

    # GADF
    p.add_argument("--gaf-method", default="difference",
                   choices=["difference", "summation"])
    p.add_argument("--sequence-length", type=int, default=None,
                   help="Max time‑steps for GAF (resample if longer). "
                        "None = use full signal length.")

    # AW-DPCNN
    p.add_argument("--n-iter", type=int, default=20)
    p.add_argument("--gamma", type=float, default=4.0)

    # File‑level split
    p.add_argument("--file-split", type=str, default="60,20,20",
                   help="Train/val/test ratios, e.g. '60,20,20'")
    p.add_argument("--split-seed", type=int, default=42)

    # Metadata & verification
    p.add_argument("--metadata", action="store_true", default=True)
    p.add_argument("--no-metadata", action="store_false", dest="metadata")
    p.add_argument("--verify", action="store_true", default=True)
    p.add_argument("--no-verify", action="store_false", dest="verify")

    # Execution
    p.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the split plan without generating images")

    return p


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    args = build_parser().parse_args()
    cmap_code = _parse_cmap(args.cmap)

    # Parse split ratios
    parts = [float(x.strip()) for x in args.file_split.split(",")]
    if len(parts) != 3:
        print("[ERROR] --file-split requires three values, e.g. '60,20,20'")
        sys.exit(1)
    total_r = sum(parts)
    ratios = tuple(p / total_r for p in parts)

    # ── Header ──
    print("═" * 60)
    print("AW-DPCNN Group 2‑4 Harmonic Dataset Builder (9‑class)")
    print("═" * 60)
    print(f"  Input dir      : {args.input_dir}")
    print(f"  Output dir     : {args.output_dir}")
    print(f"  Sample rate    : {args.sr} Hz")
    print(f"  Window / Hop   : {args.win_len} / {args.hop_len}")
    print(f"  GAF seq len    : {args.sequence_length if args.sequence_length else 'full signal'}")
    print(f"  File split     : {args.file_split}  (seed={args.split_seed})")
    print(f"  Workers        : {args.workers}")
    print(f"  Dry run        : {args.dry_run}")
    print("═" * 60)

    # ── Collect & filter ──
    files_by_class, excluded, missing = _collect_and_filter(args.input_dir)

    total_files = sum(len(v) for v in files_by_class.values())
    n_classes = len(files_by_class)

    print(f"\nSelected {total_files} .wav files → {n_classes} classes:")
    for cls in ALL_TARGET_CLASSES:
        if cls in files_by_class:
            items = files_by_class[cls]
            groups = sorted({it[3] for it in items})
            print(f"  ✓ {cls:<22s} {len(items):2d} file(s)  (groups: {', '.join(groups)})")
        else:
            print(f"  ✗ {cls:<22s} — NOT in source data")

    if missing:
        print(f"\n⚠  Missing target classes: {', '.join(missing)}")
    if excluded:
        print(f"\nExcluded {len(excluded)} files:")
        for fname, cond, reason in sorted(excluded):
            print(f"  {fname:<40s} cond={cond:<20s} reason={reason}")

    if not files_by_class:
        print("\n[ERROR] No matching files found. Aborting.")
        sys.exit(1)

    # ── File‑level split ──

    train_map, val_map, test_map = _file_level_split_small(
        files_by_class, ratios, args.split_seed,
    )
    split_map = {"train": train_map, "val": val_map, "test": test_map}

    # ── Print split plan ──
    print(f"\nFile‑level split (seed={args.split_seed}):")
    print(f"  {'Class':<22s} {'Total':>6s}  {'Train':>6s}  {'Val':>6s}  {'Test':>6s}")
    print(f"  {'-'*52}")
    t_files = v_files = te_files = 0
    for cls in ALL_TARGET_CLASSES:
        if cls not in files_by_class:
            continue
        n = len(files_by_class[cls])
        nt = len(train_map.get(cls, []))
        nv = len(val_map.get(cls, []))
        nte = len(test_map.get(cls, []))
        t_files += nt; v_files += nv; te_files += nte
        warn = "  ⚠" if (nt == 0 or nv == 0 or nte == 0) else ""
        print(f"  {cls:<22s} {n:6d}  {nt:6d}  {nv:6d}  {nte:6d}{warn}")
    print(f"  {'-'*52}")
    print(f"  {'TOTAL':<22s} {total_files:6d}  {t_files:6d}  {v_files:6d}  {te_files:6d}")

    # Leakage check
    train_names = {Path(it[0]).name for items in train_map.values() for it in items}
    val_names = {Path(it[0]).name for items in val_map.values() for it in items}
    test_names = {Path(it[0]).name for items in test_map.values() for it in items}
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

    # ── Collect window tasks ──
    tasks, metadata_rows = _collect_window_tasks(
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
        sequence_length=args.sequence_length,
    )

    print(f"\n[INFO] Total fusion tasks: {len(tasks)}")
    if not tasks:
        print("[INFO] Nothing to do — all images already exist (use --overwrite).")
        return

    # ── Generate (parallel) ──
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

    # ── Verification ──
    if args.verify and metadata_rows:
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
