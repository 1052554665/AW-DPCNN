#!/usr/bin/env python3
"""
Post‑hoc file‑level split for representation comparison datasets
=================================================================
Reads an unsplit ImageFolder dataset (class sub‑folders + metadata.csv),
performs a file‑level train/val/test split, and creates a new directory
with ``train/``, ``val/``, ``test/`` sub‑folders containing **symlinks**
to the original images.

This avoids rebuilding the expensive fused images while providing the
split structure required by the training pipeline.

Usage::

    # Split a single combination
    python scripts/split_rep_compare.py \
        --input-dir datasets/rep_compare_12k_de/mel_gadf \
        --output-dir datasets/rep_compare_12k_de_split/mel_gadf

    # Split all 12 combinations for one dataset
    python scripts/split_rep_compare.py \
        --input-root datasets/rep_compare_12k_de \
        --output-root datasets/rep_compare_12k_de_split

    # Custom ratios
    python scripts/split_rep_compare.py \
        --input-root datasets/rep_compare_12k_de \
        --output-root datasets/rep_compare_12k_de_split \
        --file-split 60,20,20 --split-seed 42
"""

import argparse
import csv
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def _file_level_split(
    files: List[str],
    ratios: tuple,
    seed: int = 42,
) -> tuple:
    """Split a list of identifiers into three groups."""
    rng = random.Random(seed)
    files = sorted(set(files))
    rng.shuffle(files)
    n = len(files)
    r_train, r_val, r_test = ratios
    n_train = max(1, round(n * r_train))
    n_val = max(1, round(n * r_val))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = max(1, n - n_train - 1)
    return files[:n_train], files[n_train:n_train + n_val], files[n_train + n_val:]


def split_one(
    input_dir: str,
    output_dir: str,
    ratios: tuple = (0.6, 0.2, 0.2),
    seed: int = 42,
    dry_run: bool = False,
) -> None:
    """Split a single unsplit dataset into train/val/test via symlinks."""

    inp = Path(input_dir)
    out = Path(output_dir)

    if not inp.exists():
        print(f"[ERROR] Input directory not found: {inp}")
        return

    # ── Read metadata to get source_file per image ──
    csv_path = inp / "metadata.csv"
    if not csv_path.exists():
        print(f"[ERROR] No metadata.csv in {inp}")
        return

    img_to_source: Dict[str, str] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_to_source[row["filename"]] = row["source_file"]

    # ── Group images by class → source_file ──
    cls_source_files: Dict[str, List[str]] = defaultdict(list)
    cls_images: Dict[str, Dict[str, List[str]]] = defaultdict(
        lambda: defaultdict(list))

    for cls_dir in sorted(inp.iterdir()):
        if not cls_dir.is_dir():
            continue
        cls_name = cls_dir.name
        for png in sorted(cls_dir.glob("*.png")):
            fname = png.name
            src = img_to_source.get(fname, fname)
            cls_source_files[cls_name].append(src)
            cls_images[cls_name][src].append(fname)

    if not cls_source_files:
        print(f"[ERROR] No images found in {inp}")
        return

    # ── File‑level split per class ──
    split_map = {"train": {}, "val": {}, "test": {}}
    for cls_name, sources in cls_source_files.items():
        train_src, val_src, test_src = _file_level_split(sources, ratios, seed)
        split_map["train"][cls_name] = set(train_src)
        split_map["val"][cls_name] = set(val_src)
        split_map["test"][cls_name] = set(test_src)

    # ── Create symlinks ──
    total_links = 0
    for split_name in ["train", "val", "test"]:
        split_cls_srcs = split_map[split_name]
        for cls_name, allowed_srcs in split_cls_srcs.items():
            out_cls_dir = out / split_name / cls_name
            if dry_run:
                count = sum(
                    len(cls_images[cls_name].get(src, []))
                    for src in allowed_srcs
                )
                total_links += count
                continue

            out_cls_dir.mkdir(parents=True, exist_ok=True)
            for src in allowed_srcs:
                for fname in cls_images[cls_name].get(src, []):
                    src_path = (inp / cls_name / fname).resolve()
                    dst_path = out_cls_dir / fname
                    if not dst_path.exists():
                        dst_path.symlink_to(src_path)
                        total_links += 1

    if dry_run:
        print(f"  Would create {total_links} symlinks in {out}")
    else:
        print(f"  Created {total_links} symlinks in {out}")
        # Copy metadata.csv to output root
        import shutil
        shutil.copy2(csv_path, out / "metadata.csv")


def main():
    parser = argparse.ArgumentParser(
        description="File‑level split for representation comparison datasets",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input-dir", default=None,
                       help="Single dataset directory to split")
    group.add_argument("--input-root", default=None,
                       help="Root containing multiple combo sub‑directories")

    parser.add_argument("--output-dir", default=None,
                        help="Output directory for single split")
    parser.add_argument("--output-root", default=None,
                        help="Output root for batch split")
    parser.add_argument("--file-split", type=str, default="60,20,20")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    parts = [float(x.strip()) for x in args.file_split.split(",")]
    ratios = tuple(p / sum(parts) for p in parts)

    if args.input_dir:
        out_dir = args.output_dir or (args.input_dir + "_split")
        split_one(args.input_dir, out_dir, ratios, args.split_seed, args.dry_run)
    else:
        inp_root = Path(args.input_root)
        out_root = Path(args.output_root or (str(args.input_root) + "_split"))
        if not inp_root.exists():
            print(f"[ERROR] Input root not found: {inp_root}")
            sys.exit(1)

        combo_dirs = sorted(
            d for d in inp_root.iterdir()
            if d.is_dir() and (d / "metadata.csv").exists()
        )
        if not combo_dirs:
            # Maybe the combos are deeper
            combo_dirs = sorted(
                d for d in inp_root.iterdir()
                if d.is_dir() and any(
                    sd.is_dir() for sd in d.iterdir()
                    if sd.name not in ("metadata.csv",)
                )
            )

        print(f"Found {len(combo_dirs)} combinations to split:")
        for cd in combo_dirs:
            print(f"  {cd.name}")

        for cd in combo_dirs:
            print(f"\n{'='*50}\n  {cd.name}\n{'='*50}")
            split_one(str(cd), str(out_root / cd.name),
                      ratios, args.split_seed, args.dry_run)

    print("\n✓ Done.")


if __name__ == "__main__":
    main()
