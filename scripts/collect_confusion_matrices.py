#!/usr/bin/env python3
"""
Confusion Matrix Collector & Renamer
=====================================
Finds all ``confusion_matrix.png`` files under ``experiments/experiment_result/``
and copies them to a flat output directory with a descriptive filename:

    confusion_matrix_{dataset}_{model}_{experiment}_{trial}.png

Usage::

    python scripts/collect_confusion_matrices.py
    python scripts/collect_confusion_matrices.py --output-dir paper/figures/confusion_matrix
    python scripts/collect_confusion_matrices.py --dry-run
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEARCH_ROOT = _PROJECT_ROOT / "experiments" / "experiment_result"


def parse_path(cm_path: Path) -> Optional[Tuple[str, str, str, str]]:
    """Extract (dataset, model, experiment, trial) from a CM file path.

    Handles three directory layouts::

        exp1/{model}/{dataset}/{exp_name}/trial_seed{seed}/figures/confusion_matrix.png
        rep_compare/{combo}/{exp_name}/trial_seed{seed}/figures/confusion_matrix.png
        ablation_results/{B#}/{exp_name}/trial_seed{seed}/figures/confusion_matrix.png
    """
    rel = cm_path.relative_to(SEARCH_ROOT)
    parts = rel.parts

    # Last 4 parts should be: exp_name / trial_seed* / figures / confusion_matrix.png
    if len(parts) < 4:
        return None
    if parts[-1] != "confusion_matrix.png" or parts[-2] != "figures":
        return None

    trial = parts[-3]  # e.g. trial_seed42
    exp_name = parts[-4]  # e.g. exp1_resnet18 or rep_compare_cwt_mtf

    # Determine model and dataset from remaining path prefix
    prefix = parts[:-4]  # everything before exp_name

    dataset = "unknown"
    model = "unknown"

    if len(prefix) >= 1:
        group = prefix[0]  # exp1, rep_compare, ablation_results

        if group == "exp1" and len(prefix) >= 2:
            # exp1/{model}/{dataset}/...
            model = prefix[1]
            if len(prefix) >= 3:
                dataset = prefix[2]

        elif group == "rep_compare" and len(prefix) >= 2:
            # rep_compare/{combo}/...  (combo like cwt_gadf, mel_mtf)
            model = "msca-vgg16"  # rep_compare always uses MSCA-VGG16
            dataset = prefix[1]   # combo name

        elif group == "ablation_results" and len(prefix) >= 2:
            # ablation_results/{B#}/...
            model = prefix[1]  # B0, B1, ..., B8
            dataset = "ablation"

        elif len(prefix) == 1:
            model = group
            dataset = "unknown"

    return (dataset, model, exp_name, trial)


def collect_and_copy(output_dir: Path, dry_run: bool = False) -> int:
    """Find all CM files, compute new names, and copy them.

    Returns the number of files processed.
    """
    cm_files = sorted(SEARCH_ROOT.rglob("confusion_matrix.png"))
    if not cm_files:
        print("[WARN] No confusion_matrix.png files found.")
        return 0

    print(f"Found {len(cm_files)} confusion matrix file(s).\n")

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0
    for cm_path in cm_files:
        info = parse_path(cm_path)
        if info is None:
            print(f"  [SKIP] Cannot parse: {cm_path}")
            skipped += 1
            continue

        dataset, model, exp_name, trial = info
        new_name = f"confusion_matrix_{dataset}_{model}_{exp_name}_{trial}.png"
        dest = output_dir / new_name

        if dest.exists() and not dry_run:
            print(f"  [SKIP] Already exists: {new_name}")
            skipped += 1
            continue

        if dry_run:
            print(f"  [DRY-RUN] {cm_path.name}  →  {new_name}")
        else:
            shutil.copy2(cm_path, dest)
            print(f"  [COPY] {new_name}")
        copied += 1

    return copied


def main():
    parser = argparse.ArgumentParser(
        description="Collect and rename confusion matrix PNGs")
    parser.add_argument("--output-dir",
                        default="paper/figures/confusion_matrix",
                        help="Destination directory for renamed files")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without copying")
    args = parser.parse_args()

    print(f"Search root: {SEARCH_ROOT}")
    print(f"Output dir:  {args.output_dir}")
    if args.dry_run:
        print("Mode:        DRY-RUN (no files will be copied)")
    print()

    count = collect_and_copy(Path(args.output_dir), dry_run=args.dry_run)

    if args.dry_run:
        print(f"\n{count} file(s) would be copied.")
    else:
        print(f"\n{count} file(s) copied to {args.output_dir}")


if __name__ == "__main__":
    main()
