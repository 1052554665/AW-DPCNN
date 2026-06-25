#!/usr/bin/env python3
"""
Representation Comparison — Batch Runner
=========================================
Generates experiment configs for all 12 (TF × temporal) combinations
using the standard VGG16-BN classifier, then runs them sequentially.

All combinations share identical training settings — only the dataset
varies, isolating the effect of the input representation.

Usage::

    # Generate configs + run all 12
    python scripts/run_rep_compare.py --num-workers 32

    # Generate configs only
    python scripts/run_rep_compare.py --gen-only

    # Run a single combination
    python scripts/run_rep_compare.py --combo mel_gadf
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────

COMBO_NAMES = [
    "mel_gadf", "mel_gasf", "mel_mtf", "mel_rp",
    "stft_gadf", "stft_gasf", "stft_mtf", "stft_rp",
    "cwt_gadf", "cwt_gasf", "cwt_mtf", "cwt_rp",
]

DATASET_ROOT = "datasets/rep_compare_12k_de_split"
CONFIG_DIR = Path("experiments/rep_compare")
RESULT_ROOT = Path("experiments/experiment_result/rep_compare")

CONFIG_TEMPLATE = """\
experiment_name: rep_compare_{combo}
device: cuda

dataset:
  root_dir: ./datasets/rep_compare_12k_de_split/{combo}
  num_classes: 10
  {num_workers_line}

model:
  name: vgg16
  num_classes: 10
  pretrained: false

train:
  epochs: 30
  lr: 1e-4
  optimizer: adamw
  weight_decay: 1e-3

scheduler:
  type: plateau
  mode: max
  factor: 0.5
  patience: 5
  min_lr: 0.000001

visualization:
  tsne: true
  confusion_matrix: true
  roc_curve: true

output:
  root_dir: ./experiments/experiment_result/rep_compare/{combo}
"""


def generate_configs(num_workers: int = 0):
    """Write all 12 experiment YAML files."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    nw_line = f"num_workers: {num_workers}" if num_workers > 0 else ""
    for combo in COMBO_NAMES:
        yaml_path = CONFIG_DIR / f"{combo}.yaml"
        content = CONFIG_TEMPLATE.format(
            combo=combo,
            num_workers_line=nw_line,
        )
        yaml_path.write_text(content)
        print(f"  Created {yaml_path}")


def run_one(combo: str, seed: int = 42) -> bool:
    """Run a single representation comparison experiment."""
    exp_config = CONFIG_DIR / f"{combo}.yaml"
    dataset_path = Path(DATASET_ROOT) / combo

    if not dataset_path.exists():
        print(f"[SKIP] Dataset not found: {dataset_path}")
        return False

    cmd = [
        sys.executable, "scripts/train.py",
        "--config", "configs/default.yaml",
        "--exp-config", str(exp_config),
        "--seed", str(seed),
    ]

    print(f"\n{'='*60}")
    print(f"  Running: {combo}  (seed={seed})")
    print(f"  Dataset: {dataset_path}")
    print(f"{'='*60}")

    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Representation Comparison — Batch Runner (VGG16-BN)",
    )
    parser.add_argument("--gen-only", action="store_true",
                        help="Only generate config files, don't run")
    parser.add_argument("--combo", default=None,
                        help="Run a single combination (e.g., mel_gadf)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Number of DataLoader workers (0 = use default config)")
    args = parser.parse_args()

    # ── Generate configs ──
    print("Generating experiment configs...")
    generate_configs(num_workers=args.num_workers)

    if args.gen_only:
        print("\nConfigs generated. Run with: python scripts/run_rep_compare.py")
        return

    # ── Run ──
    combos_to_run = [args.combo] if args.combo else COMBO_NAMES

    success, fail = 0, 0
    for combo in combos_to_run:
        ok = run_one(combo, seed=args.seed)
        if ok:
            success += 1
        else:
            fail += 1
            print(f"[FAIL] {combo} returned non-zero exit code")

    print(f"\n{'='*60}")
    print(f"  Done: {success} succeeded, {fail} failed")
    if fail:
        print(f"  Failed: check logs above")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
