#!/usr/bin/env python3
"""
Representation Comparison — Batch Runner with Repeated Trials
===============================================================
Generates experiment configs for all 12 (TF × temporal) combinations
using the standard VGG16-BN classifier, then runs them with optional
repeated independent trials for statistical validation.

Single-trial mode (quick test)::

    python scripts/run_rep_compare.py --num-workers 32

Repeated independent trials (3 seeds, publication-ready)::

    python scripts/run_rep_compare.py --num-workers 32 --num-trials 3

    # Generate configs only
    python scripts/run_rep_compare.py --gen-only

    # Run a single combination with 3 trials
    python scripts/run_rep_compare.py --combo mel_gadf --num-trials 3

    # Aggregate existing results without re-training
    python scripts/run_rep_compare.py --aggregate-only --num-trials 3

Output (per combination)::

    experiments/experiment_result/rep_compare/{combo}/
        rep_compare_{combo}/
            trial_seed42/results/test_metrics.json
            trial_seed123/results/test_metrics.json
            trial_seed456/results/test_metrics.json
            aggregated/
                aggregated_metrics.json   # Full stats
                aggregated_metrics.csv    # CSV table
                aggregated_metrics.md     # Publication-ready Markdown

Cross-combo summary::

    experiments/experiment_result/rep_compare/
        rep_compare_summary_mean_std.json
        rep_compare_summary_mean_std.csv
        rep_compare_summary_mean_std.md
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Ensure project root is on sys.path for src imports.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Configuration ──────────────────────────────────────────────────────

COMBO_NAMES = [
    "mel_gadf", "mel_gasf", "mel_mtf", "mel_rp",
    "stft_gadf", "stft_gasf", "stft_mtf", "stft_rp",
    "cwt_gadf", "cwt_gasf", "cwt_mtf", "cwt_rp",
]

DATASET_ROOT = "datasets/rep_compare_12k_de_split"
CONFIG_DIR = Path("experiments/rep_compare")
RESULT_ROOT = Path("experiments/experiment_result/rep_compare")

# Default seeds for independent trials
DEFAULT_TRIAL_SEEDS = [42, 123, 456, 789, 1024]

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

# ── Metric keys for cross-combo summary ───────────────────────────────
_CROSS_SUMMARY_KEYS = [
    "test_acc", "test_precision", "test_recall",
    "test_f1", "test_gmean", "test_kappa", "test_auc",
]

_CROSS_SUMMARY_DISPLAY = {
    "test_acc": "Acc",
    "test_precision": "Prec",
    "test_recall": "Rec",
    "test_f1": "F1",
    "test_gmean": "G-mean",
    "test_kappa": "κ",
    "test_auc": "AUC",
}


# ═══════════════════════════════════════════════════════════════════════
#  Config generation
# ═══════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════
#  Single trial runner
# ═══════════════════════════════════════════════════════════════════════

def run_one_trial(combo: str, seed: int) -> bool:
    """Run a single training trial for one combo with a given seed.

    Returns True if the subprocess exits successfully.
    """
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

    print(f"\n{'─'*60}")
    print(f"  Combo: {combo}  |  Seed: {seed}")
    print(f"  Dataset: {dataset_path}")
    print(f"{'─'*60}")

    result = subprocess.run(cmd)
    return result.returncode == 0


# ═══════════════════════════════════════════════════════════════════════
#  Trial directory resolution
# ═══════════════════════════════════════════════════════════════════════

def _get_trial_dirs(combo: str, seeds: List[int]) -> List[Path]:
    """Return the list of existing trial directories for a combo."""
    exp_name = f"rep_compare_{combo}"
    trial_dirs = []
    for seed in seeds:
        trial_dir = RESULT_ROOT / combo / exp_name / f"trial_seed{seed}"
        metrics_path = trial_dir / "results" / "test_metrics.json"
        if metrics_path.exists():
            trial_dirs.append(trial_dir)
    return trial_dirs


def _get_agg_dir(combo: str) -> Path:
    """Return the aggregation output directory for a combo."""
    return RESULT_ROOT / combo / f"rep_compare_{combo}" / "aggregated"


# ═══════════════════════════════════════════════════════════════════════
#  Repeated trials runner
# ═══════════════════════════════════════════════════════════════════════

def run_repeated_trials(
    combo: str,
    seeds: List[int],
    continue_on_error: bool = False,
) -> int:
    """Run multiple independent trials for one combo, then aggregate.

    Args:
        combo: Combination name (e.g., 'mel_gadf').
        seeds: List of random seeds.
        continue_on_error: If True, continue remaining trials after a failure.

    Returns:
        Number of successfully completed trials.
    """
    dataset_path = Path(DATASET_ROOT) / combo
    if not dataset_path.exists():
        print(f"[SKIP] Dataset not found: {dataset_path}")
        return 0

    print(f"\n{'='*60}")
    print(f"  Combo: {combo}")
    print(f"  Trials: {len(seeds)} seeds → {seeds}")
    print(f"{'='*60}")

    success_count = 0
    for i, seed in enumerate(seeds, start=1):
        label = f"Trial {i}/{len(seeds)}"
        ok = run_one_trial(combo, seed)
        if ok:
            success_count += 1
            print(f"  [{label}] ✓ (seed={seed})")
        else:
            print(f"  [{label}] ✗ (seed={seed}) — FAILED")
            if not continue_on_error:
                print(f"[ERROR] Stopping after failure. {success_count} trial(s) succeeded.")
                return success_count

    print(f"\n  Completed: {success_count}/{len(seeds)} trials successful")
    return success_count


def aggregate_combo(combo: str, seeds: List[int]) -> Optional[Dict]:
    """Aggregate trial results for a single combo.

    Returns the aggregated dict, or None if insufficient trials.
    """
    trial_dirs = _get_trial_dirs(combo, seeds)
    if len(trial_dirs) < 2:
        print(f"  [WARN] {combo}: only {len(trial_dirs)} trial(s) with results. "
              "Need ≥2 for aggregation.")
        return None

    from src.utils.aggregation import aggregate_and_save
    agg_dir = _get_agg_dir(combo)
    title = f"rep_compare_{combo} — Repeated Trials"
    aggregated = aggregate_and_save(trial_dirs, agg_dir, title=title)
    print(f"  Aggregated → {agg_dir}")
    return aggregated


# ═══════════════════════════════════════════════════════════════════════
#  Cross-combo summary
# ═══════════════════════════════════════════════════════════════════════

def build_cross_combo_summary(seeds: List[int]):
    """Aggregate across all 12 combos and produce a consolidated summary.

    Reads per-combo aggregated_metrics.json files and produces a
    publication-ready comparison table.
    """
    from src.utils.aggregation import format_mean_std

    print(f"\n{'='*60}")
    print(f"  Cross-Combo Summary ({len(seeds)} trials each)")
    print(f"{'='*60}")

    rows: List[Dict] = []
    for combo in COMBO_NAMES:
        agg_path = _get_agg_dir(combo) / "aggregated_metrics.json"
        if not agg_path.exists():
            # Try aggregating on-the-fly
            print(f"  [WARN] No aggregated results for {combo}; attempting on-the-fly aggregation...")
            aggregated = aggregate_combo(combo, seeds)
            if aggregated is None:
                rows.append({"combo": combo, "error": "insufficient trials"})
                continue
        else:
            with open(agg_path, "r") as f:
                aggregated = json.load(f)

        row = {"combo": combo}
        for key in _CROSS_SUMMARY_KEYS:
            if key in aggregated:
                info = aggregated[key]
                row[f"{key}_mean"] = info["mean"]
                row[f"{key}_std"] = info["std"]
                row[f"{key}_str"] = format_mean_std(info["mean"], info["std"], key,
                                                     decimal_places=2, as_percent=True)
            else:
                row[f"{key}_mean"] = float("nan")
                row[f"{key}_std"] = float("nan")
                row[f"{key}_str"] = "N/A"
        rows.append(row)

    # ── Save JSON ──
    json_path = RESULT_ROOT / "rep_compare_summary_mean_std.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"  JSON: {json_path}")

    # ── Save CSV ──
    csv_path = RESULT_ROOT / "rep_compare_summary_mean_std.csv"
    import csv
    fieldnames = ["combo"] + [f"{k}_{s}" for k in _CROSS_SUMMARY_KEYS
                               for s in ("mean", "std")]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  CSV:  {csv_path}")

    # ── Save Markdown ──
    md_path = RESULT_ROOT / "rep_compare_summary_mean_std.md"
    lines = [
        "# Representation Comparison — Repeated Trials Summary",
        "",
        f"**Trials**: {len(seeds)} independent runs per combination.",
        f"**Seeds**: {seeds}",
        "",
        "| Combo | " + " | ".join(_CROSS_SUMMARY_DISPLAY[k] for k in _CROSS_SUMMARY_KEYS) + " |",
        "|-------|" + "|".join(["-------:" for _ in _CROSS_SUMMARY_KEYS]) + "|",
    ]
    for row in rows:
        cells = [row["combo"]] + [row.get(f"{k}_str", "N/A") for k in _CROSS_SUMMARY_KEYS]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  MD:   {md_path}")

    # ── Console table ──
    print(f"\n{'Combo':<20} " + " ".join(f"{_CROSS_SUMMARY_DISPLAY[k]:>10}" for k in _CROSS_SUMMARY_KEYS))
    print("-" * (20 + 11 * len(_CROSS_SUMMARY_KEYS)))
    for row in rows:
        vals = " ".join(f"{row.get(f'{k}_str', 'N/A'):>10}" for k in _CROSS_SUMMARY_KEYS)
        print(f"{row['combo']:<20} {vals}")

    return rows


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Representation Comparison — Batch Runner with Repeated Trials",
    )
    parser.add_argument("--gen-only", action="store_true",
                        help="Only generate config files, don't run")
    parser.add_argument("--combo", default=None,
                        help="Run a single combination (e.g., mel_gadf)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for single-trial mode")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Number of DataLoader workers (0 = use default config)")
    parser.add_argument(
        "--num-trials", type=int, default=1,
        help="Number of independent trials (default: 1). Set to 3 for publication-ready mean±std.",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        help="Explicit list of seeds (overrides --num-trials)",
    )
    parser.add_argument(
        "--continue-on-error", action="store_true",
        help="Continue remaining trials when one fails",
    )
    parser.add_argument(
        "--aggregate-only", action="store_true",
        help="Skip training; only aggregate existing trial results",
    )
    parser.add_argument(
        "--skip-cross-summary", action="store_true",
        help="Skip the cross-combo summary (useful when running a single combo)",
    )
    args = parser.parse_args()

    # ── Resolve seeds ──
    if args.seeds:
        seeds = args.seeds
    else:
        n = max(args.num_trials, 1)
        if n <= len(DEFAULT_TRIAL_SEEDS):
            seeds = DEFAULT_TRIAL_SEEDS[:n]
        else:
            extra = [DEFAULT_TRIAL_SEEDS[-1] + i * 500 + 1
                     for i in range(1, n - len(DEFAULT_TRIAL_SEEDS) + 1)]
            seeds = DEFAULT_TRIAL_SEEDS + extra

    is_repeated = len(seeds) > 1

    # ── Generate configs ──
    print("Generating experiment configs...")
    generate_configs(num_workers=args.num_workers)

    if args.gen_only:
        print(f"\nConfigs generated. Run with: python scripts/run_rep_compare.py "
              f"--num-workers {args.num_workers}" +
              (f" --num-trials {len(seeds)}" if is_repeated else ""))
        return

    # ── Determine combos to run ──
    combos_to_run = [args.combo] if args.combo else COMBO_NAMES

    # ── Run trials ──
    if not args.aggregate_only:
        print(f"\n{'='*60}")
        print(f"  Mode: {'Repeated trials' if is_repeated else 'Single trial'}")
        print(f"  Combos: {len(combos_to_run)}")
        if is_repeated:
            print(f"  Seeds: {seeds}")
        else:
            print(f"  Seed: {args.seed}")
        print(f"{'='*60}")

        for combo in combos_to_run:
            if is_repeated:
                success = run_repeated_trials(
                    combo, seeds=seeds,
                    continue_on_error=args.continue_on_error,
                )
                if success < 2:
                    print(f"  [WARN] {combo}: insufficient trials ({success}) for aggregation.")
            else:
                ok = run_one_trial(combo, args.seed)
                if not ok:
                    print(f"[FAIL] {combo} returned non-zero exit code")

    # ── Aggregate per-combo ──
    if is_repeated and len(combos_to_run) >= 1:
        print(f"\n{'='*60}")
        print(f"  Aggregating per-combo results...")
        print(f"{'='*60}")
        for combo in combos_to_run:
            aggregate_combo(combo, seeds)

        # ── Cross-combo summary ──
        if not args.skip_cross_summary and len(combos_to_run) > 1:
            build_cross_combo_summary(seeds)

    # ── Final report ──
    print(f"\n{'='*60}")
    if is_repeated and not args.skip_cross_summary and len(combos_to_run) > 1:
        print(f"  Done. Cross-combo summary:")
        print(f"    {RESULT_ROOT / 'rep_compare_summary_mean_std.md'}")
    print(f"  Per-combo aggregated results:")
    for combo in combos_to_run:
        agg_dir = _get_agg_dir(combo)
        if (agg_dir / "aggregated_metrics.md").exists():
            print(f"    {agg_dir / 'aggregated_metrics.md'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
