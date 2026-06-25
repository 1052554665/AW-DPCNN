#!/usr/bin/env python3
"""
Repeated Independent Trials Runner
====================================
Run N independent training trials with different random seeds and
aggregate results into **mean ± std** format for publication.

This directly addresses the reviewer comment:
  "Run 3 independent runs with different random seeds;
   report mean ± std for all metrics."

Usage::

    # 3 independent runs (default)
    python scripts/run_repeated_trials.py \\
        --config configs/default.yaml \\
        --exp-config experiments/exp1/vgg16.yaml

    # 5 independent runs with custom seeds
    python scripts/run_repeated_trials.py \
        --config configs/default.yaml \
        --exp-config experiments/exp1/MSCA_VGG16.yaml \
        --num-runs 5

    # Batch: run repeated trials for all configs in an exp directory
    python scripts/run_repeated_trials.py \
        --config configs/default.yaml \
        --exp-dir experiments/exp1 \
        --num-runs 3

    # Dry-run: print commands without executing
    python scripts/run_repeated_trials.py \\
        --config configs/default.yaml \\
        --exp-config experiments/exp1/vgg16.yaml \\
        --dry-run

Output structure::

    experiments/experiment_result/{exp_name}/
        aggregated/
            aggregated_metrics.json   # Full aggregated stats
            aggregated_metrics.csv    # CSV table
            aggregated_metrics.md     # Publication-ready Markdown table
        trial_seed42/
            resolved_config.yaml
            results/test_metrics.json
            ...
        trial_seed123/
        trial_seed456/
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

# Ensure project root is on sys.path for src imports.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Default seeds for independent trials ─────────────────────────────
# Chosen to be well-separated and reproducible.
DEFAULT_TRIAL_SEEDS = [42, 123, 456, 789, 1024]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run N independent training trials and aggregate results (mean ± std).",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Base config YAML path",
    )
    parser.add_argument(
        "--exp-config",
        default="",
        help="Single experiment override YAML path",
    )
    parser.add_argument(
        "--exp-dir",
        default="",
        help="Directory containing multiple experiment YAML files (batch mode)",
    )
    parser.add_argument(
        "--pattern",
        default="*.yaml",
        help="Glob pattern for experiment YAML files (batch mode)",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="Number of independent trials (default: 3)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Explicit list of seeds (overrides --num-runs)",
    )
    parser.add_argument(
        "--device",
        default="",
        help="Override device (e.g. cuda, cuda:0, cpu)",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining trials when one fails",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the commands without executing",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip training; only aggregate existing trial results",
    )
    return parser.parse_args()


def resolve_seeds(args: argparse.Namespace) -> List[int]:
    """Determine the list of seeds for independent trials."""
    if args.seeds:
        return args.seeds
    n = max(args.num_runs, 1)
    if n <= len(DEFAULT_TRIAL_SEEDS):
        return DEFAULT_TRIAL_SEEDS[:n]
    # Generate additional seeds if needed
    extra = [DEFAULT_TRIAL_SEEDS[-1] + i * 500 + 1 for i in range(1, n - len(DEFAULT_TRIAL_SEEDS) + 1)]
    return DEFAULT_TRIAL_SEEDS + extra


def resolve_exp_configs(args: argparse.Namespace) -> List[Path]:
    """Resolve the list of experiment configs to run."""
    if args.exp_config:
        p = Path(args.exp_config)
        if not p.exists():
            raise FileNotFoundError(f"Experiment config not found: {p}")
        return [p]

    if args.exp_dir:
        exp_path = Path(args.exp_dir)
        if not exp_path.exists():
            raise FileNotFoundError(f"Experiment directory not found: {exp_path}")
        configs = sorted([p for p in exp_path.glob(args.pattern) if p.is_file()])
        if not configs:
            raise FileNotFoundError(
                f"No experiment configs found in {exp_path} with pattern {args.pattern}"
            )
        return configs

    raise ValueError("Must specify --exp-config or --exp-dir.")


def run_single_trial(
    base_config: str,
    exp_config: Path,
    seed: int,
    device: str,
    trial_label: str,
    dry_run: bool = False,
) -> Optional[int]:
    """Run a single training trial with the given seed.

    The output run_dir is overridden via a temporary experiment override
    to place each trial in a subfolder named `trial_seed{seed}`.

    Returns exit code, or None if dry_run.
    """
    command = [
        sys.executable,
        "scripts/train.py",
        "--config", base_config,
        "--exp-config", str(exp_config),
        "--seed", str(seed),
    ]
    if device:
        command.extend(["--device", device])

    print(f"  [{trial_label}] Seed={seed}")
    print(f"  Command: {' '.join(command)}")
    if dry_run:
        return None

    result = subprocess.run(command, check=False)
    return result.returncode


def build_output_dir(base_config: str, exp_config: Path) -> Path:
    """Infer the aggregation output directory from the merged config."""
    from src.utils.config import load_config
    config = load_config(base_config, str(exp_config))
    root = config.get("output", {}).get("root_dir", "experiments/runs")
    return Path(root).resolve() / "aggregated"


def run_repeated_trials(
    base_config: str,
    exp_config: Path,
    seeds: List[int],
    device: str,
    continue_on_error: bool = False,
    dry_run: bool = False,
    aggregate_only: bool = False,
) -> int:
    """Run repeated trials for a single experiment config."""
    exp_name = exp_config.stem
    print(f"\n{'='*60}")
    print(f"Experiment: {exp_name}")
    print(f"Trials: {len(seeds)} seeds → {seeds}")
    print(f"{'='*60}")

    agg_dir = build_output_dir(base_config, exp_config)
    trial_dirs: List[Path] = []

    for i, seed in enumerate(seeds, start=1):
        label = f"Trial {i}/{len(seeds)}"
        if not aggregate_only:
            exit_code = run_single_trial(
                base_config, exp_config, seed, device, label, dry_run,
            )
            if exit_code is not None and exit_code != 0:
                msg = f"Trial {i} (seed={seed}) failed with exit code {exit_code}"
                if continue_on_error:
                    print(f"[WARN] {msg}; continuing")
                    continue
                print(f"[ERROR] {msg}")
                return exit_code

        # Construct the deterministic trial directory path
        trial_dir = _get_trial_dir(base_config, exp_config, seed)
        if trial_dir.exists() and (trial_dir / "results" / "test_metrics.json").exists():
            trial_dirs.append(trial_dir)
        elif not aggregate_only:
            print(f"  [WARN] Trial results not found at {trial_dir}")

    if dry_run:
        print("\n[Dry-run] Would aggregate results from trial directories.")
        return 0

    # ── Aggregate results ──────────────────────────────────────────
    if len(trial_dirs) < 2:
        print(f"[WARN] Only {len(trial_dirs)} trial(s) completed successfully. "
              "Need ≥2 for meaningful aggregation.")
        if len(trial_dirs) == 0:
            return 1

    _aggregate_trial_results(trial_dirs, agg_dir, title=f"{exp_name} — Repeated Trials")
    return 0


def _get_trial_dir(base_config: str, exp_config: Path, seed: int) -> Path:
    """Construct the deterministic trial directory path for a given seed.

    Uses the same config merging logic as train.py (base + exp override)
    to guarantee the path matches what prepare_run_dir created.
    """
    from src.utils.config import load_config
    config = load_config(base_config, str(exp_config))
    output_cfg = config.get("output", {})
    root = Path(output_cfg.get("root_dir", "experiments/runs")).resolve()
    exp_name = str(config.get("experiment_name", exp_config.stem))
    return root / exp_name / f"trial_seed{seed}"


def _aggregate_trial_results(trial_dirs: List[Path], agg_dir: Path, title: str) -> None:
    """Aggregate results from trial directories and save reports."""
    from src.utils.aggregation import aggregate_and_save

    print(f"\n{'─'*40}")
    print(f"Aggregating {len(trial_dirs)} trials → {agg_dir}")
    aggregated = aggregate_and_save(trial_dirs, agg_dir, title=title)

    # Print summary to console
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"  Trials: {len(trial_dirs)}")
    print(f"{'='*60}")
    for key in ["test_acc", "test_precision", "test_recall",
                 "test_f1", "test_gmean", "test_auc"]:
        if key not in aggregated:
            continue
        from src.utils.aggregation import METRIC_DISPLAY_NAMES, format_mean_std
        info = aggregated[key]
        display = METRIC_DISPLAY_NAMES.get(key, key)
        print(f"  {display:<20s}: {format_mean_std(info['mean'], info['std'], key)}")
    print(f"{'='*60}")
    print(f"Reports saved to: {agg_dir}")


def main() -> int:
    args = parse_args()
    seeds = resolve_seeds(args)
    exp_configs = resolve_exp_configs(args)

    print(f"Seeds: {seeds}")
    print(f"Configs to run: {len(exp_configs)}")

    overall_exit = 0
    for exp_config in exp_configs:
        exit_code = run_repeated_trials(
            base_config=args.config,
            exp_config=exp_config,
            seeds=seeds,
            device=args.device,
            continue_on_error=args.continue_on_error,
            dry_run=args.dry_run,
            aggregate_only=args.aggregate_only,
        )
        if exit_code != 0:
            overall_exit = exit_code
            if not args.continue_on_error:
                return overall_exit

    return overall_exit


if __name__ == "__main__":
    raise SystemExit(main())
