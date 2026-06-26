"""
Aggregation utilities for repeated-trial experiments.

Computes mean ± std across multiple independent runs and produces
publication-ready summary tables in CSV, JSON, and Markdown formats.
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Metric keys to aggregate ─────────────────────────────────────────
_TEST_METRIC_KEYS = [
    "test_loss",
    "test_acc",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_gmean",
    "test_bal_acc",
    "test_kappa",
    "test_auc",
    "best_val_f1",
    "best_epoch",
]

# Human-readable names for publication tables
METRIC_DISPLAY_NAMES = {
    "test_loss": "Loss",
    "test_acc": "Accuracy",
    "test_precision": "Precision",
    "test_recall": "Recall",
    "test_f1": "F1-score",
    "test_gmean": "G-Mean",
    "test_bal_acc": "Balanced Acc.",
    "test_kappa": "Cohen's κ",
    "test_auc": "ROC-AUC",
    "best_val_f1": "Best Val F1",
    "best_epoch": "Best Epoch",
}

# Keys to format as percentage (multiply × 100)
_PERCENT_KEYS = {
    "test_acc", "test_precision", "test_recall", "test_f1",
    "test_gmean", "test_bal_acc", "test_auc", "test_kappa", "best_val_f1",
}

# Keys to format as integer
_INT_KEYS = {"best_epoch"}


def _load_trial_results(trial_dirs: List[Path]) -> List[Dict[str, float]]:
    """Load `test_metrics.json` from each trial directory."""
    results = []
    for trial_dir in trial_dirs:
        metrics_path = trial_dir / "results" / "test_metrics.json"
        if not metrics_path.exists():
            print(f"[WARN] Missing metrics file: {metrics_path}")
            continue
        with open(metrics_path, "r") as f:
            results.append(json.load(f))
    return results


def aggregate_trials(trial_dirs: List[Path]) -> Dict[str, Dict[str, float]]:
    """Aggregate metrics across trial directories.

    Args:
        trial_dirs: List of paths to individual trial run directories.

    Returns:
        Dict mapping metric_name → {mean, std, min, max, trials, values}.
    """
    trial_results = _load_trial_results(trial_dirs)
    n_trials = len(trial_results)
    if n_trials == 0:
        raise ValueError("No trial results found.")

    aggregated = {}
    for key in _TEST_METRIC_KEYS:
        values = []
        for trial in trial_results:
            if key in trial:
                values.append(float(trial[key]))
        if not values:
            continue
        arr = np.array(values)
        aggregated[key] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)),  # sample std
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "trials": n_trials,
            "values": [round(v, 6) for v in values],
        }

    # Also aggregate model complexity (constant across trials — pick first)
    for const_key in ("params", "flops", "model_name"):
        if const_key in trial_results[0]:
            aggregated[const_key] = {"value": trial_results[0][const_key]}

    return aggregated


def format_mean_std(mean: float, std: float, key: str,
                    decimal_places: int = 2, as_percent: bool = True) -> str:
    """Format a metric as 'mean ± std' string with appropriate scaling.

    Args:
        mean: Mean value.
        std: Standard deviation.
        key: Metric key (determines scaling).
        decimal_places: Number of decimal places.
        as_percent: If True and key is a percentage metric, scale ×100.

    Returns:
        Formatted string like "98.52 ± 0.34" or "98.52±0.34".
    """
    scale = 100.0 if (as_percent and key in _PERCENT_KEYS) else 1.0
    fmt = f"{{:.{decimal_places}f}}"
    return f"{fmt.format(mean * scale)} ± {fmt.format(std * scale)}"


def save_aggregated_json(aggregated: Dict, output_path: Path) -> None:
    """Save aggregated metrics as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(aggregated, f, indent=2, ensure_ascii=False)


def save_aggregated_csv(aggregated: Dict, output_path: Path) -> None:
    """Save aggregated metrics as a CSV table.

    Columns: Metric, Mean, Std, Min, Max, Trials
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for key in _TEST_METRIC_KEYS:
        if key not in aggregated:
            continue
        info = aggregated[key]
        display = METRIC_DISPLAY_NAMES.get(key, key)
        is_pct = key in _PERCENT_KEYS
        scale = 100.0 if is_pct else 1.0
        rows.append({
            "Metric": display,
            "Mean": round(info["mean"] * scale, 4),
            "Std": round(info["std"] * scale, 4),
            "Min": round(info["min"] * scale, 4),
            "Max": round(info["max"] * scale, 4),
            "Trials": info["trials"],
        })

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Metric", "Mean", "Std", "Min", "Max", "Trials"])
        writer.writeheader()
        writer.writerows(rows)


def save_aggregated_markdown(aggregated: Dict, output_path: Path,
                              title: str = "Repeated Trials Results") -> None:
    """Save aggregated metrics as a publication-style Markdown table.

    Format: | Metric | Mean ± Std | Min | Max |
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_trials = next(iter(aggregated.values()))["trials"]

    lines = [
        f"# {title}",
        "",
        f"**Trials**: {n_trials} independent runs with different random seeds.",
        "",
        "| Metric | Mean ± Std | Min | Max |",
        "|--------|-----------|-----|-----|",
    ]

    for key in _TEST_METRIC_KEYS:
        if key not in aggregated:
            continue
        info = aggregated[key]
        display = METRIC_DISPLAY_NAMES.get(key, key)
        is_pct = key in _PERCENT_KEYS
        scale = 100.0 if is_pct else 1.0
        mean_str = format_mean_std(info["mean"], info["std"], key,
                                    decimal_places=2, as_percent=True)
        min_val = info["min"] * scale
        max_val = info["max"] * scale

        if key in _INT_KEYS:
            lines.append(f"| {display} | {int(info['mean'])} ± {int(info['std'])} "
                         f"| {int(min_val)} | {int(max_val)} |")
        else:
            lines.append(f"| {display} | {mean_str} "
                         f"| {min_val:.2f} | {max_val:.2f} |")

    # Model complexity (constant)
    if "params" in aggregated:
        params_m = aggregated["params"]["value"] / 1e6
        lines.append(f"\n**Parameters**: {params_m:.2f} M")
    if "flops" in aggregated:
        flops_m = aggregated["flops"]["value"] / 1e6
        lines.append(f"**FLOPs**: {flops_m:.2f} M")

    lines.append("")  # trailing newline
    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def aggregate_and_save(trial_dirs: List[Path], output_dir: Path,
                        title: str = "Repeated Trials Results") -> Dict[str, Dict[str, float]]:
    """Full pipeline: aggregate trials and save all formats.

    Args:
        trial_dirs: List of trial run directories.
        output_dir: Where to save aggregated_*.{json,csv,md}.
        title: Title for the Markdown report.

    Returns:
        Aggregated metrics dict.
    """
    aggregated = aggregate_trials(trial_dirs)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_aggregated_json(aggregated, output_dir / "aggregated_metrics.json")
    save_aggregated_csv(aggregated, output_dir / "aggregated_metrics.csv")
    save_aggregated_markdown(aggregated, output_dir / "aggregated_metrics.md", title=title)

    return aggregated
