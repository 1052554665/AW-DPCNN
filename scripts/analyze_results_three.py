#!/usr/bin/env python3
"""
Three‑Trial Experiment Result Analyzer — IEEE LaTeX Summary Table
===================================================================
Aggregates test metrics across 3 independent trials (seeds 42, 123, 456)
for all backbone models and all 3 CWRU datasets, then produces a
publication‑ready LaTeX summary table matching the IEEE reference format.

Usage::

    python scripts/analyze_results_three.py
    python scripts/analyze_results_three.py --output-dir paper/auto_tables
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.dataset_registry import DATASET_KEYS  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════

RESULT_ROOT = Path("experiments/experiment_result/exp1")
TRIAL_SEEDS = [42, 123, 456]

# Dataset keys -> display names
DATASET_DISPLAY = {
    "12k_de": "CWRU 12k DE",
    "12k_fe": "CWRU 12k FE",
    "48k_de": "CWRU 48k DE",
}

# Model directory -> display name (proposed first)
MODEL_ORDER = [
    ("MSCA_VGG16",      "MSCA-VGG16"),
    ("convnext_tiny",   "ConvNeXt-Tiny"),
    ("efficientnet_b0", "EfficientNet-B0"),
    ("mobilenetv3_small", "MobileNetV3-S"),
    ("resnet18",        "ResNet18"),
    ("vgg16",           "VGG16"),
    ("vit",             "ViT"),
]

# Metrics to include as sub‑columns (FPR is derived from bal_acc + recall)
TABLE_METRICS = [
    ("test_acc", "Acc"),
    ("test_f1",  "F1"),
    ("test_fpr", "FPR"),
]

# Raw metrics needed to compute TABLE_METRICS
_RAW_METRICS = {"test_acc", "test_f1", "test_bal_acc", "test_recall"}

# Keys that are percentages (multiply x 100)
_PERCENT_KEYS = {"test_acc", "test_f1", "test_fpr"}


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════

def _load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _load_trial_metrics(model_dir: str, dataset_key: str,
                        seed: int) -> Optional[Dict[str, float]]:
    """Load test_metrics.json for a specific model/dataset/seed."""
    exp_dir = RESULT_ROOT / model_dir / dataset_key
    inner = sorted(exp_dir.glob("exp1_*"))
    if not inner:
        return None
    path = inner[0] / f"trial_seed{seed}" / "results" / "test_metrics.json"
    data = _load_json(path)
    if data is None:
        return None
    return {k: data[k] for k in _RAW_METRICS if k in data}


def _format_mean_std(values: list, key: str) -> str:
    """Format mean +- std scaled as percentage."""
    arr = np.array(values, dtype=float)
    mean = arr.mean()
    std = arr.std(ddof=1) if len(arr) >= 2 else 0.0
    scale = 100.0 if key in _PERCENT_KEYS else 1.0
    return f"{mean * scale:.2f} \\pm {std * scale:.2f}"


def _bold_cell(value_str: str, is_best: bool) -> str:
    """Wrap in \\textbf if best, with proper LaTeX math mode."""
    if is_best:
        parts = value_str.split(" \\pm ")
        if len(parts) == 2:
            return f"$\\mathbf{{{parts[0]}}} \\pm {parts[1]}$"
    return f"${value_str}$"


# ═══════════════════════════════════════════════════════════════════════
#  Data collection
# ═══════════════════════════════════════════════════════════════════════

def build_all_data() -> Dict[str, List[Dict]]:
    """Collect aggregated metrics for all datasets x models.

    Returns: {dataset_key: [{model_info, metrics: {metric_key: mean_std_str}}]}
    """
    all_data = {}
    for dk in DATASET_KEYS:
        model_results = []
        for model_dir, display_name in MODEL_ORDER:
            # Load raw metrics per trial
            raw_trials = []
            for seed in TRIAL_SEEDS:
                m = _load_trial_metrics(model_dir, dk, seed)
                raw_trials.append(m)

            # Aggregate direct metrics (Acc, F1)
            metrics = {}
            for metric_key, _ in TABLE_METRICS:
                if metric_key == "test_fpr":
                    continue  # derived below
                trial_vals = []
                for m in raw_trials:
                    if m is None or metric_key not in m:
                        trial_vals = []
                        break
                    trial_vals.append(m[metric_key])
                if len(trial_vals) == len(TRIAL_SEEDS):
                    metrics[metric_key] = _format_mean_std(trial_vals, metric_key)
                else:
                    metrics[metric_key] = None

            # Derive FPR from bal_acc and recall: FPR = 1 - (2*bal_acc - recall)
            if all(m and "test_bal_acc" in m and "test_recall" in m
                   for m in raw_trials):
                fpr_vals = []
                for m in raw_trials:
                    specificity = 2.0 * m["test_bal_acc"] - m["test_recall"]
                    fpr = max(0.0, 1.0 - specificity)
                    fpr_vals.append(fpr)
                metrics["test_fpr"] = _format_mean_std(fpr_vals, "test_fpr")
            else:
                metrics["test_fpr"] = None

            model_results.append({
                "model_dir": model_dir,
                "display_name": display_name,
                "metrics": metrics,
            })
        all_data[dk] = model_results
    return all_data


# ═══════════════════════════════════════════════════════════════════════
#  LaTeX table generation
# ═══════════════════════════════════════════════════════════════════════

def generate_latex_table(all_data: Dict[str, List[Dict]],
                         output_dir: Path) -> None:
    """Generate IEEE‑style table (models=rows, datasets=column groups)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    n_datasets = len(DATASET_KEYS)
    n_subcols = len(TABLE_METRICS)   # 2 (Acc, F1)
    n_data_cols = n_datasets * n_subcols  # 6

    lines = []
    lines.append("% Auto-generated by scripts/analyze_results_three.py")
    lines.append(f"% Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    lines.append("\\begin{table*}[!htbp]")
    lines.append("    \\centering")
    cap = (
        "Performance Comparison Across Three CWRU Datasets "
        f"and Seven Backbone Networks "
        f"(Mean $\\pm$ Std over {len(TRIAL_SEEDS)} Independent Trials)."
    )
    lines.append(f"    \\caption{{{cap}}}")
    lines.append("    \\label{tab:three_trial_summary}")
    lines.append("    \\renewcommand{\\arraystretch}{1.1}")
    lines.append("    \\footnotesize")
    lines.append("    \\setlength{\\tabcolsep}{3.5pt}")

    # Column spec: Model name (l) + datasets x subcols (c)
    col_spec = "l" + "c" * n_data_cols
    lines.append(f"    \\begin{{tabular}}{{{col_spec}}}")
    lines.append("        \\toprule")

    # ── Header row 1: Model + dataset groups ──
    header1 = ["\\multirow{2}{*}{\\textbf{Model}}"]
    for dk in DATASET_KEYS:
        display = DATASET_DISPLAY.get(dk, dk)
        header1.append(f"\\multicolumn{{{n_subcols}}}{{c}}{{{display}}}")
    lines.append("        " + " & ".join(header1) + " \\\\")

    # ── Header row 2: sub‑metric labels ──
    header2 = [""]
    for _ in DATASET_KEYS:
        for _, sub_label in TABLE_METRICS:
            header2.append(sub_label)
    lines.append("        " + " & ".join(header2) + " \\\\")
    lines.append("        \\midrule")

    # ── Determine best per column (per dataset × metric) for bolding ──
    best_vals = {}  # (dataset_key, metric_key) -> best_mean
    for dk in DATASET_KEYS:
        for metric_key, _ in TABLE_METRICS:
            best_val = -1.0
            for mi, (model_dir, _) in enumerate(MODEL_ORDER):
                s = all_data[dk][mi]["metrics"].get(metric_key)
                if s is None:
                    continue
                try:
                    mean_val = float(s.split(" \\pm ")[0])
                except (ValueError, IndexError):
                    continue
                if mean_val > best_val:
                    best_val = mean_val
            best_vals[(dk, metric_key)] = best_val

    # ── Data rows: one per model ──
    for mi, (model_dir, display_name) in enumerate(MODEL_ORDER):
        is_ours = mi == 0  # first model is proposed
        name_cell = f"\\textbf{{{display_name}}}" if is_ours else display_name
        row_cells = [name_cell]

        for dk in DATASET_KEYS:
            md = all_data[dk][mi]
            for metric_key, _ in TABLE_METRICS:
                s = md["metrics"].get(metric_key)
                if s is None:
                    row_cells.append("---")
                else:
                    # Bold if this model achieves the best value for this dataset×metric
                    try:
                        mean_val = float(s.split(" \\pm ")[0])
                    except (ValueError, IndexError):
                        mean_val = -1.0
                    is_best = abs(mean_val - best_vals.get((dk, metric_key), -1.0)) < 1e-6
                    row_cells.append(_bold_cell(s, is_best))
        lines.append("        " + " & ".join(row_cells) + " \\\\")

    lines.append("        \\bottomrule")
    lines.append("    \\end{tabular}")
    lines.append("\\end{table*}")

    out_path = output_dir / "three_trial_summary.tex"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"\n  LaTeX table: {out_path}")


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Three‑Trial Experiment Analyzer — IEEE LaTeX Summary Table")
    parser.add_argument("--output-dir", default="paper/auto_tables",
                        help="Output directory for LaTeX files")
    args = parser.parse_args()

    print(f"Trials:   {TRIAL_SEEDS}")
    print(f"Models:   {len(MODEL_ORDER)}")
    print(f"Datasets: {', '.join(DATASET_KEYS)}")
    print()

    print("Collecting metrics...")
    all_data = build_all_data()

    # Quick console summary
    hdr = f"  {'Dataset':<12s}  {'Model':<20s}  {'Acc':>14s}  {'F1':>14s}"
    sep = f"  {'-'*12}  {'-'*20}  {'-'*14}  {'-'*14}"
    print(hdr)
    print(sep)
    for dk in DATASET_KEYS:
        for mi, md in enumerate(all_data[dk]):
            acc = md["metrics"].get("test_acc", "---")
            f1  = md["metrics"].get("test_f1", "---")
            label = DATASET_DISPLAY.get(dk, dk) if mi == 0 else ""
            print(f"  {label:<12s}  {md['display_name']:<20s}  {acc:>14s}  {f1:>14s}")
        print()

    output_dir = Path(args.output_dir)
    generate_latex_table(all_data, output_dir)

    print(f"Done. Table saved to {output_dir / 'three_trial_summary.tex'}")


if __name__ == "__main__":
    main()
