#!/usr/bin/env python3
"""
Experiment Results → LaTeX Tables Generator
=============================================
Scans ``experiments/experiment_result/``, collects test metrics and training
logs, and produces IEEE‑formatted LaTeX tables for the paper.

Output::
    paper/tab_backbone_comparison.tex
    paper/tab_training_summary.tex

Usage::

    python scripts/results_to_latex.py
    python scripts/results_to_latex.py --trial trial_seed42
"""

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────
RESULT_ROOT = Path("experiments/experiment_result")
DEFAULT_TRIAL = "trial_seed42"
OUTPUT_DIR = Path("paper")

# Display names for models
MODEL_NAMES = OrderedDict([
    ("msca-vgg16",         r"\textbf{MSCA-VGG16 (Ours)}"),
    ("vgg16",              "VGG16"),
    ("vit",                "ViT"),
    ("efficientnet-b0",    "EfficientNet-B0"),
    ("mobilenetv3_small",  "MobileNetV3-Small"),
    ("convnext-tiny",      "ConvNeXt-Tiny"),
])

# Metric display order & labels
METRICS = OrderedDict([
    ("test_acc",       "Acc"),
    ("test_precision", "Prec"),
    ("test_recall",    "Rec"),
    ("test_f1",        "F1"),
    ("test_gmean",     "G-Mean"),
    ("test_bal_acc",   "B-Acc"),
    ("test_kappa",     r"$\kappa$"),
    ("test_auc",       "AUC"),
])

PARAM_METRICS = OrderedDict([
    ("params", "Params (M)"),
    ("flops",  "FLOPs (M)"),
])


# ── Collect ────────────────────────────────────────────────────────────
def collect_all(trial: str = DEFAULT_TRIAL) -> dict:
    """Return {experiment_group: [{model, metrics, params, ...}]}."""
    all_data = OrderedDict()
    for exp_group in sorted(RESULT_ROOT.glob("*")):
        if not exp_group.is_dir():
            continue
        group_name = exp_group.name
        models_data = []
        for model_dir in sorted(exp_group.glob("*")):
            if not model_dir.is_dir():
                continue
            inner = sorted(model_dir.glob(f"{group_name}_*"))
            if not inner:
                continue
            metrics_path = inner[0] / trial / "results" / "test_metrics.json"
            if not metrics_path.exists():
                continue
            data = json.loads(metrics_path.read_text())
            model_name = data.get("model_name", model_dir.name)
            models_data.append({
                "model": model_name,
                "params": data.get("params", 0),
                "flops": data.get("flops", 0),
                "best_epoch": data.get("best_epoch", 0),
                "best_val_f1": data.get("best_val_f1", 0),
                "test_acc": data.get("test_acc", 0) * 100,
                "test_precision": data.get("test_precision", 0) * 100,
                "test_recall": data.get("test_recall", 0) * 100,
                "test_f1": data.get("test_f1", 0) * 100,
                "test_gmean": data.get("test_gmean", 0) * 100,
                "test_bal_acc": data.get("test_bal_acc", 0) * 100,
                "test_kappa": data.get("test_kappa", 0) * 100,
                "test_auc": data.get("test_auc", 0) * 100,
                "shuffled_acc": data.get("shuffled_acc", 0) * 100,
            })
        if models_data:
            all_data[group_name] = sorted(models_data, key=lambda x: -x["test_acc"])
    return all_data


# ── LaTeX helpers ──────────────────────────────────────────────────────
def _val(v, prec=1):
    """Format a float with *prec* decimals."""
    return f"{v:.{prec}f}"


def _bold_if_best(vals, idx, prec=1):
    """Return formatted value; bold if it's the maximum."""
    v = vals[idx]
    s = _val(v, prec)
    return rf"\mathbf{{{s}}}" if v == max(vals) else s


def _latex_table(caption, label, headers, rows, placement="[!htbp]"):
    """Build a complete LaTeX table environment string."""
    ncol = len(headers)
    lines = [
        rf"\begin{{table}}{placement}",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\small",
        r"\begin{tabular}{" + "l" + "c" * (ncol - 1) + "}",
        r"\toprule",
        " & ".join(rf"\textbf{{{h}}}" for h in headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(str(c) for c in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


# ── Backbone comparison table ──────────────────────────────────────────
def generate_backbone_table(data: dict, group: str = "exp1") -> str:
    if group not in data:
        print(f"[SKIP] No data for experiment group '{group}'")
        return ""
    models = data[group]
    metrics_keys = ["test_acc", "test_f1", "test_gmean", "test_auc", "test_kappa"]
    headers = ["Network"] + [METRICS[k] for k in metrics_keys] + ["Params (M)"]

    rows = []
    for m in models:
        name = MODEL_NAMES.get(m["model"], m["model"])
        vals = [m[k] for k in metrics_keys]
        # Find best per column to bold
        all_vals = {k: [md[k] for md in models] for k in metrics_keys}
        row = [name]
        for k in metrics_keys:
            best = max(all_vals[k])
            v = m[k]
            row.append(rf"\mathbf{{{_val(v)}}}" if v == best else _val(v))
        row.append(_val(m["params"] / 1e6, 1))
        rows.append(row)

    return _latex_table(
        f"Performance Comparison on the {group.upper()} Dataset (Seed 42)",
        f"tab:{group}_comparison",
        headers, rows,
    )


# ── Training summary table ─────────────────────────────────────────────
def generate_training_summary(data: dict, group: str = "exp1") -> str:
    if group not in data:
        return ""
    models = data[group]
    headers = ["Network", "Best Epoch", "Val F1", "Params (M)", "FLOPs (M)",
               "Shuffled Acc"]
    rows = []
    for m in models:
        name = MODEL_NAMES.get(m["model"], m["model"])
        rows.append([
            name,
            str(m["best_epoch"]),
            _val(m["best_val_f1"] * 100, 1),
            _val(m["params"] / 1e6, 1),
            _val(m["flops"] / 1e6, 0),
            _val(m["shuffled_acc"], 1),
        ])
    return _latex_table(
        "Training Summary and Leakage Check",
        f"tab:{group}_training",
        headers, rows,
    )


# ═══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Generate IEEE LaTeX tables from experiment results")
    parser.add_argument("--trial", default=DEFAULT_TRIAL,
                        help=f"Trial subdirectory name (default: {DEFAULT_TRIAL})")
    parser.add_argument("--group", default=None,
                        help="Limit to a specific experiment group (e.g., exp1, ablation)")
    args = parser.parse_args()

    data = collect_all(args.trial)
    if not data:
        print("[ERROR] No experiment results found.")
        sys.exit(1)

    print(f"Experiment groups found: {list(data.keys())}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for group in data:
        if args.group and group != args.group:
            continue
        # Backbone comparison
        tex = generate_backbone_table(data, group)
        if tex:
            path = OUTPUT_DIR / f"tab_{group}_comparison.tex"
            path.write_text(tex)
            print(f"  Saved {path}")

        # Training summary
        tex2 = generate_training_summary(data, group)
        if tex2:
            path2 = OUTPUT_DIR / f"tab_{group}_training.tex"
            path2.write_text(tex2)
            print(f"  Saved {path2}")

    print("\nDone. Include in paper with \\input{{tab_<group>_comparison}}")


if __name__ == "__main__":
    main()
