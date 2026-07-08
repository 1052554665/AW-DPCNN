#!/usr/bin/env python3
"""
Generalization Validation — Grouped Bar Chart
==============================================
Collects test metrics across all models for the 12k FE and 48k DE datasets
and produces a two-panel grouped bar chart.  Each panel shows four metrics
per model: Accuracy, F1-score, AUC, G-Mean.

Directory layout::

    experiments/experiment_result/exp1/
        {model}/
            {dataset}/           ← 12k_fe | 48k_de
                exp1_{model}/
                    trial_seed{xxx}/
                        results/
                            test_metrics.json

Usage::

    # Single trial
    python scripts/plot_generalization.py --trial trial_seed42

    # Other trials
    python scripts/plot_generalization.py --trial trial_seed123
    python scripts/plot_generalization.py --trial trial_seed456
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── IEEE-compatible font configuration ────────────────────────────────
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["font.size"] = 9
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["pdf.fonttype"] = 42  # embed fonts for IEEE

# ── Configuration ──────────────────────────────────────────────────────
RESULT_ROOT = Path("experiments/experiment_result/exp1")
METRICS_FILE = "results/test_metrics.json"

DATASETS = ["12k_fe", "48k_de"]
DATASET_LABELS = {
    "12k_fe": "CWRU 12k Fan-End (Cross-Sensor)",
    "48k_de": "CWRU 48k Drive-End (Cross-Sampling-Rate)",
}

MODEL_LABELS = {
    "MSCA_VGG16":        "MSCA-VGG16\n(Ours)",
    "vgg16":             "VGG16",
    "resnet18":          "ResNet18",
    "convnext_tiny":     "ConvNeXt\nTiny",
    "efficientnet_b0":   "Efficient\nNet-B0",
    "mobilenetv3_small": "MobileNet\nV3-Small",
    "vit":               "ViT",
}

# Sort order: Ours first, then best-to-worst by typical performance
MODEL_ORDER = [
    "MSCA_VGG16", "resnet18", "vgg16",
    "convnext_tiny", "efficientnet_b0", "mobilenetv3_small", "vit",
]

COLORS  = ["#4D4D4D", "#808080", "#B3B3B3", "#D9D9D9"]
HATCHES = ["", "//", "\\\\", "xx"]

OUTPUT_DIR = Path("paper/figures/generalization")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Collect ────────────────────────────────────────────────────────────
def collect_metrics(dataset: str, trial: str) -> list:
    """Walk the result tree and extract test metrics for every model.

    Parameters
    ----------
    dataset : str   One of ``DATASETS`` (e.g. ``"12k_fe"``).
    trial   : str   Trial seed directory name (e.g. ``"trial_seed42"``).

    Returns
    -------
    list of dicts with keys: label, acc, f1, auc, gmean, params.
    """
    rows = []
    for model_name in MODEL_ORDER:
        model_dir = RESULT_ROOT / model_name
        if not model_dir.is_dir():
            print(f"[SKIP] {model_name}: directory not found")
            continue

        dataset_dir = model_dir / dataset
        if not dataset_dir.is_dir():
            print(f"[SKIP] {model_name}: no dataset dir '{dataset}'")
            continue

        # Find the innermost experiment directory (e.g. exp1_msca_vgg16)
        inner = sorted(dataset_dir.glob("exp1_*"))
        if not inner:
            print(f"[SKIP] {model_name}/{dataset}: no exp1_* subdir")
            continue

        path = inner[0] / trial / METRICS_FILE
        if not path.exists():
            print(f"[SKIP] {path}")
            continue

        d = json.loads(path.read_text())
        rows.append({
            "label":  MODEL_LABELS.get(model_name, model_name),
            "acc":    d["test_acc"] * 100,
            "f1":     d["test_f1"] * 100,
            "auc":    d["test_auc"] * 100,
            "gmean":  d["test_gmean"] * 100,
            "params": d["params"] / 1e6,
        })

    return rows


# ── Plot ───────────────────────────────────────────────────────────────
def plot_panel(ax, rows: list, title: str):
    """Plot a single dataset panel onto the given axes."""
    n = len(rows)
    if n == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=14, color="gray")
        return

    params = [r["params"] for r in rows]
    metrics = [
        ("Acc",    [r["acc"]   for r in rows]),
        ("F1",     [r["f1"]    for r in rows]),
        ("AUC",    [r["auc"]   for r in rows]),
        ("G-Mean", [r["gmean"] for r in rows]),
    ]
    m = len(metrics)
    w = 0.18
    offsets = np.linspace(-w * (m - 1) / 2, w * (m - 1) / 2, m)

    for i, (name, vals) in enumerate(metrics):
        bars = ax.bar(
            np.arange(n) + offsets[i], vals, w,
            color=COLORS[i], edgecolor="black", linewidth=0.5,
            hatch=HATCHES[i], label=name, zorder=3,
        )
        # Mark the highest bar with an asterisk
        best_idx = np.argmax(vals)
        for j, (bar, val) in enumerate(zip(bars, vals)):
            txt = f"{val:.1f}"
            if j == best_idx:
                txt += "*"
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                txt, ha="center", va="bottom",
                fontsize=7, fontweight="bold", rotation=90,
            )

    # Parameter count labels
    for i, p in enumerate(params):
        ax.text(i, -5, f"{p:.1f}M", ha="center", fontsize=7.5,
                color="dimgray", fontweight="bold")

    ax.set_xticks(np.arange(n))
    ax.set_xticklabels([r["label"] for r in rows], fontsize=8.5)
    ax.set_ylabel("Score (%)", fontsize=10)
    ax.set_ylim(-10, 110)
    ax.grid(axis="y", alpha=0.2, color="gray", zorder=0)


def plot(rows_fe: list, rows_de: list, trial: str):
    """Generate a two-panel figure: 12k FE (left) and 48k DE (right)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.2))

    plot_panel(ax1, rows_fe, DATASET_LABELS["12k_fe"])
    plot_panel(ax2, rows_de, DATASET_LABELS["48k_de"])

    # Shared legend at the bottom
    handles, labels = ax1.get_legend_handles_labels()
    # Add asterisk mark legend entry
    from matplotlib.patches import Patch
    star_patch = Patch(facecolor="white", edgecolor="black", linewidth=0.5,
                        label="*  Best per metric")
    handles.append(star_patch)
    labels.append("*  Best per metric")

    fig.legend(handles, labels, loc="lower center", fontsize=9.5,
               framealpha=0.95, edgecolor="gray", ncol=5, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    fname = f"generalization_{trial}.pdf"
    out = OUTPUT_DIR / fname
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved  {out}")

    # Print summary table
    hdr = f"{'Model':<22s} {'Params':>7s} {'Acc%':>7s} {'F1%':>7s} {'AUC%':>7s} {'G-Mean%':>8s}"
    for ds_label, rows in [("12k FE", rows_fe), ("48k DE", rows_de)]:
        print(f"\n── {ds_label} ──\n{hdr}\n{'-'*len(hdr)}")
        for r in rows:
            label = r['label'].replace('\n', ' ')
            print(f"{label:<22s} {r['params']:6.1f}M "
                  f"{r['acc']:7.2f} {r['f1']:7.2f} {r['auc']:7.2f} {r['gmean']:8.2f}")


# ── CLI ─────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generalization validation grouped bar chart")
    p.add_argument("--trial", default="trial_seed42",
                   help="Trial seed directory name (default: trial_seed42)")
    p.add_argument("--output-dir", default=str(OUTPUT_DIR),
                   help="Output directory for figures")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    globals()["OUTPUT_DIR"] = output_dir

    # Collect for both datasets
    rows_fe = collect_metrics(dataset="12k_fe", trial=args.trial)
    rows_de = collect_metrics(dataset="48k_de", trial=args.trial)

    if not rows_fe and not rows_de:
        print(f"[ERROR] No test_metrics.json files found for trial={args.trial}")
        sys.exit(1)

    plot(rows_fe, rows_de, trial=args.trial)
