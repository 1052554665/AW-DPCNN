#!/usr/bin/env python3
"""
Noise Robustness Comparison — Line Plot
========================================
Visualise how classification accuracy (and optionally F1 / AUC) of different
models degrades as additive Gaussian noise increases (SNR decreases).

Reads data from ``experiments/experiment_result/noise_robustness/noise_robustness.csv``.
Output figures are saved to ``data-optimization/``.

Usage::

    python data-optimization/noise_robustness_comparison.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── IEEE-compatible font configuration ────────────────────────────────
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"

# ── Font sizes ────────────────────────────────────────────────────────
# Centralised font size control for all figures
FONT = {
    "title":        16,
    "axis_label":   15,
    "tick_label":   13,
    "tick_label_sm": 11.5,   # smaller tick labels for multi-panel
    "legend":       12,
    "annotation":   10,
    "subplot_title": 14,
}

# ── Paths ─────────────────────────────────────────────────────────────
# External CSV data file (project-relative path)
CSV_PATH = Path("experiments/experiment_result/noise_robustness/noise_robustness.csv")

# ── Configuration ──────────────────────────────────────────────────────
OUTPUT_DIR = Path("data-optimization/noise_robustness")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model display names (consistent with existing project style)
MODEL_LABELS: Dict[str, str] = {
    "msca-vgg16":        "MSCA-VGG16\n(Ours)",
    "convnext-tiny":     "ConvNeXt\nTiny",
    "efficientnet-b0":   "Efficient\nNet-B0",
    "mobilenetv3_small": "MobileNet\nV3-Small",
    "resnet18":          "ResNet18",
    "vgg16":             "VGG16",
    "vit":               "ViT",
}

# Model display order (Ours first, then alphabetically or by performance tier)
MODEL_ORDER: List[str] = [
    "msca-vgg16",
    "resnet18",
    "vgg16",
    "convnext-tiny",
    "efficientnet-b0",
    "mobilenetv3_small",
    "vit",
]

# Distinct colour palette (colourblind-friendly, high contrast)
COLORS: Dict[str, str] = {
    "msca-vgg16":        "#D62728",  # red — ours
    "convnext-tiny":     "#1F77B4",  # blue
    "efficientnet-b0":   "#FF7F0E",  # orange
    "mobilenetv3_small": "#9467BD",  # purple
    "resnet18":          "#2CA02C",  # green
    "vgg16":             "#8C564B",  # brown
    "vit":               "#17BECF",  # cyan
}

MARKERS: Dict[str, str] = {
    "msca-vgg16":        "s",
    "convnext-tiny":     "D",
    "efficientnet-b0":   "^",
    "mobilenetv3_small": "v",
    "resnet18":          "o",
    "vgg16":             "P",
    "vit":               "X",
}


# ── Data loading ───────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    """Load the external CSV and return a tidy DataFrame with a numeric
    ``snr_val`` column (``-1`` for clean)."""
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH.resolve()}")
    df = pd.read_csv(CSV_PATH)

    def _snr_to_float(snr_str: str) -> float:
        if snr_str.strip().lower() == "clean":
            return -1.0   # place "clean" to the left of 5 dB
        return float(snr_str.replace("dB", "").strip())

    df["snr_val"] = df["snr"].apply(_snr_to_float)
    df["display_name"] = df["model"].map(MODEL_LABELS).fillna(df["model"])
    return df


# ── Plotting ───────────────────────────────────────────────────────────
def plot_accuracy_vs_snr(df: pd.DataFrame):
    """Line plot: Accuracy (%) vs SNR (dB)."""
    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    # Sort models by clean accuracy so legend is ordered
    clean_acc = df[df["snr"] == "clean"].set_index("model")["accuracy"]
    model_order = sorted(
        df["model"].unique(),
        key=lambda m: clean_acc.get(m, 0),
        reverse=True,
    )

    for model in model_order:
        sub = df[df["model"] == model].sort_values("snr_val")
        x = sub["snr_val"].values
        y = sub["accuracy"].values

        label = MODEL_LABELS.get(model, model).replace("\n", " ")
        color = COLORS.get(model, "#333333")
        marker = MARKERS.get(model, "o")
        lw = 2.2 if "ours" in label.lower() else 1.5
        ls = "-" if "ours" in label.lower() else "--"

        ax.plot(x, y, marker=marker, color=color, linestyle=ls,
                linewidth=lw, markersize=7, markeredgewidth=0.8,
                markeredgecolor="black", label=label, zorder=3,
                markerfacecolor=color)

        # Annotate worst drop for each model (at 5 dB)
        worst = sub[sub["snr_val"] == 5.0]
        if not worst.empty:
            ax.annotate(
                f'{worst["accuracy"].values[0]:.1f}',
                (5.0, worst["accuracy"].values[0]),
                textcoords="offset points", xytext=(8, -2),
                fontsize=FONT["annotation"], color=color, fontweight="bold",
            )

    # Clean reference lines
    for model in model_order:
        sub = df[df["model"] == model]
        clean_row = sub[sub["snr"] == "clean"]
        if not clean_row.empty:
            ax.axhline(y=clean_row["accuracy"].values[0],
                       color=COLORS.get(model, "#333333"),
                       linestyle=":", linewidth=0.6, alpha=0.4, zorder=1)

    # Axis formatting
    # Replace -1 with "Clean" on x-axis
    xtick_vals = sorted(df["snr_val"].unique())
    xtick_labels = ["Clean" if v == -1 else f"{int(v)} dB" for v in xtick_vals]
    ax.set_xticks(xtick_vals)
    ax.set_xticklabels(xtick_labels, fontsize=FONT["tick_label"])
    ax.set_xlim(-2.5, 31.5)

    ax.set_ylabel("Accuracy (%)", fontsize=FONT["axis_label"])
    ax.set_xlabel("Noise Level (SNR)", fontsize=FONT["axis_label"])
    ax.set_ylim(-5, 108)
    ax.grid(axis="y", alpha=0.25, color="gray", zorder=0)
    ax.grid(axis="x", alpha=0.15, color="gray", zorder=0)

    # Legend outside the plot to avoid clutter
    ax.legend(loc="lower left", fontsize=FONT["legend"], framealpha=0.95,
              edgecolor="gray", ncol=2, columnspacing=0.8)

    # Title
    ax.set_title("Noise Robustness Comparison — Accuracy vs SNR",
                 fontsize=FONT["title"], fontweight="bold", pad=12)

    plt.tight_layout()
    out = OUTPUT_DIR / "noise_robustness_accuracy.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {out}  (PNG + PDF)")


def plot_accuracy_zoomed(df: pd.DataFrame):
    """Zoomed-in line plot for SNR ≥ 10 dB (where most models recover)."""
    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    clean_acc = df[df["snr"] == "clean"].set_index("model")["accuracy"]
    model_order = sorted(
        df["model"].unique(),
        key=lambda m: clean_acc.get(m, 0),
        reverse=True,
    )

    for model in model_order:
        sub = df[(df["model"] == model) & (df["snr_val"] >= 10)]
        sub = sub.sort_values("snr_val")
        # Add clean reference point
        clean_row = df[(df["model"] == model) & (df["snr"] == "clean")]
        if not clean_row.empty:
            clean_pt = clean_row.copy()
            clean_pt["snr_val"] = -1  # visual anchor
            sub = pd.concat([clean_pt, sub], ignore_index=True).sort_values("snr_val")

        x = sub["snr_val"].values
        y = sub["accuracy"].values

        label = MODEL_LABELS.get(model, model).replace("\n", " ")
        color = COLORS.get(model, "#333333")
        marker = MARKERS.get(model, "o")
        lw = 2.2 if "ours" in label.lower() else 1.5
        ls = "-" if "ours" in label.lower() else "--"

        ax.plot(x, y, marker=marker, color=color, linestyle=ls,
                linewidth=lw, markersize=7, markeredgewidth=0.8,
                markeredgecolor="black", label=label, zorder=3,
                markerfacecolor=color)

    xtick_vals = sorted([v for v in df["snr_val"].unique() if v >= 10] + [-1])
    xtick_labels = ["Clean" if v == -1 else f"{int(v)} dB" for v in xtick_vals]
    ax.set_xticks(xtick_vals)
    ax.set_xticklabels(xtick_labels, fontsize=FONT["tick_label"])
    ax.set_xlim(-2.5, 31.5)

    ax.set_ylabel("Accuracy (%)", fontsize=FONT["axis_label"])
    ax.set_xlabel("Noise Level (SNR)", fontsize=FONT["axis_label"])
    ax.set_ylim(80, 104)
    ax.grid(axis="y", alpha=0.25, color="gray", zorder=0)
    ax.grid(axis="x", alpha=0.15, color="gray", zorder=0)

    ax.legend(loc="lower left", fontsize=FONT["legend"], framealpha=0.95,
              edgecolor="gray", ncol=2, columnspacing=0.8)

    ax.set_title("Noise Robustness Comparison — Accuracy vs SNR (Zoomed)",
                 fontsize=FONT["title"], fontweight="bold", pad=12)

    plt.tight_layout()
    out = OUTPUT_DIR / "noise_robustness_accuracy_zoomed.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {out}  (PNG + PDF)")


def plot_multi_panel(df: pd.DataFrame):
    """Three-panel figure: Accuracy, F1, AUC vs SNR."""
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.2))

    metrics = [
        ("accuracy", "Accuracy (%)"),
        ("f1",       r"$F_1$-Score (%)"),
        ("auc",      "AUC (%)"),
    ]

    clean_acc = df[df["snr"] == "clean"].set_index("model")["accuracy"]
    model_order = sorted(
        df["model"].unique(),
        key=lambda m: clean_acc.get(m, 0),
        reverse=True,
    )

    for ax_idx, (col, ylabel) in enumerate(metrics):
        ax = axes[ax_idx]
        for model in model_order:
            sub = df[df["model"] == model].sort_values("snr_val")
            x = sub["snr_val"].values
            y = sub[col].values

            label = MODEL_LABELS.get(model, model).replace("\n", " ")
            color = COLORS.get(model, "#333333")
            marker = MARKERS.get(model, "o")
            lw = 2.2 if "ours" in label.lower() else 1.5
            ls = "-" if "ours" in label.lower() else "--"

            ax.plot(x, y, marker=marker, color=color, linestyle=ls,
                    linewidth=lw, markersize=5.5, markeredgewidth=0.6,
                    markeredgecolor="black", label=label, zorder=3,
                    markerfacecolor=color)

        xtick_vals = sorted(df["snr_val"].unique())
        xtick_labels = ["Clean" if v == -1 else f"{int(v)} dB"
                        for v in xtick_vals]
        ax.set_xticks(xtick_vals)
        ax.set_xticklabels(xtick_labels, fontsize=FONT["tick_label_sm"], rotation=30)
        ax.set_xlim(-2.5, 31.5)
        ax.set_ylabel(ylabel, fontsize=FONT["subplot_title"])
        ax.set_xlabel("Noise Level (SNR)", fontsize=FONT["axis_label"] - 2)
        ax.grid(axis="y", alpha=0.25, color="gray", zorder=0)
        ax.grid(axis="x", alpha=0.15, color="gray", zorder=0)

        # Subplot letter
        ax.set_title(f"({chr(97 + ax_idx)})  {ylabel}", fontsize=FONT["subplot_title"],
                     fontweight="bold", loc="left")

    # Shared legend below
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", fontsize=FONT["legend"],
               framealpha=0.95, edgecolor="gray", ncol=len(model_order),
               bbox_to_anchor=(0.5, -0.06))

    # fig.suptitle("Noise Robustness Comparison", fontsize=14,
    #              fontweight="bold", y=1.01)

    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    out = OUTPUT_DIR / "noise_robustness_multi_panel.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {out}  (PNG + PDF)")


def print_summary_table(df: pd.DataFrame):
    """Print a markdown summary table of accuracy at each SNR level."""
    snr_levels = ["clean", "5.0 dB", "10.0 dB", "15.0 dB",
                  "20.0 dB", "25.0 dB", "30.0 dB"]
    models = MODEL_ORDER

    print("\n" + "=" * 100)
    print("Accuracy (%) Summary — Noise Robustness")
    print("=" * 100)

    # Header
    header = f"{'Model':<22s}"
    for snr in snr_levels:
        header += f" {snr:>8s}"
    print(header)
    print("-" * len(header))

    for model in models:
        row_str = f"{MODEL_LABELS.get(model, model).replace(chr(10), ' '):<22s}"
        for snr in snr_levels:
            val = df[(df["model"] == model) & (df["snr"] == snr)]
            if not val.empty:
                row_str += f" {val['accuracy'].values[0]:7.2f}"
            else:
                row_str += f" {'—':>8s}"
        print(row_str)

    print("-" * len(header))

    # Drop from clean to 5 dB
    print("\nAccuracy drop (Clean → 5 dB):")
    for model in models:
        clean = df[(df["model"] == model) & (df["snr"] == "clean")]
        noisy = df[(df["model"] == model) & (df["snr"] == "5.0 dB")]
        if not clean.empty and not noisy.empty:
            drop = clean["accuracy"].values[0] - noisy["accuracy"].values[0]
            name = MODEL_LABELS.get(model, model).replace("\n", " ")
            print(f"  {name:<22s}  Δ = {drop:.2f} pp")


# ── Main ───────────────────────────────────────────────────────────────
def main():
    df = load_data()

    # 1. Accuracy vs SNR (full range)
    plot_accuracy_vs_snr(df)

    # 2. Accuracy vs SNR (zoomed, ≥ 10 dB)
    plot_accuracy_zoomed(df)

    # 3. Three-panel: Accuracy, F1, AUC
    plot_multi_panel(df)

    # 4. Summary table
    print_summary_table(df)

    print(f"\n[Done] All figures saved to {OUTPUT_DIR.resolve()}/")


if __name__ == "__main__":
    main()
