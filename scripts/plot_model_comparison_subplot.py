#!/usr/bin/env python3
"""
Model Comparison — Multi-Metric Panel Chart
============================================
Collects test metrics from each model's test_metrics.json (trial seed 42)
and produces a 4‑panel publication‑quality figure: Params, AUC, Acc, G‑Mean.

Usage::

python scripts/plot_model_comparison.py
"""

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
plt.rcParams["mathtext.fontset"] = "stix"

# ── Configuration ──────────────────────────────────────────────────────
RESULT_ROOT = Path("experiments/experiment_result/exp1")
TRIAL = "trial_seed42"
METRICS_FILE = "results/test_metrics.json"

# Display names (model_name → pretty label)
MODEL_LABELS = {
"convnext-tiny": "ConvNeXt-\nTiny",
"efficientnet-b0": "Efficient\nNet-B0",
"mobilenetv3_small": "MobileNet\nV3-Small",
"msca-vgg16": "MSCA-VGG16\n(Ours)",
"vgg16": "VGG16",
"vit": "ViT",
}

# Colours — publication-friendly grayscale; MSCA-VGG16 gets a bronze accent
COLORS = [
"#B0B0B0", # ConvNeXt-Tiny — light gray
"#909090", # EfficientNet-B0 — medium gray
"#707070", # MobileNetV3-Small — darker gray
"#D4A574", # MSCA-VGG16 (Ours) — warm bronze accent
"#505050", # VGG16 — dark gray
"#303030", # ViT — darkest gray
]

# Hatch patterns for contrast
HATCH = ["", "//", "\\\\", "xx", "--", ".."]

OUTPUT_DIR = Path("paper/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Collect metrics ────────────────────────────────────────────────────
def collect_metrics() -> list:
    rows = []
    for exp_dir in sorted(RESULT_ROOT.glob("*")):
        if not exp_dir.is_dir():
            continue
        inner_dirs = sorted(exp_dir.glob("exp1_*"))
        if not inner_dirs:
            continue
        metrics_path = inner_dirs[0] / TRIAL / METRICS_FILE
        if not metrics_path.exists():
            print(f"[SKIP] {metrics_path} — not found")
            continue
        with open(metrics_path) as f:
            data = json.load(f)

        model_name = data.get("model_name", exp_dir.name)
        label = MODEL_LABELS.get(model_name, model_name)
        rows.append({
            "label": label,
            "model_name": model_name,
            "test_acc": data["test_acc"] * 100,
            "test_f1": data["test_f1"] * 100,
            "test_auc": data["test_auc"] * 100,
            "test_gmean": data["test_gmean"] * 100,
            "params_m": data["params"] / 1e6,
            "chance_level": data.get("chance_level", 0.1) * 100,
            "shuffled_acc": data.get("shuffled_acc", 0.0) * 100,
        })
    # Sort by AUC descending for consistent order
    return sorted(rows, key=lambda r: r["test_auc"], reverse=True)

# ── Plot ───────────────────────────────────────────────────────────────
def plot_comparison(rows: list):
    n = len(rows)
    labels = [r["label"] for r in rows]
    params_vals = [r["params_m"] for r in rows]
    auc_vals = [r["test_auc"] for r in rows]
    acc_vals = [r["test_acc"] for r in rows]
    gmean_vals = [r["test_gmean"] for r in rows]
    chance = rows[0]["chance_level"] if rows else 10.0

    x = np.arange(n)
    width = 0.7

    # ── 2×2 subplot layout ──
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.5))

    # Flatten for easy iteration
    panels = [
        (axes[0, 0], params_vals, "Params (M)", "Parameters", 0, max(params_vals) * 1.25),
        (axes[0, 1], auc_vals, "Test AUC (%)", "ROC-AUC", 60, 105),
        (axes[1, 0], acc_vals, "Test Acc (%)", "Accuracy", 60, 105),
        (axes[1, 1], gmean_vals, "G-Mean (%)", "G-Mean", 0, 105),
    ]

    for ax, vals, ylabel, title, ymin, ymax in panels:
        bars = ax.bar(
            x, vals, width,
            color=COLORS[:n], edgecolor="black", linewidth=0.6,
            zorder=3,
        )
        # Annotate values
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + (ymax - ymin) * 0.02,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold",
            )

        # Chance line (skip for Params)
        if title != "Parameters":
            ax.axhline(y=chance, color="black", linestyle="--", linewidth=0.8, zorder=2)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9.5)
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.set_ylim(ymin, ymax)
        ax.grid(axis="y", alpha=0.2, color="gray", zorder=0)

    # ── Shared legend at the top ──
    # fig.suptitle("Model Comparison on Transformer Acoustic Dataset\n"
    #              "(10-class, AW-DPCNN fused representations, seed 42)",
    #              fontsize=13, fontweight="bold", y=1.01)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "model_comparison_panel.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved to {out_path}")

    # ── Print table ──
    print(f"\n{'Model':<22s} {'Params':>7s} {'AUC%':>7s} {'Acc%':>7s} {'F1%':>7s} {'G-Mean%':>8s}")
    print("-" * 65)
    for r in rows:
        print(f"{r['label'].replace(chr(10),' '):<22s} {r['params_m']:6.1f}M "
              f"{r['test_auc']:7.2f} {r['test_acc']:7.2f} "
              f"{r['test_f1']:7.2f} {r['test_gmean']:8.2f}")

# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    rows = collect_metrics()
    if not rows:
        print("[ERROR] No test_metrics.json files found.")
        sys.exit(1)
    plot_comparison(rows)