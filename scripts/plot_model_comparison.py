#!/usr/bin/env python3
"""
Model Comparison — Grouped Bar Chart
======================================
Collects test metrics from each model's test_metrics.json (trial seed 42)
and produces a grouped bar chart with four metrics per model:
  Accuracy, F1-score, AUC, G-Mean.

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
TRIAL = "trial_seed456"
METRICS_FILE = "results/test_metrics.json"

MODEL_LABELS = {
    "convnext-tiny":      "ConvNeXt\nTiny",
    "efficientnet-b0":    "Efficient\nNet-B0",
    "mobilenetv3_small":  "MobileNet\nV3-Small",
    "msca-vgg16":         "MSCA-VGG16\n(Ours)",
    "vgg16":              "VGG16",
    "vit":                "ViT",
}

COLORS   = ["#404040", "#808080", "#B0B0B0", "#D8D8D8"]
HATCHES  = ["", "//", "\\\\", "xx"]

OUTPUT_DIR = Path("paper/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Collect ────────────────────────────────────────────────────────────
def collect_metrics() -> list:
    rows = []
    for exp_dir in sorted(RESULT_ROOT.glob("*")):
        if not exp_dir.is_dir():
            continue
        inner = sorted(exp_dir.glob("exp1_*"))
        if not inner:
            continue
        path = inner[0] / TRIAL / METRICS_FILE
        if not path.exists():
            print(f"[SKIP] {path}")
            continue
        d = json.loads(path.read_text())
        name = d.get("model_name", exp_dir.name)
        rows.append({
            "label":  MODEL_LABELS.get(name, name),
            "acc":    d["test_acc"] * 100,
            "f1":     d["test_f1"] * 100,
            "auc":    d["test_auc"] * 100,
            "gmean":  d["test_gmean"] * 100,
            "params": d["params"] / 1e6,
            "chance": d.get("chance_level", 0.1) * 100,
        })
    def _key(r):
        return (0 if "ours" in r["label"].lower() else 1, -r["acc"])
    return sorted(rows, key=_key)


# ── Plot ───────────────────────────────────────────────────────────────
def plot(rows: list):
    n = len(rows)
    labels  = [r["label"] for r in rows]
    params  = [r["params"] for r in rows]
    chance  = rows[0].get("chance", rows[0].get("chance_level", 10.0))

    metrics = [
        ("Acc",    [r["acc"]   for r in rows]),
        ("F1",     [r["f1"]    for r in rows]),
        ("AUC",    [r["auc"]   for r in rows]),
        ("G-Mean", [r["gmean"] for r in rows]),
    ]
    m = len(metrics)
    w = 0.18
    offsets = np.linspace(-w * (m - 1) / 2, w * (m - 1) / 2, m)

    fig, ax = plt.subplots(figsize=(7.0, 3.8))

    for i, (name, vals) in enumerate(metrics):
        bars = ax.bar(
            np.arange(n) + offsets[i], vals, w,
            color=COLORS[i], edgecolor="black", linewidth=0.5,
            hatch=HATCHES[i], label=name, zorder=3,
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}", ha="center", va="bottom",
                fontsize=7.5, fontweight="bold", rotation=90,
            )

    ax.axhline(y=chance, color="black", linestyle="--", linewidth=1.0, zorder=2)
    ax.text(-0.55, chance + 0.6, f"Chance ({chance:.0f}%)", fontsize=8, va="bottom")

    for i, p in enumerate(params):
        ax.text(i, -4, f"{p:.1f}M", ha="center", fontsize=8,
                color="dimgray", fontweight="bold")

    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Score (%)", fontsize=10)
    ax.set_ylim(-8, 110)
    # ax.set_title("Model Comparison — Transformer Acoustic Dataset\n"
    #              "(10-class, AW-DPCNN fused, seed 42)",
    #              fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9.5, framealpha=0.95,
              edgecolor="gray", ncol=4)
    ax.grid(axis="y", alpha=0.2, color="gray", zorder=0)

    plt.tight_layout()
    out = OUTPUT_DIR / "model_comparison_grouped.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved  {out}")

    # Table
    hdr = f"{'Model':<22s} {'Params':>7s} {'Acc%':>7s} {'F1%':>7s} {'AUC%':>7s} {'G-Mean%':>8s}"
    print(f"\n{hdr}\n{'-'*len(hdr)}")
    for r in rows:
        print(f"{r['label'].replace(chr(10),' '):<22s} {r['params']:6.1f}M "
              f"{r['acc']:7.2f} {r['f1']:7.2f} {r['auc']:7.2f} {r['gmean']:8.2f}")


if __name__ == "__main__":
    rows = collect_metrics()
    if not rows:
        print("[ERROR] No test_metrics.json files found.")
        sys.exit(1)
    plot(rows)
