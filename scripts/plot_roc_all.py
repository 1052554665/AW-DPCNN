#!/usr/bin/env python3
"""
ROC Curves — Batch Plotter (No Retraining Required)
=====================================================
Loads saved checkpoints and resolved configs for all trained models,
evaluates on the test set, and generates:

  1. Per‑model ROC plots (10‑class curves + micro/macro averages)
  2. A combined macro‑average ROC comparison across all models

All from existing checkpoints — no retraining needed.

Usage::

# Default (backward compat — broad search)
python scripts/plot_roc_all.py

# Scoped to 12k_de dataset
python scripts/plot_roc_all.py --dataset 48k_de

# Specific trial + dataset
python scripts/plot_roc_all.py --dataset 12k_de --trial trial_seed42

# Full run: plots + table
python scripts/plot_roc_all.py --dataset 48k_de --trial trial_seed42

# Custom table output directory
python scripts/plot_roc_all.py --dataset 12k_de --table-dir paper/auto_tables

"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.datasets import build_dataloaders
from src.models import build_model
from src.utils.config import load_yaml
from src.utils.dataset_registry import DATASET_KEYS

# ── IEEE-compatible font configuration ────────────────────────────────
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"

# ── Configuration ──────────────────────────────────────────────────────
RESULT_ROOT = Path("experiments/experiment_result/exp1")
_DEFAULT_OUTPUT = Path("paper/figures/roc")

MODEL_DISPLAY = {
    "convnext-tiny":      "ConvNeXt-Tiny",
    "efficientnet-b0":    "EfficientNet-B0",
    "mobilenetv3_small":  "MobileNetV3-Small",
    "msca-vgg16":         "MSCA-VGG16 (Ours)",
    "resnet18":           "ResNet18",
    "vgg16":              "VGG16",
    "vit":                "ViT",
}

LINE_STYLES = {
    "msca-vgg16":         {"color": "#D4A574", "lw": 2.5, "ls": "-"},
    "vgg16":              {"color": "#505050", "lw": 2.0, "ls": "-"},
    "efficientnet-b0":    {"color": "#909090", "lw": 2.0, "ls": "--"},
    "vit":                {"color": "#303030", "lw": 2.0, "ls": "-."},
    "mobilenetv3_small":  {"color": "#707070", "lw": 2.0, "ls": ":"},
    "convnext-tiny":      {"color": "#B0B0B0", "lw": 2.0, "ls": "-"},
}


def discover_models(trial: str, dataset_key: str = "") -> list:
    """Find all model runs with checkpoints and resolved configs.

    When ``dataset_key`` is provided (e.g. ``"12k_de"``), results are
    read from ``RESULT_ROOT / model / dataset_key / exp1_* / trial``.

    Returns list of dicts with keys: model_name, display_name, checkpoint, config.
    """
    models = []
    for exp_dir in sorted(RESULT_ROOT.glob("*")):
        if not exp_dir.is_dir():
            continue
        # Search for exp1_* directories: scoped by dataset if given
        if dataset_key:
            inner = sorted(exp_dir.glob(f"{dataset_key}/exp1_*"))
        else:
            inner = sorted(exp_dir.glob("exp1_*"))
            if not inner:
                # Fallback: look one level deeper for any dataset
                inner = sorted(exp_dir.glob("*/exp1_*"))
        if not inner:
            continue
        run_dir = inner[0]
        trial_dir = run_dir / trial
        ckpt = trial_dir / "checkpoints" / "best.pt"
        cfg = run_dir / "resolved_config.yaml"

        if not ckpt.exists():
            print(f"[SKIP] No checkpoint: {ckpt}")
            continue
        if not cfg.exists():
            # Try trial dir
            cfg = trial_dir / "resolved_config.yaml"
        if not cfg.exists():
            print(f"[SKIP] No resolved config for {exp_dir.name}")
            continue

        # Read model name from test_metrics.json if available
        metrics_path = trial_dir / "results" / "test_metrics.json"
        model_name = exp_dir.name  # fallback
        if metrics_path.exists():
            with open(metrics_path) as f:
                d = json.load(f)
            model_name = d.get("model_name", model_name)

        display = MODEL_DISPLAY.get(model_name, model_name)
        models.append({
            "model_name": model_name,
            "display_name": display,
            "checkpoint": ckpt,
            "config": cfg,
        })
        print(f"  Found: {display}")

    # Sort: MSCA-VGG16 first, then by name
    def _key(m):
        return (0 if "ours" in m["display_name"].lower() else 1, m["display_name"])
    return sorted(models, key=_key)


def _strip_orig_mod(state_dict: dict) -> dict:
    """Strip '_orig_mod.' prefix from torch.compile state dict keys."""
    new_sd = {}
    for k, v in state_dict.items():
        new_sd[k.replace("_orig_mod.", "")] = v
    return new_sd


def evaluate_one(model_info: dict, device: torch.device) -> dict:
    """Load model, run evaluation, return y_true, y_score, class_names."""
    config = load_yaml(str(model_info["config"]))
    model = build_model(config).to(device)
    sd = torch.load(str(model_info["checkpoint"]), map_location=device)
    # Handle torch.compile _orig_mod. prefix
    if any(k.startswith("_orig_mod.") for k in sd.keys()):
        sd = _strip_orig_mod(sd)
    model.load_state_dict(sd)
    model.eval()

    loaders, class_names = build_dataloaders(config)
    criterion = nn.CrossEntropyLoss()

    from src.utils.train_eval import evaluate
    _, _, y_true, _, y_score = evaluate(model, loaders["test"], criterion, device)

    return {
        "display_name": model_info["display_name"],
        "model_name": model_info["model_name"],
        "y_true": y_true,
        "y_score": y_score,
        "class_names": class_names,
    }


def plot_per_model(results: list, output_dir: Path) -> None:
    """Generate individual ROC plots for each model."""
    for r in results:
        y_true = np.asarray(r["y_true"], dtype=int)
        y_score = np.asarray(r["y_score"], dtype=float)
        class_names = r["class_names"]
        n_classes = len(class_names)
        y_true_bin = label_binarize(y_true, classes=range(n_classes))

        fig, ax = plt.subplots(figsize=(6.5, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

        fpr_dict, tpr_dict, auc_dict = {}, {}, {}
        for i in range(n_classes):
            if y_true_bin[:, i].sum() == 0:
                continue
            fpr_dict[i], tpr_dict[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
            ax.plot(fpr_dict[i], tpr_dict[i], color=colors[i], lw=1.2,
                    label=f"{class_names[i]} ({auc_dict[i]:.3f})")

        # Micro-avg
        fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
        ax.plot(fpr_micro, tpr_micro, color="navy", ls=":", lw=1.6,
                label=f"Micro-avg ({auc(fpr_micro, tpr_micro):.3f})")

        # Macro-avg
        all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in fpr_dict]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in fpr_dict:
            mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
        mean_tpr /= len(fpr_dict)
        mean_tpr[0], mean_tpr[-1] = 0.0, 1.0
        roc_auc_macro = auc(all_fpr, mean_tpr)
        ax.plot(all_fpr, mean_tpr, color="darkred", ls="--", lw=1.8,
                label=f"Macro-avg ({roc_auc_macro:.3f})")

        ax.plot([0, 1], [0, 1], color="grey", lw=0.8, ls="--", alpha=0.5)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel("False Positive Rate", fontsize=18)
        ax.set_ylabel("True Positive Rate", fontsize=18)
        ax.legend(loc="lower right", fontsize=12.5, framealpha=0.9, ncol=2)
        ax.grid(alpha=0.15, color="gray")

        plt.tight_layout()
        # Sanitize filename
        fname = r["model_name"].replace(" ", "_").lower() + "_roc.png"
        out = output_dir / fname
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out}")


def plot_combined_macro(results: list, output_dir: Path) -> None:
    """Generate a single figure comparing macro-average ROC across all models."""
    fig, ax = plt.subplots(figsize=(6.5, 6))

    for r in results:
        y_true = np.asarray(r["y_true"], dtype=int)
        y_score = np.asarray(r["y_score"], dtype=float)
        n_classes = len(r["class_names"])
        y_true_bin = label_binarize(y_true, classes=range(n_classes))

        fpr_dict, tpr_dict = {}, {}
        for i in range(n_classes):
            if y_true_bin[:, i].sum() == 0:
                continue
            fpr_dict[i], tpr_dict[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])

        if not fpr_dict:
            continue

        # Macro-average
        all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in fpr_dict]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in fpr_dict:
            mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
        mean_tpr /= len(fpr_dict)
        mean_tpr[0], mean_tpr[-1] = 0.0, 1.0
        roc_auc_macro = auc(all_fpr, mean_tpr)

        style = LINE_STYLES.get(r["model_name"],
                                {"color": "black", "lw": 2.0, "ls": "-"})
        ax.plot(all_fpr, mean_tpr,
                color=style["color"], lw=style["lw"], linestyle=style["ls"],
                label=f"{r['display_name']} (AUC={roc_auc_macro:.3f})")

    ax.plot([0, 1], [0, 1], color="grey", lw=0.8, ls="--", alpha=0.5,
            label="Chance")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=15)
    ax.set_ylabel("True Positive Rate", fontsize=15)
    ax.legend(loc="lower right", fontsize=10.5, framealpha=0.95)
    ax.grid(alpha=0.15, color="gray")

    plt.tight_layout()
    out = output_dir / "roc_comparison_macro_avg.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Combined macro-avg ROC: {out}")


def generate_roc_table(results: list, table_dir: Path) -> None:
    """Generate an IEEE-style LaTeX table of per-class AUC values.

    Produces ``roc_auc_table.tex`` in *table_dir*.
    """
    table_dir.mkdir(parents=True, exist_ok=True)

    # ── Compute AUC per class per model ──
    table_data = []  # list of {model_name, class_aucs: {cls: auc}, macro, micro}
    class_names_all = None

    for r in results:
        y_true = np.asarray(r["y_true"], dtype=int)
        y_score = np.asarray(r["y_score"], dtype=float)
        class_names = r["class_names"]
        if class_names_all is None:
            class_names_all = list(class_names)
        n_classes = len(class_names)
        y_true_bin = label_binarize(y_true, classes=range(n_classes))

        class_aucs = {}
        fpr_dict, tpr_dict = {}, {}
        for i in range(n_classes):
            if y_true_bin[:, i].sum() == 0:
                class_aucs[class_names[i]] = float("nan")
                continue
            fpr_i, tpr_i, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            fpr_dict[i] = fpr_i
            tpr_dict[i] = tpr_i
            class_aucs[class_names[i]] = auc(fpr_i, tpr_i)

        # Micro-average
        fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
        auc_micro = auc(fpr_micro, tpr_micro)

        # Macro-average
        valid_fprs = [fpr_dict[i] for i in fpr_dict]
        if valid_fprs:
            all_fpr = np.unique(np.concatenate(valid_fprs))
            mean_tpr = np.zeros_like(all_fpr)
            for i in fpr_dict:
                mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
            mean_tpr /= len(fpr_dict)
            mean_tpr[0], mean_tpr[-1] = 0.0, 1.0
            auc_macro = auc(all_fpr, mean_tpr)
        else:
            auc_macro = float("nan")

        table_data.append({
            "display_name": r["display_name"],
            "model_name": r["model_name"],
            "class_aucs": class_aucs,
            "auc_macro": auc_macro,
            "auc_micro": auc_micro,
        })

    # ── Build LaTeX (transposed: classes=rows, models=columns) ──
    lines = []
    lines.append("% Auto-generated by scripts/plot_roc_all.py")
    lines.append(f"% Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("\\begin{table*}[!htbp]")
    lines.append("    \\centering")
    lines.append("    \\caption{Per-Class AUC Values Across Backbone Networks.}")
    lines.append("    \\label{tab:roc_auc}")
    lines.append("    \\renewcommand{\\arraystretch}{1.05}")
    lines.append("    \\footnotesize")

    n_models = len(table_data)
    model_displays = [d["display_name"] for d in table_data]
    model_keys = [d["model_name"] for d in table_data]

    # Columns: Class + one per model
    col_spec = "l" + "c" * n_models
    lines.append(f"    \\begin{{tabular}}{{{col_spec}}}")
    lines.append("        \\toprule")

    # Header row: Class + model names
    header_cells = ["\\textbf{Fault Class}"]
    for md in model_displays:
        if "ours" in md.lower():
            header_cells.append(f"\\textbf{{{md}}}")
        else:
            header_cells.append(md)
    lines.append("        " + " & ".join(header_cells) + " \\\\")
    lines.append("        \\midrule")

    # Find best per row (per class) for bolding
    best_per_class = {}
    for cn in class_names_all:
        valid = [(d["display_name"], d["class_aucs"].get(cn, float("nan")))
                 for d in table_data
                 if cn in d["class_aucs"]
                 and d["class_aucs"][cn] == d["class_aucs"][cn]]
        if valid:
            best_per_class[cn] = max(valid, key=lambda x: x[1])[0]
    # Best macro / micro
    valid_macro = [(d["display_name"], d["auc_macro"]) for d in table_data
                   if d["auc_macro"] == d["auc_macro"]]
    best_macro = max(valid_macro, key=lambda x: x[1])[0] if valid_macro else None
    valid_micro = [(d["display_name"], d["auc_micro"]) for d in table_data
                   if d["auc_micro"] == d["auc_micro"]]
    best_micro = max(valid_micro, key=lambda x: x[1])[0] if valid_micro else None

    # Per-class rows
    for cn in class_names_all:
        row_cells = [cn]
        best_model = best_per_class.get(cn)
        for d in table_data:
            val = d["class_aucs"].get(cn, float("nan"))
            if val != val:
                v_str = "---"
            else:
                v_str = f"{val:.3f}"
            if best_model is not None and d["display_name"] == best_model:
                v_str = f"$\\mathbf{{{v_str}}}$"
            else:
                v_str = f"${v_str}$"
            row_cells.append(v_str)
        lines.append("        " + " & ".join(row_cells) + " \\\\")

    # Mid-rule before aggregates
    lines.append("        \\midrule")

    # Macro row
    macro_cells = ["\\textbf{Macro}"]
    for d in table_data:
        mac = d["auc_macro"]
        if mac != mac:
            s = "---"
        else:
            s = f"{mac:.3f}"
        if best_macro is not None and d["display_name"] == best_macro:
            s = f"$\\mathbf{{{s}}}$"
        else:
            s = f"${s}$"
        macro_cells.append(s)
    lines.append("        " + " & ".join(macro_cells) + " \\\\")

    # Micro row
    micro_cells = ["\\textbf{Micro}"]
    for d in table_data:
        mic = d["auc_micro"]
        if mic != mic:
            s = "---"
        else:
            s = f"{mic:.3f}"
        if best_micro is not None and d["display_name"] == best_micro:
            s = f"$\\mathbf{{{s}}}$"
        else:
            s = f"${s}$"
        micro_cells.append(s)
    lines.append("        " + " & ".join(micro_cells) + " \\\\")

    lines.append("        \\bottomrule")
    lines.append("    \\end{tabular}")
    lines.append("\\end{table*}")

    out_path = table_dir / "roc_auc_table.tex"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"\n  ROC AUC table: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch ROC curve plotter — no retraining required",
    )
    parser.add_argument("--trial", default="trial_seed42",
                        help="Trial subdirectory name (default: trial_seed42)")
    parser.add_argument("--dataset", default="", choices=[""] + DATASET_KEYS,
                        help=f"Dataset key to scope model discovery "
                             f"{{{','.join(DATASET_KEYS)}}}")
    parser.add_argument("--output", default="paper/figures/roc",
                        help="Output directory for ROC figures")
    parser.add_argument("--table-dir", default="paper/auto_tables",
                        help="Output directory for LaTeX table (default: paper/auto_tables)")
    parser.add_argument("--device", default="cuda",
                        help="Device for inference (cuda/cpu)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Trial:  {args.trial}")
    if args.dataset:
        print(f"Dataset: {args.dataset}")
    print(f"Output: {output_dir}")
    print()

    # ── Discover models ──
    print("Discovering models...")
    models = discover_models(args.trial, args.dataset)
    if not models:
        print("[ERROR] No models found.")
        sys.exit(1)
    print(f"  {len(models)} model(s) found.\n")

    # ── Evaluate each model ──
    print("Evaluating models on test set...")
    results = []
    for m in models:
        print(f"  Evaluating: {m['display_name']}...")
        try:
            r = evaluate_one(m, device)
            results.append(r)
        except Exception as exc:
            print(f"    [FAIL] {exc}")

    if not results:
        print("[ERROR] No models evaluated successfully.")
        sys.exit(1)

    # ── Plot ──
    print(f"\nGenerating per-model ROC plots...")
    plot_per_model(results, output_dir)

    print(f"\nGenerating combined macro-average comparison...")
    plot_combined_macro(results, output_dir)

    # ── LaTeX AUC table ──
    print(f"\nGenerating ROC AUC LaTeX table...")
    table_dir = Path(args.table_dir)
    generate_roc_table(results, table_dir)

    print(f"\nDone. All ROC plots saved to {output_dir}")


if __name__ == "__main__":
    main()
