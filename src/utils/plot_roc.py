"""Plot multi-class ROC curves (one-vs-rest) with macro/micro averages."""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


def plot_roc_curves(
    y_true,
    y_score,
    class_names,
    save_path="figures/roc_curve.png",
):
    """Plot and save multi-class ROC curves.

    Args:
        y_true: 1-d array of ground-truth class indices.
        y_score: 2-d array of predicted probabilities (shape [N, num_classes]).
        class_names: list of class name strings.
        save_path: path to save the figure.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    n_classes = len(class_names)

    # Binarize labels for one-vs-rest
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    # ---- IEEE-style global settings ----
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "stix",
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 15,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.dpi": 300,
        "axes.linewidth": 0.8,
    })

    # ---- Compute per-class ROC ----
    fpr = {}
    tpr = {}
    roc_auc = {}

    colors = plt.cm.tab10(np.linspace(0, 1, max(n_classes, 3)))

    fig, ax = plt.subplots(figsize=(7, 6))

    for i in range(n_classes):
        if y_true_bin[:, i].sum() == 0:
            continue  # skip classes absent from test set
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        ax.plot(
            fpr[i], tpr[i],
            color=colors[i % len(colors)],
            lw=1.5,
            label=f"{class_names[i]} (AUC={roc_auc[i]:.3f})",
        )

    # ---- Micro-average ----
    fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)
    ax.plot(
        fpr_micro, tpr_micro,
        color="navy", linestyle=":", lw=1.8,
        label=f"Micro-avg (AUC={roc_auc_micro:.3f})",
    )

    # ---- Macro-average ----
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes) if i in fpr]))
    mean_tpr = np.zeros_like(all_fpr)
    count = 0
    for i in range(n_classes):
        if i not in fpr:
            continue
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        count += 1
    if count > 0:
        mean_tpr /= count
        mean_tpr[0] = 0.0
        mean_tpr[-1] = 1.0
        roc_auc_macro = auc(all_fpr, mean_tpr)
        ax.plot(
            all_fpr, mean_tpr,
            color="darkred", linestyle="--", lw=1.8,
            label=f"Macro-avg (AUC={roc_auc_macro:.3f})",
        )

    # ---- Diagonal reference ----
    ax.plot([0, 1], [0, 1], color="grey", lw=1.0, linestyle="--", alpha=0.6)

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (Test Set)")
    ax.legend(loc="lower right", framealpha=0.9)

    plt.tight_layout()

    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=300, bbox_inches="tight")
    plt.close()
