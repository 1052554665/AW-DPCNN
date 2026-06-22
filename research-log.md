for this project, how to explain the `roc_curve` and auc in the context of multi-class classification? and implement it in the paper.


## Added: Multi-Class ROC-AUC Explanation (Section 5.2)

### Key points explained

| Concept | Explanation |
|---------|-------------|
| **Strategy** | One-vs-Rest (OvR) — treat each class as positive, all others as negative |
| **Per-class ROC** | $\mathrm{TPR}_c$ vs $\mathrm{FPR}_c$ at varying thresholds |
| **Macro-averaged AUC** | $\mathrm{AUC}_{\mathrm{macro}} = \frac{1}{C} \sum \mathrm{AUC}_c$ — equal weight to all classes |
| **Why macro** | Robust to class imbalance; each fault type contributes equally |
| **Implementation** | `sklearn.roc_auc_score(multi_class="ovr", average="macro")` |

### The math in the paper

For $C$ classes, each class $c$ gets its own binary ROC. The macro AUC is the unweighted mean:

$$\mathrm{AUC}_{\mathrm{macro}} = \frac{1}{C} \sum_{c=1}^{C} \mathrm{AUC}_c$$

This matches how metrics.py computes it with `average="macro"` and `multi_class="ovr"`.

### Contrast with alternatives (why not these)

| Strategy | Why not used |
|----------|-------------|
| **Micro-average** | Weighted by class frequency → majority classes dominate, misleading under imbalance |
| **Weighted-average** | Same problem as micro |
| **OvO (One-vs-One)** | $\binom{C}{2}$ pairs → too many curves for 10 classes, hard to interpret |

Made changes.