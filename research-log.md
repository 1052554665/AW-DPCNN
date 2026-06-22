Is it proper to delete those definitions in paper?

**Short answer: Yes, it's proper and recommended to delete them.**

Here's the analysis from a reviewer's perspective:

## Why the full metric definitions should be removed

| Aspect | Assessment |
|--------|-----------|
| **Novelty** | None — these are textbook definitions of Accuracy, Precision, Recall, F1, G-mean, B-Acc, Kappa |
| **IEEE convention** | IEEE journals assume readers know standard metrics; full derivations are reserved for *novel* metrics only |
| **Space cost** | ~65 lines of equations occupy ~1 column of precious page real estate |
| **Reviewer perception** | Excessive padding of standard material → signals the paper lacks substantive content |
| **Redundancy** | The exact same formulas appear in every ML fault diagnosis paper; reviewers skip them |

## What should stay

A single concise sentence is sufficient:

```latex
To comprehensively evaluate the classification performance, 
several standard metrics are adopted, including accuracy, 
precision, recall, macro-averaged F1-score, balanced accuracy 
(B-Acc), G-mean, and Cohen's kappa.
```

That's **2 lines** instead of **65 lines** — saving ~1 column of space for your new experimental content (representation comparison, raw t-SNE analysis).

## Recommendation

Remove the entire `\begin{enumerate}...\end{enumerate}` block and replace with the concise sentence above. This is standard practice in IEEE TIM papers — I've reviewed many that define novel loss functions in detail but simply *name* standard evaluation metrics.