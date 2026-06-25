>can the  ROC curves be plotted for all the models in one script without retraining the models?

Run with:

```bash
python scripts/plot_roc_all.py
```

**What it does:**

| Feature | Details |
|---|---|
| **No retraining** | Loads existing `best.pt` checkpoints + `resolved_config.yaml` |
| **Auto-discovery** | Finds all 6 models under exp1 |
| **`torch.compile` fix** | Strips `_orig_mod.` prefix from saved state dicts automatically |
| **Trial selection** | `--trial trial_seed42` (default), `--trial trial_seed456` etc. |

**Outputs** (saved to roc):

| File | Description |
|---|---|
| `msca-vgg16_roc.png` … `vit_roc.png` | Per-model plots: 10 class curves + micro/macro avg |
| `roc_comparison_macro_avg.png` | Combined comparison: all 6 models overlaid (macro-average only) |

All fonts are Times New Roman with STIX math, IEEE-ready.

Made changes.