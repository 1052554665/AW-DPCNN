For feature comparison experiment in script `script/representation_comparison.py`, create a script to compare the accuracy and other metrics of different features under the same classifier. Only one command used to run the script, and the script will automatically load the features, train the classifier, and output the results.


## Created: run_representation_comparison.py

**One command to run all 12 representation comparisons:**

```bash
python scripts/run_representation_comparison.py
```

### What it does

| Step | Description |
|------|-------------|
| 1 | Auto-discovers all 12 `datasets/rep_compare/{mel,stft,cwt}_{gadf,gasf,mtf,rp}/` directories |
| 2 | For each combination, trains the **same** classifier with identical hyperparameters |
| 3 | Saves per-run checkpoints, logs, and resolved configs |
| 4 | Outputs a ranked comparison table + grouped analysis + CSV summary |

### Key flags

| Flag | Default | Purpose |
|------|---------|---------|
| `--model` | `msca_vgg16` | Classifier (vgg16, convnext_tiny, efficientnet_b0, etc.) |
| `--epochs` | `30` | Training epochs per run |
| `--tf` | `all` | Filter: `mel`, `stft`, `cwt`, or comma-separated |
| `--temporal` | `all` | Filter: `gadf`, `gasf`, `mtf`, `rp` |
| `--seed` | `42` | Random seed for reproducibility |
| `--dry-run` | — | List combinations without training |

### Example outputs

```text
 Rank  TF     Temporal    Acc %     F1 %   G-mean %   B-Acc %   Kappa %
───────────────────────────────────────────────────────────────────────
 1     Mel    GADF       99.92   99.92     99.90    99.90    99.89
 2     CWT    GADF        ...
 ...

  Grouped by Time‑Frequency method (average Accuracy %)
──────────────────────────────────────────────────────────────────────
  Mel    avg Acc = xx.xx%  (n=4)
  STFT   avg Acc = xx.xx%  (n=4)
  CWT    avg Acc = xx.xx%  (n=4)

  ★ Best: CWT × GADF  → Acc = xx.xx%, F1 = xx.xx%
```

Results are saved to `experiments/rep_compare_results/{model}/comparison_summary_{timestamp}.csv`.

Made changes.

