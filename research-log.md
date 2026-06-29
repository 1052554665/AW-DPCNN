>create a script that finds all confusion matrix PNG files in the `experiments/experiment_result` directory tree and renames them with a consistent naming convention: `confusion_matrix_{dataset}_{model}_{experiment_name}_{trial_seed}.png`.

### Naming convention

| Source | Example output filename |
|---|---|
| `exp1/{model}/{dataset}/...` | `confusion_matrix_12k_de_MSCA_VGG16_exp1_MSCA_VGG16_trial_seed42.png` |
| `rep_compare/{combo}/...` | `confusion_matrix_cwt_gadf_msca-vgg16_rep_compare_cwt_gadf_trial_seed123.png` |
| `ablation_results/{B#}/...` | `confusion_matrix_ablation_B0_ablation_B0_trial_seed456.png` |

### Usage

```bash
# Dry run (preview without copying)
python scripts/collect_confusion_matrices.py --dry-run

# Copy all to paper/figures
python scripts/collect_confusion_matrices.py

# Custom output directory
python scripts/collect_confusion_matrices.py --output-dir paper/figures/confusion_matrix
```

Made changes.