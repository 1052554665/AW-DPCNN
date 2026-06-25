create a script, which functions are as follows:
- analyse all the experiments result located in the `logs` and `results` directories
- transform the results into table format or chart format (generate latex code), which fits the IEEE journal format


a comprehensive script that:

- Reads all experiment results from `experiments/experiment_result/`
- Analyzes both `logs/train_log.csv` (training curves) and `results/test_metrics.json` (final metrics)
- Generates LaTeX tables and/or charts in IEEE format

### tab_exp1_comparison.tex — Backbone Performance

| Network | Acc | F1 | G-Mean | AUC | $\kappa$ | Params (M) |
|---|---|---|---|---|---|---|
| **MSCA-VGG16 (Ours)** | **95.1** | **92.9** | **90.3** | 99.9 | 94.3 | 26.8 |
| VGG16 | 94.7 | 91.9 | 89.0 | **99.9** | 93.8 | 15.3 |
| ViT | 88.0 | 84.1 | 82.3 | 99.2 | 85.9 | 11.0 |
| EfficientNet-B0 | 91.0 | 86.8 | 83.1 | 98.9 | 89.4 | 4.0 |
| MobileNetV3-Small | 84.7 | 79.3 | 75.0 | 98.2 | 82.0 | 1.5 |
| ConvNeXt-Tiny | 76.4 | 66.0 | 45.1 | 97.1 | 72.3 | 27.8 |

### tab_exp1_training.tex — Training Summary

Includes best epoch, val F1, params, FLOPs, and shuffled-label check.

### Usage

```bash
python scripts/results_to_latex.py                    # seed 42
python scripts/results_to_latex.py trial_seed123      # other trial
```

New experiment groups (e.g., `ablation/`) are automatically picked up when their results exist.