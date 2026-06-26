create a script to analyze the experimental results, including the following functionalities:
- the trail seeds can be specified as a command line argument
- ablation studies, hyperparameter sensitivity analysis, representation comparison, noise robustness comparison, model performance comparison and others are all included
- all the results should be saved in a structured format (latex tables). just like the table in the paper


## analyze_results.py — Unified Result Analyzer

**Generates publication-ready LaTeX tables** matching the paper format from all experiment results.

### Usage

```bash
# All 5 table types, 3 independent trials (mean ± std)
python scripts/analyze_results.py --trial-seeds 42,123,456

# Single trial only
python scripts/analyze_results.py --trial-seeds 42

# Specific analyses only
python scripts/analyze_results.py --trial-seeds 42,123,456 --analyses ablation,backbone

# Dry-run (print to stdout)
python scripts/analyze_results.py --dry-run
```

### Generated Tables

| Analysis | Output File | LaTeX Label |
|---|---|---|
| Ablation (B0–B8) | `ablation_table.tex` | `tab:ablation_unified` |
| Backbone comparison | `backbone_comparison_table.tex` | `tab:network_comparison` |
| Representation comparison | `rep_compare_table.tex` | `tab:rep_compare` |
| Hyperparameter sensitivity | `hyperparam_sensitivity_table.tex` | `tab:hyperparam_sensitivity` |
| Noise robustness | `noise_robustness_table.tex` | `tab:noise_robustness` |
| **All combined** | `all_tables.tex` | — |

### Key Features

- **Trial seed selection**: `--trial-seeds 42,123,456` — aggregates across seeds with mean ± std
- **Dynamic best-detection**: Backbone table bolds the actual best value per column (not hardcoded)
- **Adaptive captions**: Shows "Seed 42" for single-trial, "Mean ± Std over 3 Independent Trials" for multi-trial
- **Kappa scaling**: Cohen's κ now properly scaled to percentage (fixed in aggregation.py)
- **Output**: All `.tex` files saved to auto_tables, can be `\input{}` directly into `tim.tex`

### Output Structure

```
paper/auto_tables/
├── ablation_table.tex
├── backbone_comparison_table.tex
├── rep_compare_table.tex
├── hyperparam_sensitivity_table.tex
├── noise_robustness_table.tex
└── all_tables.tex          # Combined file
```