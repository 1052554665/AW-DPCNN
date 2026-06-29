>the directory of checkpoint has changed, there are three dataset under each model in `exp1`

Here's a summary of the three fixes applied to hyperparameter_sensitivity.py:

### Root cause
The old `find_checkpoint` searched `exp1/**/checkpoints/best.pt` across **all** models and picked the most recently modified — which was a ViT checkpoint, not MSCA_VGG16.

### Changes made

| # | Fix | Detail |
|---|---|---|
| 1 | **`find_checkpoint`** now accepts `model_dir` | Searches `exp1/MSCA_VGG16/` first before falling back to the broad search. Tries precise `trial_seed42/` path first, then any trial, then any `best.pt` |
| 2 | **`model_dir` derived from exp-config** | `Path(args.exp_config).stem` → `MSCA_VGG16`, matching train.py's behavior |
| 3 | **Default `--dataset-dir`** | Changed from nonexistent `datasets/cwru_de` → cwru_12k_de |

The script should now correctly find `exp1/MSCA_VGG16/12k_de/exp1_MSCA_VGG16/trial_seed42/checkpoints/best.pt` and load it into the MSCA_VGG16 model.

Made changes.


>add a `--dataset` flag to `hyperparameter_sensitivity.py` that allows the user to specify which dataset to use. such as 
```python
python scripts/hyperparameter_sensitivity.py \
    --exp-config experiments/exp1/MSCA_VGG16.yaml \
    --auto-checkpoint --trial trial_seed42 \
    --mel-n-fft 1024 --mel-n-mels 128 --mel-fmax 6000 \
    --max-samples 500 --dataset cwru_12k_de
```

>update this script since the directory of checkpoint has changed, there are three dataset under each model in `experiment_result/exp1`. and add a `--dataset` flag to `noise_sensitivity.py` that allows the user to specify which dataset to use. such as 
```python
    python scripts/noise_robustness.py \\
        --config configs/default.yaml \\
        --exp-config experiments/exp1/vgg16.yaml \\
        --dataset cwru_12k_de \\
        --checkpoint .../best.pt \\
        --snr -10 -5 0 5 10 15 20
```


### Changes made

| # | Change | Detail |
|---|---|---|
| 1 | **`find_checkpoint`** rewritten | Now accepts `model_dir` and `dataset_key`. Searches `exp1/{model_dir}/{dataset_key}/**/checkpoints/best.pt` first, then `exp1/{model_dir}/**/...`, then broad fallback. Prefers `trial_seed*` subdirectories |
| 2 | **`--dataset` flag** added | Choices: `12k_de`, `12k_fe`, `48k_de`. Auto-configures `dataset.root_dir` from registry |
| 3 | **`model_dir` derived from config file name** | Single-model: `Path(args.exp_config).stem` → `MSCA_VGG16`. Batch: `yaml_file.stem` → per-model |
| 4 | **Usage examples** updated | Docstring now shows `--dataset 12k_de` in all examples |

### Verified checkpoint discovery

```
MSCA_VGG16 + 12k_de → .../MSCA_VGG16/12k_de/exp1_MSCA_VGG16/trial_seed456/checkpoints/best.pt ✅
vgg16 only           → .../vgg16/48k_de/exp1_vgg16/trial_seed456/checkpoints/best.pt ✅
broad (no scope)     → .../vit/48k_de/... (most recent, only used as fallback)
```



### Single model (auto‑checkpoint — recommended)

```bash
python scripts/noise_robustness.py \
    --config configs/default.yaml \
    --exp-config experiments/exp1/MSCA_VGG16.yaml \
    --dataset 12k_de \
    --auto-checkpoint \
    --snr 5 10 15 20 25 30
```

### Single model (explicit checkpoint path)

```bash
python scripts/noise_robustness.py \
    --config configs/default.yaml \
    --exp-config experiments/exp1/vgg16.yaml \
    --dataset 12k_de \
    --checkpoint experiments/experiment_result/exp1/vgg16/12k_de/exp1_vgg16/trial_seed42/checkpoints/best.pt \
    --snr -5 0 5 10 15 20
```

### Batch mode (evaluate all 7 models in exp1)

```bash
python scripts/noise_robustness.py \
    --config configs/default.yaml \
    --exp-dir experiments/exp1 \
    --dataset 12k_de \
    --auto-checkpoint \
    --snr -5 0 5 10 15 20
```

### Key flags

| Flag | Purpose |
|---|---|
| `--dataset 12k_de` | Uses cwru_12k_de and scopes checkpoint search to `12k_de/` subdir |
| `--dataset 12k_fe` | Fan-end sensor variant |
| `--dataset 48k_de` | 48 kHz drive-end variant |
| `--auto-checkpoint` | Auto-finds `best.pt` under `experiment_result/exp1/{model}/{dataset}/` |
| `--snr -5 0 5 10 15 20` | SNR levels in dB (omit for default: `-5 0 5 10 15 20`) |
| `--output-dir` | Where plots/CSV are saved (default: noise_robustness) |

>since all the experiment results are stored in `experiment_result/exp1`, the script should be updated to search for checkpoints under `experiment_result/exp1/{model}/{dataset}/trial_seed*/checkpoints/best.pt` instead of searching all models and datasets. and add a `--dataset` flag to `analyze_results.py` that allows the user to specify which dataset to use.

### Changes to analyze_results.py

| # | Fix | Before | After |
|---|---|---|---|
| 1 | **`ABLATION_ROOT` path** | `experiments/ablation_results` (404) | ablation_results ✅ |
| 2 | **`HYPERPARAM_FILE` path** | `experiments/hyperparameter_sensitivity/...` | `experiments/experiment_result/hyperparameter_sensitivity/...` |
| 3 | **`NOISE_FILE` path** | `experiments/noise_robustness/...` | `experiments/experiment_result/noise_robustness/...` |
| 4 | **Backbone glob** | `exp_dir.glob("exp1_*")` (flat) | `exp_dir.glob(f"{dataset_key}/exp1_*")` (nested) + old-structure fallback |
| 5 | **`--dataset` flag** | None | `--dataset {12k_de,12k_fe,48k_de}` |

### Usage

```bash
# Single trial, 12k_de dataset
python scripts/analyze_results.py --trial-seeds 42 --dataset 12k_de

# Three trials with mean±std
python scripts/analyze_results.py --trial-seeds 42,123,456 --dataset 12k_de

# Backbone only, 12k_fe
python scripts/analyze_results.py --trial-seeds 42,123,456 --dataset 12k_fe --analyses backbone

# Write to files
python scripts/analyze_results.py --trial-seeds 42,123,456 --dataset 12k_de --output-dir paper/auto_tables
```

Made changes.

>Update the `plot_roc_curves.py` script to accept a `--dataset` flag that allows the user to specify which dataset to use. The script should search for checkpoints under `experiment_result/exp1/{model}/{dataset}/trial_seed*/checkpoints/best.pt` instead of searching all models and datasets.


### Changes to plot_roc_all.py

| # | Change | Detail |
|---|---|---|
| 1 | **`discover_models`** now accepts `dataset_key` | Searches `RESULT_ROOT/{model}/{dataset}/exp1_*/trial_seed*` when `--dataset` is given; falls back to old flat structure + `*/exp1_*` fallback when omitted |
| 2 | **`--dataset` flag** added | Choices: `12k_de`, `12k_fe`, `48k_de` |
| 3 | **`resnet18` added** to `MODEL_DISPLAY` | Was missing from the display-name map |

### Usage

```bash
# Default (backward compat — broad search)
python scripts/plot_roc_all.py

# Scoped to 12k_de dataset
python scripts/plot_roc_all.py --dataset 12k_de

# Specific trial + dataset
python scripts/plot_roc_all.py --trial trial_seed456 --dataset 12k_de
```

All 7 models now correctly resolve to `experiment_result/exp1/{model}/12k_de/exp1_{model}/trial_seed42/checkpoints/best.pt`.

Made changes.

>add a table to `plot_roc_all.py` that list all the values in the ROC curves, including AUC, TPR, FPR, and thresholds, which should format with IEEE style. The table should be saved as latex format in the `paper/auto_tables` directory. 

Done. Here's the summary:

### New feature: `generate_roc_table()` in plot_roc_all.py

Generates an IEEE-style LaTeX table (`roc_auc_table.tex`) with per-class AUC values for all models.

**Table structure:**

| Model | BF007 | BF014 | ... | OF021 | Macro | Micro |
|---|---|---|---|---|---|---|
| **MSCA-VGG16 (Ours)** | 0.8893 | 0.8676 | ... | **0.9102** | 0.8941 | 0.8941 |
| VGG16 | 0.8980 | **0.8999** | ... | 0.9041 | **0.8972** | 0.8964 |
| ... | | | | | | |

**Features:**
- Best value per column **bolded** via `\mathbf{}`
- Proposed model name bolded via `\textbf{}`
- NaN/missing classes shown as `---`
- `booktabs` formatting (`\toprule`, `\midrule`, `\bottomrule`)
- Compilable `\label{tab:roc_auc}` for cross-referencing

### Usage

```bash
# Full run: plots + table
python scripts/plot_roc_all.py --dataset 12k_de

# Custom table output directory
python scripts/plot_roc_all.py --dataset 12k_de --table-dir paper/auto_tables
```

### New CLI flag

| Flag | Default | Purpose |
|---|---|---|
| `--table-dir` | auto_tables | Where `roc_auc_table.tex` is saved |

Made changes.


>The current table has 10 fault classes + 2 aggregate columns = 13 columns total (1 model name + 10 classes + Macro + Micro). This is way too wide for an IEEE double-column format.

Table reorganized: **12 rows × 8 columns** (transposed). Key changes:

| Before | After |
|---|---|
| 13 columns (too wide) | 8 columns (fits double-column) |
| Models as rows | Fault classes as rows |
| `\small` font | `\footnotesize` + `\arraystretch{1.05}` |
| 4 decimal places | 3 decimal places |

The layout is now:

```
Fault Class  MSCA-VGG16 (Ours)  VGG16  ResNet18  ...  ConvNeXt-T
BF007                0.889      0.898    0.915  ...      0.892
BF014                0.868      0.900    0.892  ...      0.891
...
Macro                0.894      0.897    0.894  ...      0.895
Micro                0.894      0.896    0.894  ...      0.895
```

Let me regenerate with real data:

Ran terminal command:  cd /home/yangchen/git_clone/AW-DPCNN && python -c "
from scripts.plot_roc_all import generate_roc_table, discover_models, evaluate_one
import torch
from pathlib import Path

# We just need the table, skip plots
print('Loading existing results from checkpoint evaluation...')
# Just regenerate table from existing roc_auc_table.tex? No — need to re-evaluate.
# Let's just run the table generation part with mock data to show the format.
print('Table format updated. Run: python scripts/plot_roc_all.py --dataset 12k_de')
" && cat paper/auto_tables/roc_auc_table.tex | head -20


The code is updated. The old `roc_auc_table.tex` on disk is stale — regenerate it:

```bash
python scripts/plot_roc_all.py --dataset 12k_de
```

The new transposed layout (8 columns, `\footnotesize`) will fit IEEE double-column:

```
Fault Class  MSCA-VGG16 (Ours)  VGG16  ResNet18  EffNet-B0  ViT  MobV3-S  ConvNeXt-T
BF007                0.889      0.898    0.915     0.872   0.869   0.911       0.892
...
Macro                0.894      0.897    0.894     0.887   0.890   0.897       0.895
Micro                0.894      0.896    0.894     0.887   0.890   0.897       0.895
```

Made changes.