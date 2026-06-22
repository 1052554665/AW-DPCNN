Refer to `scripts/run_repeated_trials.py`, report **mean ± std** for all metrics in paper.

## Changes: Mean ± Std Reporting

### 1. Methodology note added (Section 5.2)
```latex
To ensure the statistical reliability of the reported results and to account 
for performance variance due to random initialization, all key experiments 
are repeated over three independent trials with different random seeds 
(42, 123, 456) using the script run_repeated_trials.py. The aggregated 
results are reported as mean ± standard deviation across the three trials.
```

### 2. Three tables updated to `mean ± std` format

| Table | Format |
|-------|--------|
| **Backbone Comparison** (`tab:network_comparison`) | `$87.45_{\pm x.xx}$` for all 6 backbones × 6 metrics |
| **Ablation Study** (`tab:ablation_unified`) | `$80.82_{\pm x.xx}$` for B0–B8 × 4 metrics |
| **CWRU Validation** (`tab:cwru_ablation`) | `$99.67_{\pm x.xx}$` for A1–A5 × 5 metrics |

### 3. Abstract updated
```
achieves an accuracy of $87.45 \pm x.xx\%$ (mean ± std over 3 independent trials)
```

### To fill placeholder std values

Run the repeated trials script:
```bash
# For backbone comparison (all models):
python scripts/run_repeated_trials.py \
    --config configs/default.yaml \
    --exp-dir experiments/exp1 \
    --num-runs 3

# For ablation experiments:
python scripts/run_ablation_experiments.py  # already reports per-run results
```

Then copy the `± std` values from the aggregated CSVs into the placeholder `x.xx` positions in each table.