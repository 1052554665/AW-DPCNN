# Hyperparameter Sensitivity
>Assess the sensitivity of the PCNN-based feature extraction to key hyperparameters and identify optimal settings for improved classification performance.

**Parameters to sweep**:
- γ (contrast amplification): {1, 2, 4, 8, 10, 20}
- N (PCNN iterations): {5, 8, 10, 15, 20}
- α_L, α_T: {0.0001, 0.001, 0.01}
**Report**: Accuracy vs. parameter value curves for top-2 critical parameters.

---

## Script: hyperparameter_sensitivity.py

### What it does
Sweeps AW-DPCNN hyperparameters and evaluates classification accuracy on re-fused test windows using a pre-trained checkpoint. No full dataset rebuilds needed.

### Parameters swept
| Parameter | Values | Default |
|---|---|---|
| γ (contrast amplification) | {1, 2, 4, 8, 10, 20} | 4 |
| N (PCNN iterations) | {5, 8, 10, 15, 20} | 20 |
| α_L = α_T (decay) | {0.0001, 0.001, 0.01} | 0.001 |

### Usage

```bash
# Full sweep (all 3 parameters, with auto-discovered checkpoint)
python scripts/hyperparameter_sensitivity.py \
    --config configs/default.yaml \
    --exp-config experiments/ablation/B8_full_model.yaml \
    --auto-checkpoint

# Single parameter sweep with more samples
python scripts/hyperparameter_sensitivity.py \
    --config configs/default.yaml \
    --checkpoint PATH/TO/best.pt \
    --sweep gamma --max-samples 1000
```

### Output
```
experiments/hyperparameter_sensitivity/
├── sensitivity_gamma.csv / .png
├── sensitivity_N.csv / .png
├── sensitivity_alpha_LT.csv / .png
└── sensitivity_summary.json
```

### Also updated
build_CWRU_dataset.py — `aw_dpcnn_single_channel()` and `aw_dpcnn_fusion_color()` now accept `alpha_L` and `alpha_T` as parameters (was hardcoded).

> ⚠ For meaningful sensitivity results, use a checkpoint trained on **full AW-DPCNN fused data** (B4 or B8), not single-representation models (B0/B1).

Made changes.