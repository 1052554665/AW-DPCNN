Refer to `scripts/hyperparameter_sensitivity.py` and implement a hyperparameter sensitivity analysis in paper.

# Hyperparameter Sensitivity
>Assess the sensitivity of the PCNN-based feature extraction to key hyperparameters and identify optimal settings for improved classification performance.

**Parameters to sweep**:
- γ (contrast amplification): {1, 2, 4, 8, 10, 20}
- N (PCNN iterations): {5, 8, 10, 15, 20}
- α_L, α_T: {0.0001, 0.001, 0.01}
**Report**: Accuracy vs. parameter value curves for top-2 critical parameters.


## Script: hyperparameter_sensitivity.py

### What it does
Sweeps AW-DPCNN hyperparameters and evaluates classification accuracy on re-fused test windows using a pre-trained checkpoint. No full dataset rebuilds needed.

### Parameters swept
| Parameter | Values | Default |
|---|---|---|
| γ (contrast amplification) | {1, 2, 4, 8, 10, 20} | 4 |
| N (PCNN iterations) | {5, 8, 10, 15, 20} | 20 |
| α_L = α_T (decay) | {0.0001, 0.001, 0.01} | 0.001 |