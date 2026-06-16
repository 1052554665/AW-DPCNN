>Add **noise robustness experiment**: Inject Gaussian white noise at SNR = [-5, 0, 5, 10, 15, 20] dB into test signals, report accuracy degradation curve for proposed method vs. baselines.

## Script: noise_robustness.py

### How it works
1. Loads a trained checkpoint
2. Injects **additive Gaussian white noise** into test images at the pixel level (before ImageNet normalization) at SNR ∈ {−5, 0, 5, 10, 15, 20} dB + clean
3. Evaluates accuracy, F1, and AUC at each SNR
4. Saves CSV, JSON, and publication-quality accuracy-vs-SNR plot

### Noise model
$$\text{SNR}_{\text{dB}} = 10 \log_{10}\left(\frac{P_{\text{signal}}}{P_{\text{noise}}}\right), \quad P_{\text{signal}} = \mathbb{E}[x^2]$$

Noise is injected after `ToTensor()` (pixel values in [0, 1]) and before `Normalize()`, so the model sees noisy inputs that simulate corrupted acoustic measurements.

### Usage

```bash
# Single model (requires a trained checkpoint)
python scripts/noise_robustness.py     --config configs/default.yaml     --exp-config experiments/exp1/mobilenetv3_small.yaml     --auto-checkpoint

# Batch mode — evaluates ALL models in exp1 with auto-found checkpoints
python scripts/noise_robustness.py \
    --config configs/default.yaml \
    --exp-dir experiments/exp1 \
    --auto-checkpoint

# Custom SNR range
python scripts/noise_robustness.py \
    --config configs/default.yaml \
    --exp-config experiments/exp1/vgg16.yaml \
    --checkpoint .../best.pt \
    --snr -10 -5 0 5 10 15 20
```

### Output
```
experiments/noise_robustness/
├── noise_robustness.csv       # model, snr, accuracy, f1, auc, loss
├── noise_robustness.json      # structured results
└── noise_robustness.png       # multi-model accuracy-vs-SNR plot
```

> ⚠ **Note**: Models must be trained first (with checkpoints saved). Use `--auto-checkpoint` to auto-discover `best.pt` from the latest run of each experiment config.

Made changes.