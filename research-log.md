>rewrite the script `hyperparameter_sensitivity.py`, the dataset is `datasets/cwru_de`.


```bash
# Full sweep
python scripts/hyperparameter_sensitivity.py \
    --exp-config experiments/exp1/MSCA_VGG16.yaml \
    --auto-checkpoint

# Single parameter
python scripts/hyperparameter_sensitivity.py \
    --exp-config experiments/exp1/MSCA_VGG16.yaml \
    --auto-checkpoint --sweep gamma
```

### What was fixed

| Issue | Fix |
|---|---|
| `model.name: baseline` unsupported | `num_classes` read from config, not hardcoded |
| `_orig_mod.` prefix in checkpoint | Stripped before `load_state_dict()` |
| Checkpoint not found (1-level search) | Recursive glob up to 4 levels |
| `.wav` data source unavailable | Reads CWRU `.mat` files via class mapping |
| Wrong mel params for CWRU 12k | Uses correct `n_fft=1024, fmax=6000` |
| No test-split isolation | Derives test source files from test PNGs |

### ⚠️ 0% accuracy investigation

The test run showed 0% across all γ values. This likely means the re-fused images don't match the training distribution. Check:
1. Was cwru_de built with the **same** build_cwru_dataset.py that the script imports from?
2. Does the MSCA_VGG16 checkpoint produce non-zero accuracy on the pre-built test PNGs directly? (Quick sanity check: run `evaluate.py` on that checkpoint)
3. Are the `generate_mel_image` / `generate_gadf_image` defaults in build_cwru_dataset.py the same as what was used to build cwru_de?

git commit -m "fix: hyperparameter_sensitivity.py for cwru_de dataset"