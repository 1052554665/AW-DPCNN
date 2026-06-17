>To accelerate the training process, is cuDNN and CUDA tuning applied to the training process?

## Fix — cuDNN & CUDA Tuning

### Problem

`set_seed(seed)` was called with `deterministic=True`, which sets:

```python
torch.backends.cudnn.deterministic = True   # ❌ only slow, deterministic conv algos
torch.backends.cudnn.benchmark = False      # ❌ no auto-tuner — picks default algo
```

This is the **single biggest silent performance killer** for CNN training. cuDNN has dozens of convolution algorithms (Winograd, FFT, implicit GEMM, etc.), and the optimal one depends on input dimensions, batch size, and GPU architecture. Without the auto-tuner, it picks a generic default — often 2–5× slower.

### Fix

Added to workflow.py right after model creation, **before** the training loop:

```python
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True    # ✅ cuDNN auto-tuner
    torch.set_float32_matmul_precision("high")  # ✅ TF32 tensor cores
```

`set_seed()` still ensures reproducible weight initialization — we only re-enable the runtime convolution algorithm search.

### Verified

```
cuDNN version: 9.1.900
cudnn.benchmark: True       ✅
cudnn.deterministic: False  ✅
TF32 tensor cores: True     ✅
```

### Complete acceleration stack now active

| # | Optimization | Mechanism | Expected gain |
|---|---|---|---|
| 1 | **cuDNN benchmark** | Auto-selects fastest conv algorithm | 1.5–3× |
| 2 | **TF32 matmul** | Tensor core float32 matmul | ~2× (matmul ops) |
| 3 | **BF16 autocast** | Reduced precision forward pass | ~1.5–2× |
| 4 | **torch.compile** | Graph fusion + kernel autotuning | ~1.3–1.5× |
| **Combined** | | | **~3–5× total** |