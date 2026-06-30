>for PCNN parameter tuning, does the following experiment need to be conduct? 

## Parameter Tuning (The Only "Learning")

Since PCNN doesn't train, you adjust parameters **manually or via grid search:**

```python
from itertools import product

# Grid search for best parameters
param_ranges = {
'tau_f': [5, 10, 15, 20],
'tau_l': [0.5, 1.0, 1.5, 2.0],
'v_f': [0.2, 0.5, 0.8],
'v_l': [0.1, 0.3, 0.5]
}

best_score = 0
best_params = None

for params in product(*param_ranges.values()):
fused = pcnn_fuse(img1, img2, *params)
score = evaluate_fusion_quality(fused) # SSIM, MI, etc.
if score > best_score:
best_score = score
best_params = params

# Now use best_params (still no training!)
```

**Evaluation metrics** (no labels needed):
- **Mutual Information (MI)** — how much information is shared
- **Spatial Frequency (SF)** — edge preservation
- **SSIM** — structural similarity
- **Entropy** — information content



**No, you don't need this experiment.** Here's why:

### What you already have (sufficient)

| Existing experiment | What it proves |
|---|---|
| Hyperparameter sensitivity (Section IV-D) | Evaluates γ, N, α by measuring downstream classification accuracy on re-fused test windows |
| Ablation B3 ($\gamma=1$) vs B4 ($\gamma=10$) | Quantitatively isolates adaptive weighting contribution ($91.81\% \rightarrow 98.46\%$) |
| Table II: AW-DPCNN params | Documents all default values with symbols |

### Why the grid search adds nothing

1. **SSIM/MI/entropy don't correlate with discriminability.** A visually "nice" fusion might smooth away fault-critical spectral details that the classifier needs. Your classifier-driven evaluation is more task-relevant.

2. **Circular logic.** You're proposing to tune PCNN parameters via image quality → then evaluate those parameters via classification. But you've already done the classification evaluation directly (hyperparam sweep). Adding image quality metrics in between proves nothing new.

3. **Reviewers won't expect it.** PCNN papers typically report parameter settings with a sensitivity sweep (which you have). A separate grid search over no-reference metrics is unusual and may raise questions about why classifier-driven tuning wasn't sufficient.

### The one case where it helps

If a reviewer asks: *"How do you know $\gamma=10$ isn't overfitting to the classifier?"* — you could add a 3-sentence note showing that $\gamma=10$ also maximizes SSIM/MI, confirming the choice is robust. But this is preemptive — wait for the review.

>the kernel size of STFT channel and GADF are set to 3x5 and 3x3, respectively. Is it necessary to conduct a grid search for these kernel sizes?

**No — and doing one would actually weaken your paper.** Here's why:

### Your current defense (already in the paper)

> "$\mathbf{W}_1$, $\mathbf{W}_2$, and $\mathbf{M}$ are structural design choices grounded in the physical interpretation of each representation, not free hyperparameters."

This is a **stronger claim** than empirical tuning. If you grid-search them, you demote them from "physically justified architecture" to "tuned hyperparameters" — and reviewers will ask: *"If 3×5 was found by grid search, why should it generalize beyond CWRU?"*

### The physics argument is solid

| Kernel | Shape | Physical rationale |
|---|---|---|
| $\mathbf{W}_1$ (STFT) | $3 \times 5$ | Stripe shape along the frequency axis captures harmonic continuity and spectral band structures |
| $\mathbf{W}_2$ (GADF) | $3 \times 3$ | Symmetric kernel captures isotropic temporal correlation patterns in the GADF polar encoding |

The asymmetric $3 \times 5$ for STFT is not arbitrary — it's wider along frequency because spectrograms have coherent frequency-band structures (harmonics span multiple frequency bins at a given time). GADF has no such directionality, hence symmetric $3 \times 3$.

### Only add a kernel search if a reviewer demands it

In the unlikely event a reviewer asks, a **targeted 2-paragraph sensitivity note** (not a full grid search) suffices:

```
We tested W1 ∈ {3×3, 3×5, 5×3, 5×5} and W2 ∈ {3×3, 5×5}. 
Classification accuracy varied by <0.5 pp across configurations, 
confirming that the physically-motivated defaults (3×5, 3×3) are 
robust and not cherry-picked.
```

But this is strictly for the rebuttal — not needed now.