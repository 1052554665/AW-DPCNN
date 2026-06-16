# Ablation: Component Decomposition
> To isolate and quantify the contribution of each component (MS, CA, EH) in the AW-DPCNN framework. Conducting ablation studies by systematically removing or replacing components and evaluating performance impacts.

**Current**: MS, CA, EH on VGG16

**Expand to include AW-DPCNN components**:

| Exp | AW-DPCNN | MS | CA | EH | Expected |
|---|---|---|---|---|---|
| B0 | ✗ (single Mel) | ✗ | ✗ | ✗ | Lower bound |
| B1 | ✗ (single GADF) | ✗ | ✗ | ✗ | GADF > Mel |
| B2 | ✗ (Concat) | ✗ | ✗ | ✗ | Baseline fusion |
| B3 | ✓ (γ=1, fixed weight) | ✗ | ✗ | ✗ | PCNN without adaptive |
| B4 | ✓ (full AW-DPCNN) | ✗ | ✗ | ✗ | Fusion contribution |
| B5 | ✓ | ✓ | ✗ | ✗ | +MS contribution |
| B6 | ✓ | ✗ | ✓ | ✗ | +CA contribution |
| B7 | ✓ | ✗ | ✗ | ✓ | +EH contribution |
| B8 | ✓ | ✓ | ✓ | ✓ | **Full model** |

----

## Ablation: Component Decomposition — Complete Setup

### Dataset Variants (B0–B3)

| Exp | Dataset | Builder |
|---|---|---|
| B0 | `datasets/ablation/mel_only/` | Mel pseudo-color only |
| B1 | `datasets/ablation/gadf_only/` | GADF pseudo-color only |
| B2 | `datasets/ablation/concat/` | Mel+GADF pixel-wise average |
| B3 | `datasets/ablation/awdpcnn_gamma1/` | AW-DPCNN with γ=1 |
| B4–B8 | transformer-five | Full AW-DPCNN (γ=4) |

Build with: `python build_ablation_datasets.py --workers 16`

### Model Variants (B5–B8)

| Exp | Classifier | Params | MS | CA | EH |
|---|---|---|---|---|---|
| B0–B4 | VGG16 | 134.3M | — | — | — |
| B5 | MSCA-VGG16 | 26.5M | ✓ | ✗ | ✗ |
| B6 | MSCA-VGG16 | 17.7M | ✗ | ✓ | ✗ |
| B7 | MSCA-VGG16 | 15.5M | ✗ | ✗ | ✓ |
| B8 | MSCA-VGG16 | 26.8M | ✓ | ✓ | ✓ |

### Running the Ablation

```bash
# 1. Build ablation datasets (B0–B3)
python scripts/build_ablation_datasets.py --workers 16

# 2. Run all 9 experiments
for exp in experiments/ablation/B*.yaml; do
    python scripts/train.py --config configs/default.yaml --exp-config "$exp"
done

# 3. Compile results table
python scripts/noise_robustness.py --config configs/default.yaml \
    --exp-dir experiments/ablation --auto-checkpoint
```


# put results in a unified folder

All experiment outputs now go to a unified folder:

```
experiments/
├── experiment_result/          ← NEW unified output
│   ├── exp1/
│   │   ├── vgg16_20260616_.../
│   │   ├── mobilenetv3_small_20260616_.../
│   │   └── ...
│   └── ablation/
│       ├── B0_mel_only_20260616_.../
│       ├── B1_gadf_only_20260616_.../
│       └── ...
├── exp1/                       ← configs only (no more output)
│   └── *.yaml
└── ablation/                   ← configs only (no more output)
    └── B*.yaml
```