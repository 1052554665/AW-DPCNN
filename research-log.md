The ablation experiments are implemented as follows, refer to the `scripts/build_ablation_datasets.py` script for dataset preparation and the `scripts/run_ablation_experiments.py` script for running the experiments. Revise the paper `paper/tim.tex` accordingly.

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

# Ablation: Component Decomposition — Implementation Details

### 1. New script: run_ablation_experiments.py

Runs all 9 experiments (B0–B8) with one command:

```bash
python scripts/run_ablation_experiments.py          # full 30-epoch run
python scripts/run_ablation_experiments.py --dry-run  # list only
python scripts/run_ablation_experiments.py --exp-ids B0,B4,B8  # subset
```

| Feature | Detail |
|---------|--------|
| B0–B4 | VGG16 classifier with different input representations |
| B5–B8 | MSCA-VGG16 variants (MS/CA/EH flags) with full AW-DPCNN |
| Output | Per-run checkpoints + `ablation_summary_*.csv` |

### 2. Revised tim.tex — Ablation Study section

**Replaced** the old two-table design (E1–E4 + M1–M5) with a **unified B0–B8 design**:

| Before | After |
|--------|-------|
| Two separate tables with inconsistent naming | One unified table (`tab:ablation_unified`) with systematic B0–B8 progression |
| E1–E4: "Comparison of Fusion Strategies" | B0–B4: Fusion-level ablation (VGG16 classifier) |
| M1–M5: "Ablation on VGG16-Based Models" | B5–B8: Classifier-level ablation (MSCA-VGG16 variants) |
| Fragmented analysis | Three-part analysis: fusion, classifier, overall decomposition |
| Old labels: E1/E2/E3/E4, M1/M2/M3/M4/M5 | Unified labels: B0 through B8, cross-referenced throughout |

The t-SNE figures are preserved and their captions now reference the B0/B1/B2/B4 experiment IDs for consistency.

### 3. Placeholder values (`xx.xx`)

The B3–B7 rows contain placeholder values. Fill them by running:
```bash
python scripts/run_ablation_experiments.py
```
Then copy values from `experiments/ablation_results/ablation_summary_*.csv` into the table.

Made changes.