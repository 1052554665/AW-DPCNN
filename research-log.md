>add a `--outout` argument to specify the output directory

>how to use the MIMII dataset for fault diagnosis within this project.

# Usage Modes for MIMII

Train and evaluate entirely on MIMII — proves the method generalizes beyond bearing vibration to acoustic signals.


>how to select proper classes for MIMII.

## MIMII Class Selection Analysis

### Data Reality

Every noise level contains **identical recordings** — same machines, same faults, just different background SNR:

```
         fan      pump     slider   valve     normal : abnormal
-6_dB    4075/1475  3749/456  3204/890  3691/479   ~4:1 imbalance
 0_dB    4075/1475  3749/456  3204/890  3691/479   identical recordings
 6_dB    4075/1475  3749/456  3204/890  3691/479   identical recordings
```

⚠️ **Critical**: If you mix noise levels without id-level splitting, the same physical recording leaks across train/test.

---

### Five Class Schemes — Ranked by Suitability

| # | Scheme | Classes | Pros | Cons | Recommendation |
|---|---|---|---|---|---|
| **A** | Machine × Condition | 8 | Most granular | 4:1 imbalance, 8-way hard for small model | ⭐⭐ |
| **B** | Machine type only | 4 | Balanced (~14k each) | Loses fault info — not "diagnosis" | ⭐ |
| **C** | Binary per machine | 2×4 tasks | Matches MIMII benchmark | 4 separate models, messy reporting | ⭐⭐⭐ |
| **D** | **Normal vs Abnormal** | **2** | Universal "health" concept, transfers to any domain | Coarse, 4:1 imbalance | ⭐⭐⭐⭐ |
| **E** | **Normal vs Abnormal (balanced)** | **2** | Like D but subsample normal to 1500/machine | Small dataset (~12k total) | ⭐⭐⭐⭐⭐ |

---

### My Recommendation: **Scheme D + Class-Weighted Loss**

For a paper on **method generalization** (not MIMII-specific benchmarking):

1. **2 classes**: `normal` and `abnormal` — the concept of "healthy vs faulty" transcends domains (bearings, transformers, pumps, fans, valves all share this)
2. **Handle imbalance** via `class_weighting: true` (already in your default.yaml)
3. **Cross-domain story**: "AW-DPCNN trained on CWRU bearing faults achieves X% on MIMII abnormal detection — proving the fused representation captures universal fault signatures"

This gives you a clean narrative:

```
CWRU (10-class bearing)  →  MIMII (2-class acoustic)
         ↓                            ↓
   "Fault type + severity"      "Healthy vs faulty"
                                  ← tests universal representation
```

### What About Noise Levels?

Best experimental design for your paper:

| Experiment | Train on | Test on | Purpose |
|---|---|---|---|
| **Baseline** | MIMII 0dB | MIMII 0dB | In-domain performance |
| **Noise robustness** | MIMII 0dB | MIMII -6dB, 6dB | Generalization across SNR |
| **Cross-domain** | CWRU 12k DE | MIMII 0dB | Domain shift (vibration→acoustic) |

For the noise-robustness test, you'd need to build datasets **per noise level** with id-consistent splits (same id_XX always in same split across noise levels).



>for MIMII (located in `raw-data/MIMII`), create scripts to build datasets from raw audio files, and create new configuration files for MIMII dataset, which is similar to `exp1/MSCA_VGG16.yaml`, but the dataset is changed to generated datasets. Making three datasets for MIMII: `mimii_0dB`, `mimii_6dB`, and `mimii_12dB` (the noise level is added to the dataset name). The datasets are built from the raw audio files in `raw-data/MIMII` and the output datasets are saved in `datasets/mimii_0dB`, `datasets/mimii_6dB`, and `datasets/mimii_12dB`. The configuration files are saved in `experiments/exp2`

## Created / Modified

| File | What |
|---|---|
| build_mimii.py | **Rewritten** — `--noise-level` (0_dB/6_dB/-6_dB/all), `--class-scheme` (binary/machine_condition), lazy audio loading, id-level split |
| dataset_registry.py | Added `mimii_0dB`, `mimii_6dB`, `mimii_-6dB` entries |
| mimii_0dB.yaml | Binary config, 30 epochs, class_weighting |
| mimii_6dB.yaml | Same as 0dB |
| mimii_-6dB.yaml | 35 epochs, higher weight_decay (5e-3), dropout 0.6 |

## Build Commands

```bash
python scripts/build_mimii.py --noise-level 0_dB  --output-dir datasets/mimii_0dB
python scripts/build_mimii.py --noise-level 6_dB  --output-dir datasets/mimii_6dB
python scripts/build_mimii.py --noise-level='-6_dB' --output-dir datasets/mimii_-6dB
```

## Train Commands

```bash
# Per noise level
python scripts/run_repeated_trials.py \
    --config configs/default.yaml \
    --exp-config experiments/exp2/mimii_0dB.yaml \
    --output experiments/experiment_result/exp2 \
    --dataset mimii_0dB --num-runs 3

python scripts/run_repeated_trials.py \
    --config configs/default.yaml \
    --exp-config experiments/exp2/mimii_6dB.yaml \
    --output experiments/experiment_result/exp2 \
    --dataset mimii_6dB --num-runs 3

python scripts/run_repeated_trials.py \
    --config configs/default.yaml \
    --exp-config experiments/exp2/mimii_-6dB.yaml \
    --output experiments/experiment_result/exp2 \
    --dataset mimii_-6dB --num-runs 3
```

## Experimental Design

| Experiment | Train | Test | Purpose |
|---|---|---|---|
| Baseline | mimii_0dB | mimii_0dB | In-domain binary acoustic |
| Noise robustness | mimii_0dB | mimii_6dB, mimii_-6dB | SNR generalization |
| Cross-domain | CWRU 12k DE | mimii_0dB | Vibration → acoustic transfer |

Made changes.

>how to reduce the number of generated images


Currently, for each 10-second audio file at 16kHz (160,000 samples), with win_len=8192 and hop_len=4096, we get:

(160000 - 8192) / 4096 + 1 ≈ 38 windows per file

With 18,019 files per noise level, that's ~680k images per noise level. The user wants to reduce this.

Options:

- Increase `hop_len` — fewer overlapping windows (e.g., `hop_len=8192` gives ~19 windows/file, `hop_len=16384` gives ~10)
- Decrease `win_len` — smaller windows mean more windows (don't do this)
- Use fewer windows per file — process only a subset (e.g., take every Nth window)
- Reduce per file: use --win-len 0 to get one image per file (the whole 10s signal as one image)
- Add a --max-windows-per-file argument to cap windows per file
- Use the middle portion — take only the middle N seconds of each file


Added `--max-windows-per-file N`. Here's how to control image count:

```bash
# Default: ~38 windows/file → ~680k images/noise-level
python scripts/build_mimii.py --noise-level 0_dB

# 10 windows per file → ~180k images  
python scripts/build_mimii.py --noise-level 0_dB --max-windows-per-file 10

# 5 windows per file → ~90k images
python scripts/build_mimii.py --noise-level 0_dB --max-windows-per-file 5

# 1 image per file → ~18k images (use --win-len 0)
python scripts/build_mimii.py --noise-level 0_dB --win-len 0

# Combine with longer hop for fewer windows + cap
python scripts/build_mimii.py --noise-level 0_dB --hop-len 16384 --max-windows-per-file 5
```

| `--max-windows-per-file` | `--hop-len` | Windows/file | Images/18k files |
|---|---|---|---|
| 0 (unlimited) | 4096 | ~38 | ~680k |
| 10 | 4096 | 10 | ~180k |
| 5 | 4096 | 5 | ~90k |
| 0 (unlimited) | 16384 | ~10 | ~180k |
| `--win-len 0` | — | 1 | ~18k |

Made changes.