>modify the dataset building scripts to replace Mel spectrogram with STFT (Short-Time Fourier Transform) representation, while keeping GADF and AW-DPCNN fusion.

## Summary: Mel → STFT Replacement for Dataset Building

### Core change
Added `generate_stft_image()` — a linear STFT spectrogram generator — alongside the existing `generate_mel_image()` in build_cwru_dataset.py. Added a `TF_GENERATORS` registry dict for dispatching between `'mel'` and `'stft'`.

### Files modified

| File | Changes |
|------|---------|
| build_cwru_dataset.py | Added `generate_stft_image()`, `TF_GENERATORS` registry, `--tf-method` CLI (default: `stft`). Updated `process_one_window`, `_collect_tasks`, `_append_window_tasks` to dispatch per method. Backward-compatible: old callers without `tf_method` default to `'mel'`. |
| build_cwru_12k_de.py | Added `--tf-method` (default: `stft`), updated imports, task tuples, and `_build_tasks_and_metadata`. |
| build_cwru_12k_fe.py | Same as 12k_de. |
| build_cwru_48k_de.py | Same as 12k_de. |
| build_ablation_datasets.py | Replaced `_make_mel` with `_make_tf` that dispatches via `TF_METHOD` global (set by `--tf-method`). Kept variant keys (`mel_only`, etc.) backward-compatible. |
| hyperparameter_sensitivity.py | Updated `_precompute_mel_gadf` → accepts `tf_method` param. Added `--tf-method` CLI (default: `stft`). |

### Key behaviors
- **Default is now STFT** (`--tf-method stft`) — all dataset builders produce STFT+GADF fused images by default
- **Backward compatible** — `--tf-method mel` restores the original Mel spectrogram behavior
- **STFT bypasses the Mel filterbank** — uses `librosa.stft` + `amplitude_to_db`, preserving linear frequency resolution
- **No API break** — `process_one_window` unpacks the task tuple with backward-compatible defaults

### Usage examples
```bash
# Build STFT+GADF dataset (default)
python scripts/build_cwru_12k_de.py --output-dir ./datasets/cwru_12k_de_stft

# Build Mel+GADF dataset (original behavior)
python scripts/build_cwru_12k_de.py --tf-method mel --output-dir ./datasets/cwru_12k_de_mel

# Ablation study with STFT
python scripts/build_ablation_datasets.py --tf-method stft --workers 16
```

Made changes.

>since the directory of `test_metrics` has been changed, and there are three different datasets, three trail seeds, the `plot_model_comparison.py` script should be updated to include a new command line argument `--dataset` to specify dataset and `--trial` to specify trial seed.

### Changes

| Aspect | Before | After |
|--------|--------|-------|
| **Directory navigation** | `exp1/{model}/exp1_*/trial/results/` | `exp1/{model}/{dataset}/exp1_*/trial/results/` |
| **Trial selection** | Hardcoded `TRIAL = "trial_seed456"` | `--trial` CLI argument (default: `trial_seed42`) |
| **Dataset selection** | None | `--dataset` CLI argument (choices: `12k_de`, `12k_fe`, `48k_de`) |
| **Output filename** | `model_comparison_grouped.png` | `model_comparison_{dataset}_{trial}.png` |
| **Plot title** | (commented out) | `Model Comparison — {dataset_label} ({trial})` |

### Usage

```bash
# All three datasets, each trial
python scripts/plot_model_comparison.py --dataset 12k_de --trial trial_seed42
python scripts/plot_model_comparison.py --dataset 12k_fe --trial trial_seed42
python scripts/plot_model_comparison.py --dataset 48k_de --trial trial_seed42

# Other trials
python scripts/plot_model_comparison.py --dataset 12k_de --trial trial_seed123
python scripts/plot_model_comparison.py --dataset 12k_de --trial trial_seed456
```

Made changes.

>since the directory of `best.pt` has been changed, and there are three different datasets, three trail seeds, the `plot_tsne_all.py` script should be updated to include a new command line argument `--dataset` to specify dataset and `--trial` to specify trial seed.

### Changes

| Aspect | Before | After |
|--------|--------|-------|
| **Checkpoint discovery** | `exp1/{model}/exp1_*/trial/` | `exp1/{model}/{dataset}/exp1_*/trial/` |
| **`find_model_runs()`** | `find_model_runs(trial)` | `find_model_runs(dataset, trial)` |
| **`generate_model_tsne()`** | `generate_model_tsne(device, max_samples, trial)` | `generate_model_tsne(device, max_samples, dataset, trial)` |
| **Output filename** | `tsne_{name}.png` | `tsne_{name}_{dataset}_{trial}.png` |
| **Plot title** | `{name} — Feature t-SNE` | `{name} — {dataset} ({trial})` |
| **CLI** | `--trial` only | `--dataset` (choices: `12k_de`, `12k_fe`, `48k_de`) + `--trial` |

### Usage

```bash
# Model t-SNE for each dataset
python scripts/plot_tsne_all.py --mode models --dataset 12k_de --trial trial_seed42
python scripts/plot_tsne_all.py --mode models --dataset 12k_fe --trial trial_seed42
python scripts/plot_tsne_all.py --mode models --dataset 48k_de --trial trial_seed42

# Other trials
python scripts/plot_tsne_all.py --mode models --dataset 12k_de --trial trial_seed123

# Raw input + model t-SNE
python scripts/plot_tsne_all.py --mode all --dataset 12k_de --data-dir ./datasets/cwru_12k_de/test --trial trial_seed42
```

Made changes.

>since the mel spectrogram has been replaced with STFT, the `build_ablation_datasets.py` script should be updated, and the output structure is like the following:

    datasets/ablation/
        stft_only/         train/{BF007,...,Normal}/  val/  test/  metadata.csv
        gadf_only/        ...
        concat/           ...
        awdpcnn_gamma1/   ...
        awdpcnn_full/     ...



| Item | Before | After |
|------|--------|-------|
| Variant key | `mel_only` | `stft_only` |
| Output directory | `datasets/ablation/mel_only/` | `datasets/ablation/stft_only/` |
| Docstring B0 | "Mel spectrogram only" | "STFT spectrogram only" |
| Docstring B2 | "Mel + GADF pixel‑wise average" | "STFT + GADF pixel‑wise average" |
| Config B0_mel_only.yaml | `root_dir: ./datasets/ablation/mel_only` | `root_dir: ./datasets/ablation/stft_only` |

The `--tf-method mel` flag still restores the original Mel behavior if needed, with the dataset written to `stft_only/` (the variant key is just a directory label — the actual representation is controlled by `--tf-method`).

Made changes.


>Where is "clean" plotted?

### Where is "clean" plotted?
**Not at 0 dB.** "Clean" means no noise — conceptually infinite SNR. Plotting clean at 0 dB would be misleading (0 dB = signal power equals noise power). Instead, clean is now plotted at `max_real_SNR + 5 dB` (e.g., if your max SNR is 30, clean appears at 35 on the x-axis), clearly separated from the real SNR levels.

Made changes.

>according to the new dataset, update the table.

AW-DPCNN CWRU 12kHz Drive End Dataset Builder (10‑class)
════════════════════════════════════════════════════════════
  Fault dir      : raw-data/CWRU-dataset/12k_Drive_End_Bearing_Fault_Data
  Normal dir     : raw-data/CWRU-dataset/Normal
  Sample rate    : 12000 Hz
  Mat key        : DE_time
  OR position    : @6
  Window / Hop   : 2048 / 1024
  TF method      : stft
  File split     : 60,20,20  (seed=42)
  Workers        : 32
  Dry run        : False
════════════════════════════════════════════════════════════

Collected 40 .mat files → 10 classes:
  ✓ BF007      4 file(s)
  ✓ BF014      4 file(s)
  ✓ BF021      4 file(s)
  ✓ IF007      4 file(s)
  ✓ IF014      4 file(s)
  ✓ IF021      4 file(s)
  ✓ OF007      4 file(s)
  ✓ OF014      4 file(s)
  ✓ OF021      4 file(s)
  ✓ Normal     4 file(s)

File‑level split (seed=42):
  Class     Total   Train     Val    Test
  ----------------------------------------
  BF007         4       2       1       1
  BF014         4       2       1       1
  BF021         4       2       1       1
  IF007         4       2       1       1
  IF014         4       2       1       1
  IF021         4       2       1       1
  OF007         4       2       1       1
  OF014         4       2       1       1
  OF021         4       2       1       1
  Normal        4       2       1       1
  ----------------------------------------
  TOTAL        40      20      10      10

[INFO] Total fusion tasks: 5886
Building fused dataset: 100%|██████████████████████████████████| 5886/5886 [00:20<00:00, 283.60it/s]

✅ Done — 5886/5886 images written to ./datasets/cwru_12k_de
📋 Metadata saved to ./datasets/cwru_12k_de/metadata.csv (5886 rows)

────────────────────────────────────────────────────────────
Distribution verification
────────────────────────────────────────────────────────────
  train:   2824 samples  {'BF007': 234, 'BF014': 236, 'BF021': 235, 'IF007': 236, 'IF014': 234, 'IF021': 235, 'Normal': 708, 'OF007': 236, 'OF014': 234, 'OF021': 236}
  val:   1530 samples  {'BF007': 117, 'BF014': 117, 'BF021': 118, 'IF007': 117, 'IF014': 117, 'IF021': 117, 'Normal': 473, 'OF007': 118, 'OF014': 118, 'OF021': 118}
  test:   1532 samples  {'BF007': 118, 'BF014': 118, 'BF021': 118, 'IF007': 119, 'IF014': 117, 'IF021': 118, 'Normal': 471, 'OF007': 117, 'OF014': 118, 'OF021': 118}
  JS(train, val) = 0.002123  [OK]
  JS(train, test) = 0.002005  [OK]
  JS(val, test) = 0.000006  [OK]

AW-DPCNN CWRU 12kHz Fan End Dataset Builder (10‑class)
════════════════════════════════════════════════════════════
  Fault dir      : raw-data/CWRU-dataset/12k_Fan_End_Bearing_Fault_Data
  Normal dir     : raw-data/CWRU-dataset/Normal
  Sample rate    : 12000 Hz
  Mat key        : FE_time
  OR position    : @6
  Window / Hop   : 2048 / 1024
  TF method      : stft
  File split     : 60,20,20  (seed=42)
  Workers        : 32
  Dry run        : False
════════════════════════════════════════════════════════════

Collected 34 .mat files → 10 classes:
  ✓ BF007      4 file(s)
  ✓ BF014      4 file(s)
  ✓ BF021      4 file(s)
  ✓ IF007      4 file(s)
  ✓ IF014      4 file(s)
  ✓ IF021      4 file(s)
  ✓ OF007      4 file(s)
  ✓ OF014      1 file(s)
  ✓ OF021      1 file(s)
  ✓ Normal     4 file(s)

File‑level split (seed=42):
  Class     Total   Train     Val    Test
  ----------------------------------------
  BF007         4       2       1       1
  BF014         4       2       1       1
  BF021         4       2       1       1
  IF007         4       2       1       1
  IF014         4       2       1       1
  IF021         4       2       1       1
  OF007         4       2       1       1
  OF014         1       1       1       1
  OF021         1       1       1       1
  Normal        4       2       1       1
  ----------------------------------------
  TOTAL        34      18      10      10

[INFO] Total fusion tasks: 5155
Building fused dataset: 100%|██████████████████████████████████| 5155/5155 [00:18<00:00, 278.40it/s]

✅ Done — 5155/5155 images written to ./datasets/cwru_12k_fe
📋 Metadata saved to ./datasets/cwru_12k_fe/metadata.csv (5155 rows)

────────────────────────────────────────────────────────────
Distribution verification
────────────────────────────────────────────────────────────
  train:   2484 samples  {'BF007': 233, 'BF014': 235, 'BF021': 233, 'IF007': 234, 'IF014': 234, 'IF021': 234, 'Normal': 708, 'OF007': 235, 'OF014': 69, 'OF021': 69}
  val:   1337 samples  {'BF007': 117, 'BF014': 118, 'BF021': 117, 'IF007': 117, 'IF014': 117, 'IF021': 117, 'Normal': 473, 'OF007': 117, 'OF014': 22, 'OF021': 22}
  test:   1334 samples  {'BF007': 117, 'BF014': 117, 'BF021': 117, 'IF007': 117, 'IF014': 117, 'IF021': 117, 'Normal': 471, 'OF007': 117, 'OF014': 22, 'OF021': 22}
  JS(train, val) = 0.003737  [OK]
  JS(train, test) = 0.003679  [OK]
  JS(val, test) = 0.000001  [OK]


AW-DPCNN CWRU 48kHz Drive End Dataset Builder (10‑class)
════════════════════════════════════════════════════════════
  Fault dir      : raw-data/CWRU-dataset/48k_Drive_End_Bearing_Fault_Data
  Normal dir     : raw-data/CWRU-dataset/Normal
  Sample rate    : 48000 Hz
  Mat key        : DE_time
  OR position    : @6
  Window / Hop   : 8192 / 4096
  Mel params     : n_fft=4096, n_mels=128, fmax=20000
  TF method      : stft
  File split     : 60,20,20  (seed=42)
  Workers        : 32
  Dry run        : False
════════════════════════════════════════════════════════════

Collected 40 .mat files → 10 classes:
  ✓ BF007      4 file(s)
  ✓ BF014      4 file(s)
  ✓ BF021      4 file(s)
  ✓ IF007      4 file(s)
  ✓ IF014      4 file(s)
  ✓ IF021      4 file(s)
  ✓ OF007      4 file(s)
  ✓ OF014      4 file(s)
  ✓ OF021      4 file(s)
  ✓ Normal     4 file(s)

File‑level split (seed=42):
  Class     Total   Train     Val    Test
  ----------------------------------------
  BF007         4       2       1       1
  BF014         4       2       1       1
  BF021         4       2       1       1
  IF007         4       2       1       1
  IF014         4       2       1       1
  IF021         4       2       1       1
  OF007         4       2       1       1
  OF014         4       2       1       1
  OF021         4       2       1       1
  Normal        4       2       1       1
  ----------------------------------------
  TOTAL        40      20      10      10

[INFO] Total fusion tasks: 4058
Building fused dataset: 100%|██████████████████████████████████| 4058/4058 [00:16<00:00, 250.89it/s]

✅ Done — 4058/4058 images written to ./datasets/cwru_48k_de
📋 Metadata saved to ./datasets/cwru_48k_de/metadata.csv (4058 rows)

────────────────────────────────────────────────────────────
Distribution verification
────────────────────────────────────────────────────────────
  train:   2228 samples  {'BF007': 234, 'BF014': 234, 'BF021': 234, 'IF007': 234, 'IF014': 236, 'IF021': 235, 'Normal': 175, 'OF007': 235, 'OF014': 175, 'OF021': 236}
  val:    938 samples  {'BF007': 118, 'BF014': 59, 'BF021': 117, 'IF007': 58, 'IF014': 117, 'IF021': 118, 'Normal': 117, 'OF007': 58, 'OF014': 117, 'OF021': 59}
  test:    892 samples  {'BF007': 58, 'BF014': 117, 'BF021': 58, 'IF007': 117, 'IF014': 14, 'IF021': 58, 'Normal': 117, 'OF007': 117, 'OF014': 118, 'OF021': 118}
  JS(train, val) = 0.018164  [OK]
  JS(train, test) = 0.035615  [OK]
  JS(val, test) = 0.063977  [WARN]

>for Representation Comparison Table, bold the best performance for each metric (accuracy, precision, recall, F1-score) across the three representations (Mel, STFT, GADF). Use LaTeX `\textbf{}` to bold the values in the table.

>since the ablation study result output has been changed, the `ablation_table.py` script should be updated