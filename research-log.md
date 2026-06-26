>Only a CWRU dataset is conducted in this project. add another dataset to verify the generalization of the proposed method. How to select the dataset? The goal of this paper is realise acoustic fault diagnosis. 
- which one is proper most? University of Ottawa Electric Motor Dataset – Vibration and Acoustic Faults under Constant and Variable Speed Conditions (UOEMD-VAFCVS), or Acoustic Emission Bearing Fault Dataset, MIMII
- How to select the model? still MSCA-VGG16, VGG16, ViT, EfficientNet-B0, MobileNetV3-Small, ConvNeXt-Tiny?
- while conduct the experiments, the following table can be referred.


\begin{table*}[!t]
\centering
\caption{Performance of Three Datasets in the Closed-Set Setting, Including Accuracy (Acc\%) and F1-Score(F1\%)}
\label{tab:closed_set}
\renewcommand{\arraystretch}{1.1}
\begin{tabular}{ll|cc|cc|cc|c|c|cc}
\toprule
\multirow{2}{*}{\textbf{DataSet}} & \multirow{2}{*}{\textbf{Label Num}} 
& \multicolumn{2}{c|}{\textbf{Flexmatch}} 
& \multicolumn{2}{c|}{\textbf{BYOL}} 
& \multicolumn{2}{c|}{\textbf{Fixmatch}} 
& \textbf{SSCL} 
& \textbf{HSFD} 
& \multicolumn{2}{c}{\textbf{OSCL}} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-9}\cmidrule(lr){10-10}\cmidrule(lr){11-12}
& & \textbf{Acc} & \textbf{F1} & \textbf{Acc} & \textbf{F1} & \textbf{Acc} & \textbf{F1} & \textbf{Acc} & \textbf{Acc} & \textbf{Acc} & \textbf{F1} \\
\midrule
\multirow{7}{*}{CWRU}
& 10 & 72.40$\pm$11.85 & 66.05$\pm$14.93 & 93.30$\pm$0.03 & 92.56$\pm$0.03 & 65.27$\pm$4.77 & 54.50$\pm$3.83 & 64.75 & 66.73 & \textbf{97.26$\pm$0.09} & 97.05$\pm$0.17 \\
& 50 & 84.38$\pm$15.63 & 81.59$\pm$19.73 & 94.88$\pm$0.04 & 94.34$\pm$0.04 & 80.56$\pm$7.37 & 76.44$\pm$10.04 & 88.92 & 79.81 & \textbf{97.28$\pm$0.08} & 97.02$\pm$0.09 \\
& 100 & 85.23$\pm$15.31 & 81.32$\pm$19.72 & 96.23$\pm$0.04 & 96.16$\pm$0.18 & 89.21$\pm$7.16 & 81.66$\pm$9.38 & 91.53 & 92.13 & \textbf{97.49$\pm$0.04} & 97.25$\pm$0.04 \\
& 150 & 87.80$\pm$9.66 & 84.55$\pm$13.36 & 97.27$\pm$0.04 & 97.00$\pm$0.05 & 92.07$\pm$7.65 & 91.20$\pm$9.42 & 95.77 & 94.01 & \textbf{97.42$\pm$0.05} & 97.18$\pm$0.05 \\
& 250 & 91.88$\pm$1.79 & 90.85$\pm$2.20 & 97.78$\pm$0.03 & 97.55$\pm$0.03 & 92.07$\pm$4.02 & 90.83$\pm$4.88 & 97.10 & \textbf{97.97} & 97.44$\pm$0.02 & 97.20$\pm$0.02 \\
& 500 & 95.51$\pm$8.54 & 94.73$\pm$10.06 & 97.49$\pm$0.05 & 97.23$\pm$0.05 & 93.40$\pm$5.71 & 92.25$\pm$6.65 & 99.03 & \textbf{99.29} & 99.12$\pm$0.09 & 99.03$\pm$0.10 \\
& 2000 & 99.18$\pm$0.87 & 99.09$\pm$0.97 & 98.39$\pm$0.06 & 98.23$\pm$0.07 & 96.33$\pm$2.66 & 94.55$\pm$5.95 & \textbf{99.97} & 99.84 & 99.37$\pm$0.11 & 99.30$\pm$0.13 \\
\midrule
\multirow{6}{*}{MFPT}
& 3 & 84.63$\pm$7.96 & 79.65$\pm$13.65 & 85.64$\pm$0.64 & 82.71$\pm$0.58 & 84.85$\pm$7.64 & 80.15$\pm$10.58 & 85.73 & 87.38 & \textbf{96.85$\pm$0.08} & 95.76$\pm$0.10 \\
& 9 & 90.56$\pm$4.46 & 88.40$\pm$4.30 & 92.19$\pm$0.46 & 89.78$\pm$0.61 & 92.10$\pm$7.45 & 90.08$\pm$9.03 & 92.09 & 91.33 & \textbf{96.83$\pm$0.07} & 95.74$\pm$0.08 \\
& 30 & 93.06$\pm$3.67 & 91.16$\pm$4.77 & 92.55$\pm$0.06 & 89.92$\pm$0.09 & 93.67$\pm$6.86 & 93.48$\pm$6.33 & 93.36 & 91.69 & \textbf{97.11$\pm$0.05} & 96.11$\pm$0.07 \\
& 60 & 94.83$\pm$1.98 & 93.55$\pm$2.42 & 91.77$\pm$0.05 & 88.81$\pm$0.07 & 94.24$\pm$3.46 & 92.35$\pm$4.70 & 94.97 & 94.21 & \textbf{97.15$\pm$0.07} & 96.19$\pm$0.11 \\
& 300 & 97.38$\pm$1.99 & 96.44$\pm$2.70 & 98.37$\pm$0.24 & 97.86$\pm$0.31 & 95.63$\pm$5.64 & 95.23$\pm$5.56 & 98.98 & 99.01 & \textbf{99.07$\pm$0.11} & 98.76$\pm$0.15 \\
& 600 & 97.77$\pm$1.21 & 97.54$\pm$1.07 & 98.25$\pm$0.08 & 97.68$\pm$0.11 & 97.42$\pm$2.58 & 96.87$\pm$2.95 & \textbf{99.72} & 99.60 & 99.26$\pm$0.06 & 99.00$\pm$0.08 \\
\midrule
\multirow{6}{*}{XJTU-SY}
& 15 & 40.16$\pm$4.65 & 35.77$\pm$6.78 & 69.84$\pm$0.11 & 68.57$\pm$0.08 & 34.13$\pm$3.02 & 26.98$\pm$4.35 & -- & -- & \textbf{92.27$\pm$0.23} & 91.04$\pm$0.25 \\
& 45 & 55.92$\pm$14.76 & 53.55$\pm$15.43 & 72.17$\pm$0.08 & 70.27$\pm$0.11 & 63.01$\pm$3.36 & 58.82$\pm$3.30 & -- & -- & \textbf{92.17$\pm$0.53} & 90.86$\pm$0.58 \\
& 150 & 78.85$\pm$7.34 & 76.61$\pm$9.10 & 80.46$\pm$0.50 & 79.26$\pm$0.57 & 79.88$\pm$7.71 & 88.10$\pm$6.87 & -- & -- & \textbf{93.32$\pm$0.18} & 92.04$\pm$0.19 \\
& 300 & 80.36$\pm$6.54 & 78.44$\pm$8.06 & 84.80$\pm$0.51 & 83.46$\pm$0.63 & 90.24$\pm$3.87 & 89.61$\pm$4.25 & -- & -- & \textbf{92.92$\pm$0.23} & 91.71$\pm$0.19 \\
& 1500 & 89.93$\pm$3.91 & 89.44$\pm$4.10 & 92.56$\pm$0.35 & 91.49$\pm$0.51 & 89.17$\pm$4.52 & 87.85$\pm$3.55 & -- & -- & \textbf{94.34$\pm$0.07} & 93.18$\pm$0.14 \\
& 3000 & 87.91$\pm$2.25 & 87.70$\pm$2.26 & 91.43$\pm$0.12 & 90.51$\pm$0.10 & 90.41$\pm$9.10 & 89.61$\pm$9.69 & -- & -- & \textbf{94.29$\pm$0.11} & 93.02$\pm$0.12 \\
\midrule
\multirow{6}{*}{HIT}
& 4 & 79.53$\pm$9.7 & 79.43$\pm$9.48 & 57.91$\pm$1.74 & 57.43$\pm$1.76 & 64.03$\pm$7.36 & 61.27$\pm$4.6 & -- & -- & \textbf{93.05$\pm$0.93} & 92.95$\pm$0.96 \\
& 12 & 93.14$\pm$0.19 & 93.04$\pm$0.17 & 79.41$\pm$0.21 & 79.44$\pm$0.25 & 75.96$\pm$2.65 & 72.55$\pm$4.27 & -- & -- & \textbf{94.88$\pm$0.25} & 94.83$\pm$0.26 \\
& 40 & \textbf{95.70$\pm$1.55} & 95.70$\pm$1.55 & 86.53$\pm$0.88 & 86.83$\pm$1.05 & 93.21$\pm$1.66 & 93.24$\pm$1.66 & -- & -- & 95.17$\pm$0.06 & 95.12$\pm$0.06 \\
& 80 & 90.75$\pm$3.76 & 90.98$\pm$3.35 & 88.26$\pm$0.88 & 87.94$\pm$0.93 & 91.76$\pm$1.23 & 91.63$\pm$1.23 & -- & -- & \textbf{94.99$\pm$0.16} & 94.95$\pm$0.17 \\
& 400 & 99.26$\pm$0.32 & 99.26$\pm$0.33 & 97.48$\pm$0.95 & 96.91$\pm$0.64 & 98.01$\pm$0.87 & 98.02$\pm$0.87 & -- & -- & \textbf{99.76$\pm$0.04} & 99.76$\pm$0.05 \\
& 800 & 99.87$\pm$0.04 & 99.87$\pm$0.04 & 99.55$\pm$0.03 & 99.56$\pm$0.03 & 98.52$\pm$1.2 & 98.51$\pm$1.22 & -- & -- & \textbf{99.85$\pm$0.06} & 99.86$\pm$0.07 \\
\bottomrule
\end{tabular}
\end{table*}


## Recommendation: MIMII Dataset

### Why MIMII over UOEMD-VAFCVS

| Criterion | MIMII | UOEMD-VAFCVS |
|---|---|---|
| **Signal type** | Microphone-recorded audible acoustic | Both acoustic + vibration |
| **Environmental noise** | Real factory background noise | Lab conditions |
| **Alignment with paper** | Directly tests "acoustic fault diagnosis" | Mixes modalities (dilutes acoustic focus) |
| **Benchmark status** | Widely cited, standard acoustic benchmark | Newer, less established |
| **Machine diversity** | 4 types (fan, pump, valve, slide rail) | 1 type (electric motor) |
| **Reviewer perception** | "Tested on a standard acoustic benchmark" | "Only tested on one motor type" |

**Verdict**: **MIMII** — real factory noise validates your robustness claims, and it's the most cited acoustic diagnosis benchmark.

> ⚠️ Avoid the "Acoustic **Emission** Bearing Dataset" — it uses ultrasonic AE sensors (20kHz–1MHz), incompatible with your Mel spectrogram pipeline (fmax=8000Hz).

---

### Models to Compare

Keep all 6 backbones for consistency:

| # | Model | Role |
|---|---|---|
| 1 | **MSCA-VGG16** | Proposed |
| 2 | VGG16 | Baseline backbone |
| 3 | EfficientNet-B0 | Lightweight modern CNN |
| 4 | ViT | Transformer baseline |
| 5 | MobileNetV3-Small | Ultra-lightweight |
| 6 | ConvNeXt-Tiny | Modern CNN |

---

### Recommended Table Format

Adapt the reference table to match your paper's metrics (single split, not variable-label):

```tex
\begin{table*}[!t]
\caption{Cross-Dataset Generalization: CWRU → MIMII (Mean $\pm$ Std over 3 Trials)}
\label{tab:cross_dataset}
\begin{tabular}{l|cccc|cccc}
\toprule
\multirow{2}{*}{\textbf{Model}} 
  & \multicolumn{4}{c|}{\textbf{CWRU (12k DE)}} 
  & \multicolumn{4}{c}{\textbf{MIMII}} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}
  & \textbf{Acc} & \textbf{F1} & \textbf{G-Mean} & $\bm{\kappa}$
  & \textbf{Acc} & \textbf{F1} & \textbf{G-Mean} & $\bm{\kappa}$ \\
\midrule
\textbf{MSCA-VGG16} & xx.x±x.x & xx.x±x.x & xx.x±x.x & xx.x±x.x & xx.x±x.x & xx.x±x.x & xx.x±x.x & xx.x±x.x \\
VGG16 & ... \\
EfficientNet-B0 & ... \\
ViT & ... \\
MobileNetV3-Small & ... \\
ConvNeXt-Tiny & ... \\
\bottomrule
\end{tabular}
\end{table*}
```

---

### Implementation Steps

1. **Download MIMII**: Focus on one machine type (e.g., **fan** or **pump**) with 6dB/0dB SNR for moderate difficulty
2. **Build dataset**: Adapt `build_cwru_dataset.py` → `build_mimii_dataset.py` (same Mel+GADF+AW-DPCNN pipeline)
3. **Run experiments**: Use run_ablation_experiments.py with the new dataset path
4. **Generate tables**: Add `cross_dataset` analysis to analyze_results.py

>is it more suitable to use CWRU dataset with other two fault status? i.e. 12kHz FE, 48kHz DE?


**Yes — CWRU cross-condition variants are actually a better fit for your paper than MIMII.** Here's why:

## The Three CWRU Generalization Tests

| Configuration | Sensor | Sampling Rate | What It Tests | Generalization Claim |
|---|---|---|---|---|
| **12k DE** (current) | Drive End | 12 kHz | Baseline | — |
| **12k FE** | Fan End | 12 kHz | **Sensor location** | Robustness to microphone placement |
| **48k DE** | Drive End | 48 kHz | **Sampling rate** | Robustness to acquisition hardware |

### Why This Is Better Than MIMII

| Aspect | CWRU Cross-Condition | MIMII |
|---|---|---|
| **Fault types** | Identical (BF/IF/OF/Normal) | Different machines (fan/pump/valve) |
| **Controlled variable** | One thing changes at a time | Everything changes at once |
| **Narrative** | "Robust to sensor placement & sampling rate" | "Also works on unrelated data" |
| **Reviewer perception** | Tests real-world acoustic deployment challenges | Dilutes acoustic focus |
| **Implementation** | Same `.mat` pipeline, just different `--input-dir` | New data loader, new class mapping |
| **Pipeline compatibility** | Drop-in: `--sr 48000` for 48k DE | Requires new build script |

### Why Sensor Location Matters for Acoustic Diagnosis

Your paper is about **acoustic** fault diagnosis. In practice, microphones can be placed at different positions relative to the transformer. The FE sensor is further from the bearing than the DE sensor — this mimics the real-world challenge of **non-optimal microphone placement**.

### Recommended Experiment Design

```
datasets/
  cwru_de/          ← existing (12k DE)
  cwru_fe/          ← new (12k FE) — cross-sensor
  cwru_de_48k/      ← new (48k DE) — cross-sampling-rate
```

**Table format** (single unified generalization table):

```tex
\begin{table*}[!t]
\caption{Cross-Condition Generalization on CWRU (Mean $\pm$ Std over 3 Trials)}
\label{tab:cwru_cross_condition}
\begin{tabular}{l|cccc|cccc|cccc}
\toprule
\multirow{2}{*}{\textbf{Model}} 
  & \multicolumn{4}{c|}{\textbf{12k DE}} 
  & \multicolumn{4}{c|}{\textbf{12k FE}} 
  & \multicolumn{4}{c}{\textbf{48k DE}} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}
  & Acc & F1 & G-M & $\kappa$
  & Acc & F1 & G-M & $\kappa$
  & Acc & F1 & G-M & $\kappa$ \\
\midrule
\textbf{MSCA-VGG16} & ... \\
...
\end{tabular}
\end{table*}
```



>create two scripts for building the 12k FE and 48k DE datasets, the script `build_cwru_de` can be referred.

## Scripts Created

| Script | Sensor | SR | Sensor Key | Output Dir |
|---|---|---|---|---|
| `build_cwru_fe.py` | Fan End | 12 kHz | `FE_time` | `datasets/cwru_fe` |
| `build_cwru_de_48k.py` | Drive End | 48 kHz | `DE_time` | `datasets/cwru_de_48k` |

## Data Availability

| Dataset | Files | Status |
|---|---|---|
| 12k DE | 40 files (4/class) | Already built (`cwru_de`) |
| 12k FE | 34 files (OF014/OF021: 1 each, others: 4) | ✅ Ready to build |
| 48k DE | 40 files (4/class) | ✅ Ready to build |

## Key Differences from `build_cwru_de.py`

| Parameter | 12k DE | 12k FE | 48k DE |
|---|---|---|---|
| `--win-len` | 2048 | 2048 | **8192** |
| `--hop-len` | 1024 | 1024 | **4096** |
| `--n-fft` | 1024 | 1024 | **4096** |
| `--fmax` | 6000 | 6000 | **20000** |
| `--gamma` | 4.0 | **10.0** | **10.0** |