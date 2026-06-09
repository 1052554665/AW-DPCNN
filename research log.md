# Dataset Construction and Public Dataset Recommendations
```markdown
The dataset of this project is created as follows:
- Firstly, the raw data is transformed with mel spectrograms `scripts/mel.py` and GADF images `scripts/GAF.py`, seprately.
- Then, the mel spectrograms and GADF images are fused with awdpcnn `scripts/awdpcnn.py`.

which public datasets are suitable for this project and reliable for publishing in IEEE?

meanwhile, refer to these scripts and reconstruct the three scripts using one script.
```

## 1. Public Datasets Suitable for IEEE Publication

Based on your project's pipeline (acoustic signal → Mel spectrogram + GADF → AW-DPCNN fusion → classification), here are the most reliable and well-cited public datasets:

### Primary Recommendations

| Dataset | Domain | Why Suitable | IEEE Track Record |
|---|---|---|---|
| **CWRU Bearing** (already used) | Vibration / rotating machinery | Gold standard; >10,000 citations; 4 fault types at multiple severities | Excellent |
| **MIMII** (Hitachi / DCASE) | Real **acoustic** recordings of industrial machines (valves, pumps, fans, slide rails) | Directly matches your acoustic pipeline; designed for anomaly detection | Strong & growing |
| **ToyADMOS** (DCASE 2020 Task 2) | Machine operating sounds with anomaly labels | Acoustic anomaly detection benchmark; 6 machine types | Strong |
| **PU Bearing** (Paderborn Univ.) | Bearing vibration + current signals under real damage | More realistic than CWRU; compound faults; well-documented | Excellent |
| **XJTU-SY Bearing** (Xi'an Jiaotong Univ.) | Full run-to-failure bearing data | Accelerated life tests; 3 operating conditions; gaining citations rapidly | Excellent |

### Secondary / Supporting Datasets

| Dataset | Domain | Notes |
|---|---|---|
| **MFPT** (Society for MFPT) | Bearing fault acoustic & vibration | Multiple fault severities; real-world conditions |
| **SEU Gearbox** (Southeast Univ.) | Gearbox vibration under 5 conditions | Well-structured for transfer learning experiments |
| **IMS Bearing** (NSF I/UCRC) | Run-to-failure bearing vibration | 3 test-to-failure experiments; useful for robustness |
| **HUST Bearing** (Huazhong Univ. Sci. & Tech.) | Motor bearing faults | 5 fault types; good for cross-dataset generalization |

### Recommendation for IEEE Submission

For a strong IEEE paper, I recommend **2–3** public datasets beyond your own field transformer data:

1. **CWRU** (already in your paper — keep it as external validation)
2. **MIMII** or **ToyADMOS** — directly validates your *acoustic* pipeline on publicly available acoustic data (not just vibration), which substantially strengthens your claim
3. **PU** or **XJTU-SY** — validates generalization to different machinery and operating conditions

All of these have been published in IEEE Trans. on Industrial Electronics, IEEE Trans. on Power Delivery, IEEE Access, MSSP, etc.

---

## 2. Unified Script: build_fused_dataset.py

I've created the unified script that merges all three stages (Mel, GAF, AW-DPCNN) into **one in-memory pipeline**. Key improvements over the original three-script workflow:

- **In-memory processing**: Mel and GAF are generated from the *same signal window* and fused immediately — no fragile index-matching across folders needed
- **Single-pass**: WAV → fused PNG in one shot, respecting class sub-directory structure (ImageFolder-ready output)
- **Configurable**: sliding window, colormap, Mel/GAF params, PCNN iterations, all via CLI
- **Parallel**: `ProcessPoolExecutor` with configurable workers

### Quick usage

```bash
# With sliding window (3000-sample windows, 75% overlap)
python scripts/build_fused_dataset.py \
    --input-dir  ./raw_wavs/train \
    --output-dir ./datasets/train \
    --win-len 3000 --hop-len 750 --img-size 224 \
    --n-iter 8 --workers 16

# Pre-segmented WAVs (one image per WAV)
python scripts/build_fused_dataset.py \
    --input-dir  ./pre_segmented/test \
    --output-dir ./datasets/test \
    --win-len 0 --img-size 224

# Also save intermediate Mel & GAF for inspection
python scripts/build_fused_dataset.py \
    --input-dir ./raw_wavs/val \
    --output-dir ./datasets/val \
    --save-intermediates
```

# Selection for ToyADMOS Subset
```markdown
for ToyADMOS dataset, which is the most suitable for this project? ToyCar, ToyConveyor or ToyTrain? "toy car" for product inspection task, "toy conveyor" for fault diagnosis for fixed machine task, and "toy train" for fault diagnosis for moving machine task.
```


**ToyConveyor** is the most suitable subset for your project.

The reasoning is straightforward:

| Subset | Task | Match to Your Project |
|---|---|---|
| ToyCar | Product inspection (pass/fail of individual items) | ❌ Quality inspection, not equipment fault diagnosis |
| **ToyConveyor** | **Fault diagnosis for fixed machine** | ✅ **Best match** — transformers are stationary equipment, same monitoring paradigm |
| ToyTrain | Fault diagnosis for moving machine | ❌ Involves Doppler effects, moving sound sources — unrepresentative of transformer acoustics |

Transformers are **fixed/stationary** assets monitored by fixed-position sensors. ToyConveyor replicates exactly this scenario: a stationary machine with continuous acoustic monitoring under known fault conditions. The absence of motion-induced acoustic artifacts (Doppler shift, varying distance) makes ToyConveyor the most transferable public benchmark for validating your AW-DPCNN pipeline beyond your own field transformer data.

