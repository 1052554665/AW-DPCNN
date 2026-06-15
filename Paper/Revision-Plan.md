# AW-DPCNN Paper — Comprehensive Revision Plan

> Compiled from reviewers' comments across IEEE Sensors, Measurement, MSSP, and other journals.
> Status: ✅ = resolved in codebase  |  ⚡ = in progress / partially resolved  |  ❌ = not yet addressed

---

## 1. Main Contributions — How to Highlight Them

### Current Problems (per reviewers)
- "Incremental improvements to existing modules" (IEEE Sensors R1)
- "Combination of existing techniques — what have you basically added?" (Measurement R5)
- "Permutation and combination of existing methods, lacking true innovation" (Measurement R2)
- Insufficient decoupling of contributions across modules → "package-style" impression

### Proposed Reframing

| Original Framing (weak) | Revised Framing (stronger) |
|---|---|
| "We propose AW-DPCNN for fusion" | "We introduce a **contrast-guided adaptive weighting mechanism** within a dual-channel PCNN that dynamically regulates the contribution of Mel vs. GADF based on **local saliency** — unlike fixed-weight or heuristic fusion in prior work" |
| "We propose MSCA-VGG16" | "We design a **structurally aligned** classification backbone where multi-scale receptive fields and channel attention are **jointly optimized** to match the heterogeneous statistics of fused time–frequency representations" |
| General "improved accuracy" | Quantify: "AW-DPCNN fusion alone improves F1 by **X%** over concatenation; MSCA-VGG16 adds **Y%** over VGG16; combined improvement over baseline = **Z%**" |

### Actionable Steps
1. **[P0]** Add a **contribution decomposition table** (see §3 — Ablation Studies) that isolates each component's marginal gain.
2. **[P0]** Rewrite the Abstract and Introduction to frame contributions as **solutions to specific technical gaps**, not as a list of modules.
3. **[P1]** Add a **novelty positioning paragraph** in the Introduction explicitly contrasting AW-DPCNN with: (a) fixed-weight fusion [cite], (b) simple concatenation, (c) standard PCNN without adaptive weighting.
4. **[P1]** Add a **design rationale subsection** (§3.2) explaining *why* contrast-guided weighting, *why* stripe vs. symmetric kernels, *why* the specific PCNN dynamics — with references to signal processing principles.

---

## 2. Main Weaknesses — How to Address Them

### Weakness 1: Novelty perception — "package-style" innovation
**Reviewers**: IEEE Sensors R1, Measurement R2, R5

**Fix**:
- ✅ Already implemented: file-level split, duration-aware balancing, leakage verification
- ❌ Need: Contribution decomposition table with marginal gains per component
- ❌ Need: Explicit positioning against prior work in a dedicated "Related Work" comparison table

### Weakness 2: Why Mel + GADF? Why not other representations?
**Reviewers**: IEEE Sensors R1 (GADF choice), IEEE Sensors R2 (STFT, CWT, GASF, recurrence plots, MTF, MFCC)

**Fix**:
- ❌ Add **representation comparison experiment** (§3 — new experiment E0):
  - Compare Mel vs. STFT spectrogram vs. CWT scalogram (time–frequency domain)
  - Compare GADF vs. GASF vs. Markov Transition Field vs. Recurrence Plot (temporal encoding)
  - Report accuracy under same classifier
  - This justifies the Mel+GADF combination empirically
- ❌ Add physical motivation: GADF preserves temporal ordering via polar coordinates → suitable for non-stationary acoustic transients; Mel aligns with human auditory perception → suitable for harmonic fault signatures

### Weakness 3: Data leakage concerns
**Reviewers**: Measurement R3 ("high accuracy on small dataset suggests leakage"), IEEE Sensors R2

**Fix**:
- ✅ Already implemented: file-level split, session-level partitioning, JS divergence verification, metadata.csv with source_file tracking
- ⚡ Need to document in paper: Add a **"Data Split Protocol" paragraph** in §5.1 explicitly stating:
  - Split at recording-session level (not segment level)
  - All segments from one recording → same subset
  - Verification: JS divergence < 0.01 between splits
  - Random seed fixed for reproducibility

### Weakness 4: No noise robustness evaluation
**Reviewers**: IEEE Sensors R1, IEEE Sensors R2

**Fix**:
- ❌ Add **noise robustness experiment**: Inject Gaussian white noise at SNR = [-5, 0, 5, 10, 15, 20] dB into test signals, report accuracy degradation curve for proposed method vs. baselines.

### Weakness 5: Single fixed split — no repeated trials
**Reviewers**: IEEE Sensors R2, Measurement R3

**Fix**:
- ❌ Run 5-fold cross-validation or 3 independent runs with different random seeds; report **mean ± std** for all metrics.

### Weakness 6: CWRU is vibration, not acoustic
**Reviewers**: IEEE Sensors R2, Measurement R3

**Fix**:
- ⚡ Tone down generalization claims: "cross-domain validation" → "supplementary validation on a different modality"
- ⚡ The transformer-five dataset (5-class acoustic) and Group2_4 dataset (8-class acoustic) should become the **primary** generalization datasets — replace or supplement CWRU.
- ✅ transformer-five dataset already built (5,859 images, 5 classes)
- ✅ Group2_4 dataset already built (7,497 images, 8 classes)

### Weakness 7: No model complexity analysis
**Reviewers**: IEEE Sensors R2

**Fix**:
- ✅ Already implemented: `count_parameters()` and `compute_flops()` in `src/utils/metrics.py`
- ✅ Already integrated into `workflow.py` — prints params/FLOPs per model
- ❌ Need: Add a **Complexity Comparison table** (Params, FLOPs, inference time per sample) for all compared models.

---

## 3. Experiments — What to Add / Improve

### 3.1 Priority Matrix

| Priority | Experiment | Status | Effort |
|---|---|---|---|
| **P0** | Ablation: component contribution decomposition | ❌ | Medium |
| **P0** | Add 3+ SOTA baselines (EfficientNet, ViT, MobileNet) | ⚡ | Low |
| **P0** | Model complexity table (params, FLOPs, time) | ⚡ | Low |
| **P1** | Representation comparison (Mel vs STFT vs CWT; GADF vs GASF vs MTF) | ❌ | High |
| **P1** | Noise robustness (multi-SNR test) | ❌ | Medium |
| **P1** | Repeated trials with mean ± std | ❌ | Low |
| **P1** | Validation on transformer-five (5-class acoustic) | ⚡ | Low |
| **P1** | Validation on Group2_4 (8-class acoustic) | ⚡ | Low |
| **P2** | Hyperparameter sensitivity (γ, N, α_L, α_T) | ❌ | Medium |
| **P2** | ROC curves for all models (already implemented) | ✅ | Done |
| **P2** | t-SNE visualization (already implemented) | ✅ | Done |

### 3.2 Experiment Design Details

#### E0: Representation Justification (NEW)
**Goal**: Empirically justify Mel + GADF over alternatives.
**Design**:
- Time–frequency alternatives: STFT spectrogram, CWT scalogram, Mel spectrogram
- Temporal encoding alternatives: GASF, GADF, Markov Transition Field (MTF), Recurrence Plot (RP)
- Fixed classifier: MSCA-VGG16
- Fixed fusion: AW-DPCNN
- Metric: Accuracy, F1
**Expected outcome**: Mel > STFT > CWT for harmonic faults; GADF > GASF ≈ MTF > RP for temporal patterns

#### E1: SOTA Baselines (EXPAND existing Table IV)
**Current**: Baseline CNN, AlexNet, ResNet18, ConvNeXt-Tiny, VGG16
**Add**:
- EfficientNet-B0 (✅ config exists)
- ViT / Patch Transformer (✅ config exists)
- CE-ViT (✅ config exists)
- MobileNetV3-Small (❌ need to add)
- A **dedicated acoustic diagnosis network** from recent literature (e.g., MCNN, WDCNN, or TFT-based)
**All trained on same AW-DPCNN fused representations for fair comparison.**

#### E2: Ablation — Component Decomposition (EXPAND existing Table V)
**Current**: MS, CA, EH on VGG16 (5 rows)
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

This table **clearly decomposes** every component's marginal contribution — directly addressing the "package-style" criticism.

#### E3: Noise Robustness (NEW)
**Design**:
- Add Gaussian white noise to test-set signals at SNR ∈ {−5, 0, 5, 10, 15, 20, ∞} dB
- Compare: Baseline CNN, VGG16, ConvNeXt-Tiny, MSCA-VGG16
- Report accuracy vs. SNR curves
- This directly addresses IEEE Sensors R1 and R2

#### E4: Hyperparameter Sensitivity (NEW)
**Parameters to sweep**:
- γ (contrast amplification): {1, 2, 4, 8, 10, 20}
- N (PCNN iterations): {5, 8, 10, 15, 20}
- α_L, α_T: {0.0001, 0.001, 0.01}
**Report**: Accuracy vs. parameter value curves for top-2 critical parameters

#### E5: Transformer-Five & Group2_4 Validation (NEW)
**Already built**:
- `datasets/transformer-five/` — 5,859 images, 5 classes, file-level split ✅
- `datasets/Group2_4/` — 7,497 images, 8 classes, file-level split ✅
**Run**: Same pipeline (AW-DPCNN + MSCA-VGG16) on both datasets, report per-class metrics.
**Purpose**: Replace/supplement CWRU as the primary generalization validation (acoustic→acoustic instead of vibration→acoustic).

---

## 4. Related Work — What to Cite & Discuss

### 4.1 Missing Citation Categories

| Category | Examples | Why Needed |
|---|---|---|
| **Transformer-based fault diagnosis** | ViT, Swin Transformer, TimesNet for fault diagnosis | IEEE Sensors R1 explicitly asks for Transformer comparisons |
| **Efficient lightweight models** | MobileNetV3, ShuffleNet, EfficientNet for acoustic diagnosis | Reviewers ask for recent SOTA |
| **Alternative time-series encodings** | MTF [Wang & Oates 2015], Recurrence Plots [Eckmann 1987], GASF/GADF [Wang & Oates 2015] | Justify GADF choice |
| **Alternative fusion strategies** | Attention fusion, Cross-modal transformers, Bilinear pooling | Position AW-DPCNN against fusion literature |
| **Acoustic transformer diagnosis** | Recent acoustic-based transformer fault diagnosis papers (2022–2025) | Contextualize within the specific application domain |
| **PCNN applications in signal processing** | PCNN for image fusion, PCNN for medical imaging, DPCNN variants | Support the PCNN design rationale |

### 4.2 Integration Strategy
1. Add a **"Related Work" section** (§2) structured as:
   - §2.1 Acoustic Representation for Transformer Diagnosis
   - §2.2 Time-Series to Image Encoding
   - §2.3 PCNN and Multi-Modal Fusion
   - §2.4 Deep Learning for Fault Diagnosis
2. End each subsection with a **gap statement** that the proposed method addresses.
3. Add a **comparison table** in Related Work summarizing: method, representation, fusion strategy, dataset, accuracy — to position the proposed work.

---

## 5. Writing Issues — How to Fix

### 5.1 Critical Fixes (P0)

| Issue | Reviewer | Fix |
|---|---|---|
| "Traning" → "Training" | IEEE Sensors R2 | Global find-replace |
| "Loosenness" → "Loosen" | IEEE Sensors R2 | Fix in all tables/figures |
| Inconsistent class names across tables/figures | IEEE Sensors R2 | Unify to exact names from Table I |
| Overstatements: "superior robustness", "generalization capability" | IEEE Sensors R2 | Add caveats: "under the tested conditions", "on the evaluated datasets" |
| "the Mel spectrogram transformers the original..." | Measurement R5 | Grammar: "transforms" |
| Missing references for compared algorithms | MSSP | Add citations for Baseline CNN, AlexNet, ResNet18, ConvNeXt-Tiny, VGG16, CWRU |

### 5.2 Structural Improvements (P1)

| Issue | Fix |
|---|---|
| No dedicated Related Work section | Add §2 — Related Work (see §4 above) |
| Equations lack physical interpretation | Add 1–2 sentences after each key equation explaining *why* this form |
| Fig. 1 (framework) is crowded | Redraw with clearer module boundaries |
| Abstract too generic | Quantify key results: "87.45% accuracy (↑14% over baseline)" |
| Repetitive descriptions in §3 and §4 | Consolidate; remove duplicate explanations |

### 5.3 Language Polish (P2)
- Run full grammar check (Grammarly / Writefull)
- Unify terminology: always "Mel spectrogram" (not "Mel-spectrogram"), always "GADF" (not "GADF image" in some places)
- Convert passive → active where appropriate: "We designed..." instead of "A ... was designed"
- Shorten sentences >30 words

---

## 6. Future Work — How to Outline

### Current (too generic):
> "further optimize the proposed framework to improve recognition accuracy under more diverse and complex operating conditions"

### Proposed (specific, grounded in limitations):

1. **Lightweight deployment**: Investigate knowledge distillation from MSCA-VGG16 to MobileNetV3-scale models for edge deployment on embedded acoustic sensors.
2. **Multi-sensor fusion**: Extend AW-DPCNN to fuse >2 modalities (e.g., acoustic + vibration + current) via multi-channel PCNN.
3. **Open-set fault diagnosis**: Adapt the embedding head for open-set recognition to handle unknown fault types not seen during training.
4. **Self-supervised pre-training**: Explore contrastive learning on unlabeled transformer acoustic data to reduce dependence on labeled fault samples.
5. **Temporal context modeling**: Incorporate sequential modeling (LSTM / Transformer) over consecutive fused frames to capture fault evolution over time.
6. **Physics-informed regularization**: Integrate physical models of transformer acoustics (winding vibration models, partial discharge physics) as regularization terms in the loss function.

---

## 7. Prioritized Action Plan (Execution Order)

### Phase A — Quick Wins (1–2 days, already partially done)
| # | Task | Status |
|---|---|---|
| A1 | Run all SOTA baselines (EfficientNet, ViT, CE-ViT, MobileNet) on AW-DPCNN fused data | ⚡ Configs exist, need MobileNet |
| A2 | Generate model complexity table (params, FLOPs) for all models | ⚡ Code ready, need to compile table |
| A3 | Run transformer-five & Group2_4 validation | ⚡ Datasets built, need training runs |
| A4 | Fix all typos and naming inconsistencies | ❌ |
| A5 | Add missing citations (CWRU, AlexNet, ResNet18, etc.) | ❌ |

### Phase B — Core Improvements (3–5 days)
| # | Task | Status |
|---|---|---|
| B1 | Design and run full ablation table (B0–B8, see §3.2 E2) | ❌ Need additional configs |
| B2 | Representation comparison experiment (E0) | ❌ Need to implement STFT/CWT/MTF/RP generation |
| B3 | Noise robustness experiment (E3) | ❌ Need noise injection code |
| B4 | Repeated trials (5 runs) for mean ± std | ❌ |
| B5 | Rewrite Introduction & Abstract with quantified contributions | ❌ |
| B6 | Add Related Work section with comparison table | ❌ |

### Phase C — Polish & Strengthen (2–3 days)
| # | Task | Status |
|---|---|---|
| C1 | Hyperparameter sensitivity analysis (E4) | ❌ |
| C2 | Redraw crowded figures (framework, confusion matrices) | ❌ |
| C3 | Full language polish pass | ❌ |
| C4 | Response letter to reviewers (point-by-point) | ❌ |
| C5 | Future work section rewrite | ❌ |

---

## 8. Summary: Key Differentiators After Revision

| Before Revision | After Revision |
|---|---|
| "We combine Mel, GADF, PCNN, and VGG16" | "We introduce contrast-guided adaptive weighting (γ-mechanism) within PCNN, and structurally align multi-scale attention to fused heterogeneous statistics" |
| 5 baselines | 9+ baselines including Transformers, EfficientNet, MobileNet |
| 5-row ablation | 9-row full decomposition table covering fusion AND classifier components |
| Single split, no std | 5-run mean ± std |
| CWRU only for generalization | transformer-five + Group2_4 acoustic datasets as primary generalization |
| No noise test | Multi-SNR robustness curves |
| No complexity analysis | Full params/FLOPs/time table |
| Generic future work | 6 specific, grounded directions |
