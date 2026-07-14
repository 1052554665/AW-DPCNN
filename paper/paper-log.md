some detail are revised.

# Abstract
>the current abstract sentence frames MSCA-VGG16's motivation as "noise robustness," but the real contribution is the multi-scale receptive fields and dynamic channel attention architecture. The noise robustness is just one result/benefit, not the core motivation. 

The current sentence over-emphasizes noise robustness as the *motivation*, when the real contribution is the **multi-scale + channel attention architecture**. Noise robustness is just a demonstrated benefit, not the core reason MSCA-VGG16 exists. Let me reframe:



| | Before | After |
|---|--------|-------|
| **Motivation (WHY)** | *"to address the noise sensitivity... under degraded signal conditions"* | *"to fully exploit the complementary information encoded in the fused representations"* |
| **Primary outcome** | *"improve noise robustness and fault diagnosis performance"* | *"improve fault diagnosis performance and noise robustness"* |

### Why this matters

The old version **reduces MSCA-VGG16 to a denoising module** — but the paper's real innovation is architectural: multi-scale receptive fields + SE channel attention + compact embedding. Aligning the abstract's motivation with the introduction's contribution statement (*"to further exploit the complementary information encoded in the fused representations"*) makes the value proposition clear:

1. **AW-DPCNN** → fuses heterogeneous representations
2. **MSCA-VGG16** → exploits the fused representations via multi-scale + channel attention mechanisms
3. **Noise robustness** → a demonstrated outcome, not the raison d'être


# Introduction
>reconstruct the sentence about conventional deep learning classifiers

### Before
> *"Furthermore, conventional deep learning classifiers, **while powerful**, often lack **explicit** architectural mechanisms **to exploit** the multi-scale and channel-wise characteristics of fused representations."*

### After
> *"Furthermore, conventional deep learning classifiers **are typically optimized for single-representation inputs and** lack **dedicated** architectural mechanisms **for capturing** the multi-scale and channel-wise characteristics **that emerge from** multi-representation fusion."*

### What was improved

| Change | Rationale |
|--------|-----------|
| Removed *"while powerful"* | Weak filler that disrupts flow; the concessive adds no information |
| Added *"are typically optimized for single-representation inputs"* | Establishes **why** they lack the mechanisms — logical causation rather than assertion |
| *"explicit"* → *"dedicated"* | Matches consistent phrasing used throughout the paper (e.g., Deep Learning subsection, MSCA-VGG16 section) |
| *"to exploit"* → *"for capturing"* | More neutral academic tone; "exploit" carries a slightly negative connotation |
| *"of fused representations"* → *"that emerge from multi-representation fusion"* | Ties the limitation directly to the fusion context, strengthening the logical transition to the proposed MSCA-VGG16 solution |

Made changes.



>make the contribution bullet more concise

| Removed | Reason |
|---------|--------|
| *"comprehensive"* before "ablation studies" | Implied by "ablation" |
| *"across twelve time--frequency and temporal encoding combinations"* | Overly specific for a contribution bullet; the detail belongs in Section 4 |
| *"systematically"* | Implied by experimental design |
| *"each component of"* | Redundant; "validate the proposed framework" is sufficient |
| *"confirming robustness beyond the primary benchmark"* | Already conveyed by "generalization" and the dataset names |

**Before** (48 words):
> *"Extensive experiments on the CWRU bearing dataset, including backbone comparisons, **comprehensive** ablation studies, representation comparisons **across twelve time--frequency and temporal encoding combinations**, and noise robustness evaluations, **systematically** validate **each component of** the proposed framework. Cross-sensor and cross-sampling-rate generalization is further demonstrated on the 12 kHz fan-end and 48 kHz drive-end datasets**, confirming robustness beyond the primary benchmark**."*

**After** (35 words):
> *"Extensive experiments on the CWRU bearing dataset, including backbone comparisons, ablation studies, representation comparisons, and noise robustness evaluations, validate the proposed framework. Cross-sensor and cross-sampling-rate generalization is further demonstrated on the 12 kHz fan-end and 48 kHz drive-end datasets."*

Made changes.


>make it fluent

Time--frequency representations have been extensively applied to vibration-based
bearing fault diagnosis, such as STFT spectrograms, CWT scalograms, and Mel spectrograms.

>reconstruct the sentence, it seems awkward and unnatural

Despite these successes, time--frequency representations fundamentally characterize local
spectral energy distributions and do not explicitly model long-range temporal
dependencies that may contain complementary diagnostic information.


| | Before | After |
|---|--------|-------|
| **Verb** | *fundamentally characterize* | *inherently capture* |
| **Connector** | *and do not* (weak additive) | *but do not* (sharp contrast) |
| **Qualifier** | *that **may** contain* (tentative, weak) | *that **carry*** (definitive, confident) |
| **Article** | *model long-range* (missing "the") | *model **the** long-range* |

Three problems were fixed:

1. **"fundamentally characterize"** → unnatural collocation. "Inherently capture" is both more idiomatic and more precise — time-frequency representations are *inherently* local, that's their nature.

2. **"and"** → the original used an additive connector to link a positive statement (what they do) with a negative one (what they don't do). "But" creates the logical contrast the sentence is trying to make: *they're good at X, but bad at Y.*

3. **"may contain"** → in a limitation statement, hedging with "may" undermines your argument. If the dependencies *may* contain complementary information, the reader wonders whether the limitation even matters. Dropping the hedge makes the claim stronger and more persuasive.

Made changes.



>Revise `In a parallel direction`

**"On another aspect"** is not idiomatic in English academic writing. The standard phrase for contrasting two parallel research directions is:

| ❌ Current | ✅ Recommended |
|---|---|
| *"On another aspect"* | *"In parallel"* |

This is the conventional transition when presenting two complementary lines of work (time-frequency ↔ temporal encoding), and it was in fact already used in an earlier version of the manuscript. Let me fix it:


| | Before | After |
|---|--------|-------|
| **Phrase** | *"On another aspect"* | *"In parallel"* |
| **Problem** | Not idiomatic English; sounds like a calque/translation | Standard academic transition for contrasting two complementary research directions |

*"In parallel"* is the conventional phrase used in academic writing to introduce a second body of work that runs alongside (but is distinct from) the first — here, temporal encoding methods as a parallel direction to time-frequency representations.

Made changes.

# References

>some references have extra spaces in the page numbers, like "pp. 2 9 9669-29 990" instead of "pp. 29969-29990".


I can see the issue. The `IEEEtran.bst` style automatically inserts thin-space thousand separators (`\,`) in page numbers with 5+ digits (e.g., `29\,969`), which renders as visible spaces in the PDF. Let me fix this by modifying the `.bst` file.


**Problem 1 — Spaces in page numbers:** The IEEEtran.bst style inserts thin-space thousand separators (`\,`) for numbers with 5+ digits (e.g., `29\,969`), which rendered as visible spaces in the PDF.

- **Fix:** Changed `large.number.trigger` from `#5` to `#10` in IEEEtran.bst (line 814), effectively disabling thousand separators for all practical page numbers.

**Problem 2 — BibTeX compilation error:** The references.bib file had two syntax issues:
- The `@article{SMITH2015100}` entry was missing its closing `}`
- An extra trailing `}` at the end of the file

- **Fix:** Added missing `}` to close the SMITH2015100 entry and removed the extraneous trailing brace. Also fixed the `@ARTICLE{10659159}` doi field's `}}` placement.

The PDF now compiles cleanly with page numbers rendered as `pp. 29969--29990` instead of `pp. 29 969--29 990`.

Made changes.

>remove all the "[Online]. Available: https://..." entries from the .bbl file. These appear in the bibliography for entries that have a url field. 

**What was changed:** IEEEtran.bst line 87 — `default.is.use.url` from `#1` (enable URLs) to `#0` (disable URLs). This permanently suppresses URL output for all bibliography entries, so the `.bib` file's `url` fields are preserved for reference but no longer appear in the PDF.

Made changes.

>The rejection of reviewers are as follows, how to add relative references to the paper, and how to add more comparisons to other works published in journals with their primary field being I&M.

`It also does not have any references or comparisons to other works published in journals with their primary field being I&M`

revise the tables, mark relative references. or add a new table similar to the following.

```latex
\begin{table*}[!t]
\caption{Summary of previous works (in chronological order) that consider similar acoustic signal datasets.}
\label{tab:previous_works}
\centering
\begin{tabular}{c c l l l p{6.5cm}}
\toprule
\textbf{Ref.} & \textbf{Year} & \textbf{Method} & \textbf{Input feature} & \textbf{Performance measures} & \textbf{Remarks} \\
\midrule
\cite{ref19} & 2019 & AE & Mel spectrogram & AUC &
Performance results, i.e., valve: 0.67, pump: 0.81, fan: 0.94; and slide rail: 0.90 \\

\cite{ref49} & 2020 & IDNN & Mel spectrogram & AUC &
Achieve 27\% improvement against non-stationary machine sounds. \\

\cite{ref46} & 2020 & FCN & MDF & AUC &
Performance results, i.e., valve: 0.7362, pump: 0.9996, fan: 0.9978, and slide rail: 0.9646 \\

\cite{ref44} & 2020 & One-shot learning & Spectrogram & AUC and F1 scores &
Utilizing a neural network-based feature extractor and attention mechanism. \\

\cite{ref47} & 2021 & Fully connected U-Net & Mixed features, i.e., MFCC, chroma feature, Mel spectrogram, spectral contrast, and Tonnetz & AUC and pAUC &
The proposed model achieves 83.38\% AUC and 64.51\% pAUC on average over all machine types. \\

\cite{ref50} & 2021 & CNN & Mel spectrogram & Precision, sensitivity, and accuracy &
The designed model has only 1--3 convolutional layers and gives higher accuracy than AlexNet. \\

\cite{ref48} & 2021 & Convolutional VAE & Mel spectrogram & AUC and pAUC &
The use of fully-connected models yields better performance in comparison with convolutional models. \\

\cite{ref5} & 2021 & ResNet, Gaussian mixture models, and One-class SVM & Mel spectrogram & AUC &
The use pre-trained model on the task of image classification. \\

\cite{ref51} & 2021 & SVM and MLP & Mel spectrogram & AUC &
The proposed approach improves the performance by up to 39.5\% compared with the baselines. \\

\cite{ref41} & 2021 & RawdNet & Mel spectrogram & AUC &
The proposed method is the fusion of supervised feature learning and unsupervised deep one-class neural network. \\
\bottomrule
\end{tabular}
\end{table*}

```

>look for literatures for comparison table, add a comparison table with the previous works.
- only STFT, GADF, CWT, Mel, MTF, RP, GASF, ours
- VGG16, ResNet18, EfficientNet-B0, ViT, ours

>add the following contents to proper location.

\thanks{This work is supported by the National Natural Science Foundation of
China (No. 12404545), Science Research Project of Hebei Education Depart
ment, China (No. QN2025334), and the Fundamental Research Funds for the
Central Universities (No. 2026MS144).}