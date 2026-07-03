>using "+" (plus sign) to connect method names in the paper is not standard for IEEE-TIM

The full AW-DPCNN + MSCA-VGG16 pipeline achieves 99.19\% accuracy on the 12 kHz drive-end benchmark, with strong generalization to fan-end sensor data (95.33\%) and cross-sampling-rate scenarios (94.81\% at 48 kHz).

>look through the two standard templates, `New_IEEEtran_how-to.tex` and `bare_jrnl_new_sample4.tex`. Refer to the two templates for the correct way to format paper `tim.tex` with IEEE-TIM style.

>using em-dashes (—) or en-dashes (–) in the paper text is not standard for IEEE-TIM style.

All em-dashes and en-dashes have been removed from active text. Only one remains in a `%`-commented line (line 829). Here's a summary of the 10 replacements:

| # | Location | Original | Replacement |
|---|----------|----------|-------------|
| 1 | Intro (l.46) | `characteristics — their` | `characteristics, as their` |
| 2 | Contributions (l.59) | `dataset — including...analyses — systematically` | `dataset, including...analyses, systematically` |
| 3 | Overview (l.101) | `representations—...image—to capture` | `representations, namely...image, to capture` |
| 4 | Backbone (l.359) | `modules—...head)—onto` | `modules, namely...head), onto` |
| 5 | Backbone (l.361) | `point—that is, an` | `point, that is, an` |
| 6 | Table caption (l.532) | `Dataset — Operating` | `Dataset: Operating` |
| 7 | Rep Compare (l.703) | `representations — ... — and ... — ... — are` | `representations (...) and (...) are` |
| 8 | Hyperparam (l.911) | `representation—which...domain—receives` | `representation, which...domain, receives` |
| 9 | Hyperparam (l.913) | `15–20 iterations` | `15 to 20 iterations` |
| 10 | Conclusion (l.962) | `representations—...structures—through` | `representations (...structures) through` |

All conform to IEEE-TIM style: commas for parenthetical clauses, colons for caption subtitles, parentheses for appositive lists, and "to" for numeric ranges.

Made changes.

>It seems that subsection `signal representation for fault diagnosis` in section II is repetitive with `introduction`, how to revise this subsection? also review other subsections in section II if needed.

## Diagnosis: Repetition Analysis

| Intro ¶3 | Related Work II.A | Overlap |
|----------|-------------------|---------|
| Time–frequency representations (STFT, Mel, CWT) provide joint time–frequency energy distributions | STFT: linear TF, uniform resolution, good for bearing defects; Mel: nonlinear scaling, low-freq emphasis; CWT: variable resolution, good for transients | **~90%** — II.A just expands the definitions |
| Temporal encoding (GAF, MTF, RP) preserve global temporal correlations | GAF: polar coords, GASF/GADF; MTF: transition probabilities; RP: recurrent states | **~90%** — same definitions, more detail |
| Most studies adopt single representation → underexploited complementarity | Individual representations capture partial aspects → motivates fusion | **~100%** — same conclusion |

**Root cause**: II.A currently serves as a *textbook introduction* to signal representations (what they are, how they work), not as a *literature review* (who used them for what, with what results, and what limitations were found). The definitions belong in Methodology (§III.B), which already covers STFT and GAF in detail.

---

## Recommended Revision for II.A

The rewrite below shifts focus from *defining* representations to *reviewing their application in fault diagnosis literature*, with specific gaps that motivate this work:

```latex
\subsection{Signal Representations for Fault Diagnosis}

Time--frequency representations have been extensively applied to vibration-based
bearing fault diagnosis. STFT spectrograms combined with 2-D CNNs have demonstrated
effective classification of bearing defects by converting raw vibration signals into
time--frequency images that convolutional architectures can process~\cite{7558591},
\cite{10091539}. CWT scalograms have been employed to achieve improved time--frequency
localization for transient fault signatures, with multi-resolution analysis offering
advantages over fixed-resolution STFT for certain non-stationary conditions~\cite{9760386}.
Mel spectrograms have also been explored for their perceptually motivated frequency
compression, particularly for harmonic-rich fault patterns~\cite{10091539}. Despite
these successes, time--frequency representations fundamentally characterize local
spectral energy distributions and do not explicitly model long-range temporal
dependencies that may contain complementary diagnostic information.

In a parallel direction, temporal encoding methods have been introduced to transform
one-dimensional vibration signals into two-dimensional images that preserve pairwise
temporal relationships. GAF-based encodings~\cite{11048665},~\cite{10114345} have been
applied to machinery fault diagnosis by mapping time-series segments to polar
coordinates, enabling CNNs to exploit correlation textures for fault
classification~\cite{10281942}. MTF and RP encodings have similarly been adopted to
capture sequential transition dynamics and recurrent phase-space patterns,
respectively. However, these temporal encoding methods primarily preserve correlation
structures and lack explicit frequency-domain specificity, which is critical for
distinguishing faults with similar temporal patterns but distinct spectral signatures
(e.g., different fault severities at the same bearing location).

A limited number of recent studies have explored combining multiple representations
for fault diagnosis. Representative efforts include channel-wise concatenation of
time--frequency and temporal-encoded images for CNN-based classification, and
pixel-level averaging of complementary representation pairs. However, these approaches
employ fixed fusion strategies that treat all spatial locations and feature channels
uniformly, without adaptively regulating the relative contribution of each
representation. Given that time--frequency and temporal-encoded representations
possess fundamentally different statistical properties and structural characteristics,
na\"ive fusion may introduce destructive interference rather than constructive
complementarity~\cite{8513874}~\cite{10260234}. The systematic exploration and
adaptive fusion of heterogeneous representations for bearing fault diagnosis therefore
remains an open research problem, motivating the AW-DPCNN framework proposed in this
work.
```

---

## Assessment of II.B and II.C

**II.B "PCNN and Multi-Modal Fusion"** — Generally fine. Well-structured: PCNN motivation → single-stimulus limitation → DPCNN extension → fixed-coupling limitation. Minor suggestion: add one sentence connecting DPCNN to fault diagnosis specifically (e.g., "DPCNN-based fusion has been preliminarily explored in image processing but remains largely unexplored in machinery fault diagnosis").

**II.C "Deep Learning for Fault Diagnosis"** — The first paragraph is a solid survey. The second paragraph's conclusion about "single representation" partially overlaps with II.A's new closing paragraph. Suggestion: refocus II.C's closing on the *architectural* gap (lack of multi-scale + channel attention mechanisms for fused representations) rather than the *representation-level* gap, which is now covered by II.A.

>apply the II.A revision to manuscript.tex and make the minor adjustments to II.B and II.C as well

>since the VGG16 is applied as backbone in this paper, is it necessary to discuss its specific advantages in the context of fault diagnosis, and its suitable for msca module rather than other network in this subsection?

Good question. Currently II.C lists VGG16 alongside AlexNet, ResNet18, and ConvNeXt-Tiny as architectures that "were introduced into intelligent fault diagnosis frameworks," but doesn't explain *why* VGG16 specifically suits the MSCA enhancement. That justification lives entirely in §III.D.2 (three paragraphs on architectural compatibility, interpretability, and avoidance of confounding mechanisms). 

**The problem**: II.C reads as a generic survey. It should preview the rationale for choosing VGG16, creating a logical bridge from Related Work → Proposed Method.

**Suggested revision** — replace the middle sentence of II.C to briefly note VGG16's suitability:

```latex
Subsequently, classical deep convolutional backbones, including AlexNet~\cite{},
VGG16~\cite{}, and ResNet18~\cite{}, were introduced into intelligent fault
diagnosis frameworks and achieved notable improvements in hierarchical feature
representation and classification accuracy across machinery health monitoring
applications~\cite{},~\cite{},~\cite{}. Among these, VGG16's purely sequential
topology with no skip connections or built-in attention mechanisms provides a
clean architectural foundation for attaching auxiliary modules such as multi-scale
convolution and channel attention, making it particularly suitable as a backbone
for the MSCA enhancement strategy pursued in this work.
```

This:
1. Keeps the historical survey context
2. Adds one sentence previewing the VGG16 rationale (without duplicating the full §III.D.2 argument)
3. Removes the redundant "CWRU dataset" sentence (already covered in §IV.A)
4. Creates a smooth transition to the closing paragraph about "lack of architectural mechanisms"

>apply this change


>is it recommended to use symbolic notation instead of concrete numerical values in methology?

This is an excellent question. The answer is: **Yes, it is standard academic convention to use symbolic notation in the Methodology section and defer concrete numerical values to the Experiments section.** Here's the rationale:

## Why Symbols Belong in Methodology, Values in Experiments

| Principle | Explanation |
|-----------|-------------|
| **Generality** | Methodology should describe *what* the method computes, independent of specific parameter choices — the framework should be reproducible with any reasonable settings |
| **Separation of concerns** | Concrete values (e.g., $n_{\mathrm{fft}}=1024$, $H=224$) are experimental design choices, not methodological contributions |
| **IEEE-TIM convention** | Methodology presents the math; Section IV ("Dataset and Experiments" / "Implementation Details") reports parameter settings |
| **Reproducibility** | A reader implementing the method from Section III alone should not be locked into your specific values — they can choose their own, then validate against your reported settings in Section IV |

## Concrete Values Currently in Section III (Methodology)

In your manuscript, three locations mix concrete values into the methodology:

| # | Location | Concrete Values |
|---|----------|-----------------|
| 1 | **III.B.1** (STFT, ~line 159) | $n_{\mathrm{fft}}=1024$, hop $=256$, $\Delta f \approx 11.7$ Hz, $f_s = 12$ kHz, image size $224 \times 224$, Viridis |
| 2 | **III.B.2** (GADF, ~line 238) | $n=224$, image size $224 \times 224$, Viridis |
| 3 | **III.E** (Training Strategy) | $\eta=10^{-4}$, $\lambda=10^{-3}$, scheduler factor $0.5$, patience $5$, $\eta_{\min}=10^{-6}$, batch $=32$, epochs $=30$, flip prob $0.5$, rotation $\pm10^\circ$, early stop patience $15$, seeds $42,123,456$, ImageNet stats |

Would you like me to **rewrite these three passages** to use symbolic notation and forward-reference Section IV for the concrete values? The edits would be:

- **III.B.1**: Replace concrete STFT parameters with $n_{\mathrm{fft}}$, $n_{\mathrm{hop}}$, $f_s$, $\Delta f$, $H \times W$, and add `\ref` to Section IV
- **III.B.2**: Replace $n=224$, $224 \times 224$ with $n$, $H \times W$, and forward-reference
- **III.E**: Replace all concrete training hyperparameters with symbols ($\eta$, $\lambda$, $B$, $E_{\max}$, $p_{\mathrm{es}}$, etc.) and note that values are given in Section IV

>**rewrite these three passages** to use symbolic notation and forward-reference Section IV for the concrete values

### Changes Applied

| # | Location | What Changed |
|---|----------|--------------|
| 1 | **III.B.1** (STFT, line 159) | `$n_{\mathrm{fft}}=1024$`, `256 samples`, `11.7 Hz`, `12 kHz`, `224×224`, `Viridis` → `$n_{\mathrm{fft}}$`, `$n_{\mathrm{hop}}$`, `$\Delta f = f_s/n_{\mathrm{fft}}$`, `$f_s$`, `$H \times W$` + `\ref` to §IV.A |
| 2 | **III.B.2** (GADF, line 238) | `$n=224$`, `224×224`, `Viridis` → `$n$`, `$H \times W$`, "perceptually uniform colormap" + `\ref` to §IV.A |
| 3 | **III.E** (Training Strategy) | All concrete values replaced with symbols: `$\eta$`, `$\lambda$`, `$\alpha_{\mathrm{reduce}}$`, `$p_{\mathrm{reduce}}$`, `$\eta_{\min}$`, `$B$`, `$E_{\max}$`, `$p_{\mathrm{flip}}$`, `$\theta_{\max}$`, `$p_{\mathrm{es}}$`; ImageNet stats, PyTorch version, and specific seeds removed; closing sentence added with `\ref` to §IV |

### Symbols Introduced

| Symbol | Meaning |
|--------|---------|
| $n_{\mathrm{fft}}$ | FFT window size |
| $n_{\mathrm{hop}}$ | Hop length |
| $f_s$ | Sampling rate |
| $\Delta f$ | Frequency resolution |
| $H \times W$ | Output image spatial dimensions |
| $\eta$, $\lambda$ | Learning rate, weight decay |
| $\alpha_{\mathrm{reduce}}$, $p_{\mathrm{reduce}}$ | LR scheduler reduction factor, patience |
| $\eta_{\min}$ | Minimum learning rate |
| $B$, $E_{\max}$ | Batch size, max epochs |
| $p_{\mathrm{flip}}$, $\theta_{\max}$ | Augmentation flip probability, rotation range |
| $p_{\mathrm{es}}$ | Early stopping patience |

The manuscript compiles successfully (exit code 0). The concrete values remain in Section IV where they belong — the STFT/GADF parameters in §IV.A (Dataset Construction), and the training hyperparameters can be placed in §IV.B (Implementation Details). The AW-DPCNN parameters already have their own table in §IV.D.

Made changes.

>the list may not proper, and also a bit redundant.

I need to clarify — which list are you referring to? Based on the attached excerpt (the GADF subsection, lines 181–219), the most likely candidate is the **three-step `enumerate` list for GAF construction**. Let me analyze it:

## Problem: The GAF Construction List (III.B.2)

The enumerated list spans lines 186–215 and has two issues:

| Issue | Explanation |
|-------|-------------|
| **Not proper** | An `enumerate` list with multi-paragraph equations is unusual in IEEE methodology sections — flowing prose with embedded equations is the standard |
| **Redundant** | Each step's text essentially paraphrases the equation that immediately follows. For example, Step 1 says "normalized to $[-1, 1]$ using min–max normalization" — then Eq. (10) shows exactly that |

The three equations ($\tilde{x}_i$ normalization, polar encoding, GADF definition) already convey the method. The surrounding text can be streamlined to provide context without duplicating the math.

---

### Proposed Revision

Replace the `enumerate` list with concise prose that introduces each equation without restating it:

```latex
The construction of a GAF representation proceeds as follows. The time series
$\boldsymbol{X} = \{x_1, x_2, \ldots, x_n\}$ is first normalized to $[-1, 1]$
via min--max scaling:
\begin{equation}
\tilde{x}_i = \frac{(x_i - \min \boldsymbol{X}) + (x_i - \max \boldsymbol{X})}
{\max \boldsymbol{X} - \min \boldsymbol{X}}.
\label{normalization}
\end{equation}
The normalized values are then mapped to polar coordinates, where each
$\tilde{x}_i$ is encoded as an angular coordinate $\phi_i$ and its temporal
position is preserved as the radial coordinate $r_i$:
\begin{equation}
\begin{cases}
\phi_i = \arccos(\tilde{x}_i), & -1 \leq \tilde{x}_i \leq 1,\
\tilde{x}_i \in \boldsymbol{\tilde X} \\[4pt]
r_i = \dfrac{i}{n}, & i = 1, \ldots, n
\end{cases}.
\label{cosine angle}
\end{equation}
Finally, pairwise angular differences or sums are aggregated to form the
Gramian Angular Summation Field (GASF) and Gramian Angular Difference Field
(GADF). The GADF variant,
\begin{equation}
\boldsymbol{G_d}(i,j) = \sin(\phi_i - \phi_j), \quad i,j = 1,\ldots,n,
\label{eq:gadf}
\end{equation}
is adopted in this study due to its higher sensitivity to abrupt temporal
variations~\cite{11278186}.
```

This reduces ~35 lines to ~20, removes the `enumerate` scaffolding, and lets the equations speak for themselves — the text only provides transition and motivation.

>apply the revision to the GAF construction list - replacing the enumerate list with flowing prose.


>In our pipeline, the vibration signals are first transformed into time–frequency and temporal-encoded representations using STFT and GADF, respectively. Consequently, is it more proper that the STFT and GADF for vibration signals processing should be unified in terms of notation and presentation style? i.e., Firstly, define a vibration signal, then apply the STFT and GADF transformations.


This is an excellent observation. Let me analyze the current notation inconsistency before providing a recommendation.

## Analysis: Notation Inconsistency in Section III.B

Looking at the current two subsections:

| Aspect | **STFT (III.B.1)** | **GADF (III.B.2)** |
|---|---|---|
| Signal notation | $x(n)$ — samples indexed by $n$ | $\boldsymbol{X} = \{x_1, x_2, \ldots, x_n\}$ — time series |
| Domain framing | "Vibration signals generated by rotating machinery are inherently non-stationary..." | "To address the limitations of the STFT spectrogram..." |
| Transformation | Implicit: the signal $x(n)$ is the operand in Eq. (7) | Explicit three-step pipeline: normalize → polar → GADF |
| Connection to pipeline | None — STFT is presented standalone | Opens with reference back to STFT as complementary |
| Signal length | $N$ (FFT window) vs $L$ (from Section III.A) | $n$ (GADF sequence length) vs $L$ (from Section III.A) |

**Three problems:**

1. **Different signal variable names**: $x(n)$ in STFT vs $\boldsymbol{X} = \{x_1,\dots,x_n\}$ in GADF — a reader may wonder if these refer to the same underlying vibration signal.
2. **Different length symbols**: $N$ (STFT window) vs $n$ (GADF sequence) vs $L$ (already defined in Section III.A overview, Eq. 4).
3. **No unified framing**: Section III.A already defines $\boldsymbol{x} \in \mathbb{R}^{L}$ as "a raw vibration signal segment of length $L$" and the pipeline equations $\boldsymbol{S}_{\mathrm{STFT}} = \mathcal{T}_{\mathrm{STFT}}(\boldsymbol{x})$, $\boldsymbol{S}_{\mathrm{GADF}} = \mathcal{T}_{\mathrm{GADF}}(\boldsymbol{x})$. But III.B.1 and III.B.2 ignore this and re-introduce their own notation.

## Recommendation

**Yes — unify the notation.** The cleanest approach is:

1. **Open Section III.B** with a brief paragraph that inherits $\boldsymbol{x} \in \mathbb{R}^{L}$ from the overview (III.A), frames both representations as transformations of the same signal, and defines the STFT window length $N$ and GADF sequence length $n$ as parameters.
2. **III.B.1**: Use $\boldsymbol{x}$ (or $x[k]$ for discrete samples), use $N$ for the window size, consistent with the STFT literature.
3. **III.B.2**: Use $\boldsymbol{x}$ (subsampled to length $n$), keeping $n$ distinct from $N$.

This achieves:
- **Consistency**: One signal, two transformations — not two unrelated signal definitions.
- **IEEE-TIM standard**: Methodology presents the math with unified symbols; concrete values deferred to Section IV.
- **Reader clarity**: The complementary nature of STFT and GADF is reinforced by the shared notation.

---

Would you like me to draft the revised opening paragraph for Section III.B and the adjusted equations in both subsections to implement this unification? I would:

- Add a 3–4 line bridging paragraph at the start of III.B
- Update the STFT equation to use $\boldsymbol{x}$ / $x[k]$
- Update the GADF equation to reference the same $\boldsymbol{x}$
- Keep all symbolic notation (no concrete values)

>proceed with the edit?

## Notation Unification Summary

### Problem
The STFT and GADF subsections used inconsistent notation — STFT defined the signal as $x(n)$ with window size $N$, while GADF independently defined $\boldsymbol{X} = \{x_1,\ldots,x_n\}$, despite Section III.A already establishing $\boldsymbol{x} \in \mathbb{R}^{L}$ as the unified signal.

### Changes Applied

| # | Location | Change |
|---|----------|--------|
| 1 | **III.B opening** | Added bridging paragraph that inherits $\boldsymbol{x} \in \mathbb{R}^{L}$ from III.A, framing both representations as transformations of the same signal |
| 2 | **III.B.1 STFT** | `$x(n)$` → `$x[k]$` (explicitly defined as "the $k$-th sample of $\boldsymbol{x}$"); `$N$` → `$n_{\mathrm{fft}}$` in equation (eliminates conflict with PCNN iteration $N$) |
| 3 | **III.B.2 GADF** | `$\boldsymbol{X} = \{x_1, x_2, \ldots, x_n\}$` → `$\{x_1, x_2, \ldots, x_n\}$ drawn from $\boldsymbol{x}$`; `$\min\boldsymbol{X}$`/`$\max\boldsymbol{X}$` → `$\min_i x_i$`/`$\max_i x_i$`; removed dangling `$\boldsymbol{\tilde X}$` |

### Result
Both subsections now share a common notation spine: $\boldsymbol{x}$ (signal) → $\mathcal{T}_{\mathrm{STFT}}(\boldsymbol{x})$ / $\mathcal{T}_{\mathrm{GADF}}(\boldsymbol{x})$ (representations), consistent with the pipeline equations in III.A. The GADF-specific sequence of $n$ samples is clearly presented as a subsequence drawn from $\boldsymbol{x}$ rather than an independently defined entity.


>is it more proper to remove the following description and define the STFT directly?
Vibration signals generated by rotating machinery are inherently non-stationary and exhibit frequency-dependent characteristics that evolve over time. Direct time-domain analysis is therefore insufficient to capture discriminative fault-related information. To address this issue, the Short-Time Fourier Transform (STFT) spectrogram is adopted as a linear time--frequency representation that preserves uniform frequency resolution across the entire spectrum. 

## Change Summary

**Removed** (3 redundant sentences from III.B.1 opening):

> ~~Vibration signals generated by rotating machinery are inherently non-stationary and exhibit frequency-dependent characteristics that evolve over time. Direct time-domain analysis is therefore insufficient to capture discriminative fault-related information. To address this issue, the Short-Time Fourier Transform (STFT) spectrogram is adopted as a linear time--frequency representation that preserves uniform frequency resolution across the entire spectrum.~~

**Kept** (1 crisp definition sentence):

> The STFT spectrogram is a linear time--frequency representation that preserves uniform frequency resolution across the entire spectrum.

**Why this is proper:**

| Principle | How it applies |
|-----------|---------------|
| **Methodology = what, not why** | The "why" (non-stationarity, insufficiency of time-domain analysis) belongs in Introduction; the "what" (STFT construction) belongs here |
| **No duplication** | The bridging paragraph already states the role of each representation; the Introduction already motivates time-frequency analysis |
| **IEEE-TIM conciseness** | Methodology sections should be mathematically dense — every sentence should advance the definition or construction |

>is it necessary to describe the STFT and GADF in detail in the methodology section? and also the following pipeline properly and continuously?

The STFT spectrogram is parameterized by the FFT window size $n_{\mathrm{fft}}$ and hop length $n_{\mathrm{hop}}$, which jointly determine the temporal resolution and frequency resolution $\Delta f = f_s / n_{\mathrm{fft}}$ at a given sampling rate $f_s$. After conversion to a decibel scale via logarithmic amplitude compression, the resulting grayscale representation is mapped to a pseudo-color image of spatial dimensions $H \times W$ for compatibility with standard CNN input architectures. The concrete parameter values adopted in this study are provided in Section~\ref{Dataset Construction and Preprocessing}. Fig.~\ref{representation}(a) illustrates an example of a raw bearing vibration signal and its corresponding STFT spectrogram.

In our pipeline, the GADF is computed with a sequence length $n$ matching the target image dimension $H$, yielding a square matrix $\boldsymbol{G_d} \in \mathbb{R}^{n \times n}$. The resulting matrix is normalized to $[0, 1]$, converted to an 8-bit grayscale representation, and mapped to a pseudo-color image of spatial dimensions $H \times W$ via a perceptually uniform colormap. The concrete parameter values adopted in this study are provided in Section~\ref{Dataset Construction and Preprocessing}.

Let me now check the current state and make the edits. The core issue is:

1. **Too much implementation detail in Methodology** — dB compression, 8-bit grayscale, colormap names belong in Section IV
2. **Broken pipeline continuity** — the two post-processing paragraphs are separated by a figure and describe the same pipeline independently

**Plan:** Remove both pipeline paragraphs, replace with a single unified closing paragraph that describes the common post-processing for both representations, creating a clean bridge to the AW-DPCNN section.


## Restructured Section III.B — Before vs. After

| Aspect | Before | After |
|--------|--------|--------|
| **STFT pipeline** | Separate paragraph: dB compression → grayscale → pseudo-color → colormap → Section IV ref | Removed. Kept only parameterization sentence ($n_{\mathrm{fft}}$, $n_{\mathrm{hop}}$, $\Delta f$) |
| **GADF pipeline** | Separate paragraph: $n=H$ → $[0,1]$ norm → 8-bit → colormap → Section IV ref | Removed |
| **Narrative continuity** | Broken — pipeline details scattered across two subsections, separated by a figure | **Unified closing paragraph** after both subsections describes common post-processing for both representations |
| **Abstraction level** | Mixed — mathematical equations alongside implementation details ("8-bit grayscale", "perceptually uniform colormap") | Clean — Methodology has math; implementation details deferred to Section IV |

### New unified closing paragraph (after GADF subsection):

> Both representations are subsequently converted to pseudo-color RGB images of spatial dimensions $H \times W$ for compatibility with standard CNN input architectures, through logarithmic amplitude compression and colormap mapping. Fig.~\ref{representation} illustrates representative examples of the STFT spectrogram and GADF image. The concrete parameter values, including $n_{\mathrm{fft}}$, $n_{\mathrm{hop}}$, $f_s$, $n$, $H$, and $W$, are provided in Section~\ref{Dataset Construction and Preprocessing}.

This creates a clean bridge: **math definitions → unified post-processing → forward reference → AW-DPCNN fusion**.

Made changes.

>does the description of PCNN seem a bit redundant? is it more proper to remove this part?

Yes — the middle paragraph is redundant. Let me analyze:

| Paragraph | Content | Verdict |
|-----------|---------|---------|
| **Opening** | STFT/GADF recap → why fusion needed → 3 innovations | Keep (but trim the recap — III.B just covered this) |
| **Middle** | "The pulse coupling characteristic of PCNN enables..." | **Remove** — this is a prose paraphrase of the mathematical formulation that follows |
| **Transition** | "Building upon these characteristics..." | **Merge** into opening — it's a single transitional sentence |

The middle paragraph redundantly explains in prose what the equations (Input Normalization → Contrast-Guided Weighting → Dual-Channel PCNN) will define formally. The STFT/GADF recap at the start is also redundant with III.B. Let me consolidate all three paragraphs into one concise opening:


## Condensed AW-DPCNN Opening

**Removed** (middle paragraph — redundant prose paraphrase of the math):

> ~~The pulse coupling characteristic of PCNN enables dynamic interaction among neighboring feature responses, allowing salient structures shared across STFT and GADF representations to be synchronously enhanced while suppressing irrelevant or inconsistent activations. Such a mechanism is particularly suitable for multi-representation fusion, where local spectral textures and global temporal correlation patterns coexist across different feature domains. Moreover, the adaptive weighting strategy further regulates the contribution of each representation according to their fault-sensitive responses, thereby improving feature complementarity and discriminative capability.~~

**Removed** (opening STFT/GADF recap — already covered by III.B):

> ~~As discussed in Section II, the STFT spectrogram and GADF provide fundamentally different yet complementary representations of bearing vibration signals. Specifically, the STFT spectrogram primarily characterizes local time--frequency energy distributions with uniform spectral resolution, whereas the GADF emphasizes global temporal dependencies and dynamic correlation structures.~~

**Result**: Three paragraphs (~200 words) → one paragraph (~100 words). The section now opens directly with the fusion motivation, lists the three innovations, and transitions immediately to the mathematical formulation — no prose paraphrase of what the equations will define.

Made changes.

>check the definition of the awdpcnn, are there any redundant description? and also check the equations logic comprehensively.

## Review: AW-DPCNN Formulation (Lines 223–342)

### 🔴 Equation Logic Concern

The PCNN initialization causes **all neurons to fire indiscriminately at iteration 1**:

| Step | Value | Consequence |
|------|-------|-------------|
| L⁰ = 0, T⁰ = 0, Y⁰ = 0 | Initialization | — |
| F₁¹ = S̃₁, F₂¹ = S̃₂ | Y⁰=0 removes coupling | Stimuli = raw inputs |
| L¹ = e^{-α_L}·0 + M*0 = **0** | Linking stays at zero | **Critical** |
| U¹ = **0**·(1+β₁F₁)(1+β₂F₂) + σ = **σ** | Input information zeroed out | All neurons get σ |
| Y¹ = 1 (since σ > T⁰=0) | **All neurons fire** | No discriminative firing |

The linking term L multiplying the feeding inputs means that at n=1, all input information is discarded and firing is determined solely by σ. The network only becomes discriminative from n=2 onward. Consider verifying this is intentional.

### 🟡 Redundancy Issues

| # | Text | Issue |
|---|------|-------|
| 1 | "The variables $i$ and $j$ denote the pixel indices, and $\epsilon$ is a small positive constant preventing division by zero" | Obvious from notation; $\epsilon$ already explained in III.B |
| 2 | "Local contrast operators $C_1(i,j)$ and $C_2(i,j)$ are computed to quantify the saliency..." | Restates what Eq. (13) just defined |
| 3 | "Considering the distinct structural characteristics... stripe-shaped kernels... symmetric kernels..." | 4 lines of physical justification; verbose for methodology |
| 4 | "In this study, $\boldsymbol{W}_1$, $\boldsymbol{W}_2$, and $\boldsymbol{M}$ are structural design choices... not free hyperparameters" | Defensive/reviewer-response language |
| 5 | "Through the proposed adaptive fusion mechanism, STFT and GADF representations are no longer combined with fixed weights..." | Interpretive prose (~60 words); belongs in Discussion |
| 6 | "By integrating STFT spectral perception and GADF temporal encoding within a unified adaptive fusion framework, the proposed AW-DPCNN produces robust and discriminative representations..." | Marketing language |
| 7 | `\label{eq:normalization}` | Near-identical to GADF's `\label{normalization}`; rename to `eq:aw_input_norm` |

### 🟢 Equation Logic (Correct)

The equations themselves are mathematically sound:
- **Local contrast** = local standard deviation: $C_k = \sqrt{|\mathbb{E}[\tilde{S}_k^2] - \mu_k^2|}$ ✓
- **Adaptive weights**: $\beta_1 + \beta_2 \approx 1$ ✓
- **Threshold decay**: $T^n = e^{-\alpha_T}T^{n-1} + V_T Y^n$ (standard PCNN) ✓
- **Accumulated response + normalization**: standard ✓

## AW-DPCNN Formulation Review — Summary

### 🔴 Equation Logic — Flag for Verification

The neuron input equation $U^n = L^n(1 + \beta_1 F_1^n)(1 + \beta_2 F_2^n) + \sigma$ with $L^0 = 0$ causes **all neurons to fire indiscriminately at n=1** (U¹ = σ > T⁰ = 0). The linking term L multiplying the feeding inputs zeros out all input information at the first iteration. The network becomes discriminative only from n=2. Verify this is intentional.

### 🟢 Changes Applied

| # | Change | Before | After |
|---|--------|--------|-------|
| 1 | **Label renamed** | `eq:normalization` | `eq:aw_input_norm` (disambiguates from GADF's `normalization`) |
| 2 | **Label fixed** | `eq:eq:local_contrast` | `eq:local_contrast` (double `eq:` prefix) |
| 3 | **Input Normalization** | 3-line explanation of $i,j,\epsilon$ | 1 line: `where S₁ and S₂ denote STFT and GADF, and ε is a small positive constant` |
| 4 | **Contrast-Guided Weighting** | "Local contrast operators C₁ and C₂ are computed to quantify the saliency..." (restates equation) | Removed. Added "local standard deviation" as definition. Equation now flows directly into weights. |
| 5 | **Dual-Channel PCNN kernels** | 5-line justification of stripe vs symmetric kernels + "not free hyperparameters" | Condensed to 1 line: `W₁ and W₂ are representation-specific convolutional kernels: stripe-shaped... symmetric...` |
| 6 | **Fused Representation closing** | ~60 words interpretive prose + ~30 words marketing language | Removed both. Kept only figure reference + forward reference to Section IV. |

### 📐 Equation Correctness (Verified)

| Equation | Check |
|----------|-------|
| Local mean $\mu_k$ | ✓ Standard box filter |
| Local contrast $C_k = \sqrt{|\mathbb{E}[\tilde{S}^2] - \mu^2|}$ | ✓ Local standard deviation |
| $\beta_1 + \beta_2 \approx 1$ | ✓ Follows from definition |
| $F_k^n = W_k * Y^{n-1} + \tilde{S}_k$ | ✓ Standard PCNN feeding with coupling |
| $L^n = e^{-\alpha_L}L^{n-1} + M * Y^{n-1}$ | ✓ Standard linking field |
| $T^n = e^{-\alpha_T}T^{n-1} + V_T Y^n$ | ✓ Standard threshold dynamics |
| $U_{\text{sum}}$ + normalization | ✓ Standard accumulation |

Made changes.

>is it recommanded to define $n \geq 2$?

**No — defining $n \geq 2$ is not recommended.** It would make the formulation mathematically incomplete, since the equations need iteration 1 to compute iteration 2. The domain $n \in [1, N]$ is correct; the issue is in the **formulation**, not the domain.

### Three cleaner alternatives (if the all-fire behavior is not intentional):

| Fix | Change | U¹ becomes | Pros/Cons |
|-----|--------|------------|-----------|
| **A. Init $L^0 = \boldsymbol{1}$** | `L⁰=0` → `L⁰=1` | $U^1 = (1+\beta_1\tilde{S}_1)(1+\beta_2\tilde{S}_2) + \sigma$ | Simplest; discriminative from n=1; L then decays via $e^{-\alpha_L}$ |
| **B. Add 1 to linking** | $U = L(1+\beta_1 F_1)(1+\beta_2 F_2)$ → $U = (1+L)(1+\beta_1 F_1)(1+\beta_2 F_2)$ | $U^1 = (1+\beta_1\tilde{S}_1)(1+\beta_2\tilde{S}_2) + \sigma$ | Keeps L⁰=0; standard PCNN pattern of "1 + linking modulation" |
| **C. Keep as-is** | No change | $U^1 = \sigma$ (uniform) | Acceptable if n=1 serves as a "reset" initialization before discriminative iterations begin at n=2 |

**Recommendation**: **Option B** is the most standard PCNN-aligned fix — the linking term should modulate the feeding input, not replace it. This preserves the original design intent while fixing the n=1 degeneracy. Would you like me to apply Option B?

>apply Option B

## Option B Applied

| | Before | After |
|---|--------|-------|
| **Equation** | $U^n = L^n(1+\beta_1 F_1^n)(1+\beta_2 F_2^n) + \sigma$ | $U^n = (1+L^n)(1+\beta_1 F_1^n)(1+\beta_2 F_2^n) + \sigma$ |
| **U¹ (n=1)** | $\sigma$ (uniform, input-independent) | $(1+\beta_1\tilde{S}_1)(1+\beta_2\tilde{S}_2) + \sigma$ (input-dependent) |

The linking term now **modulates** the feeding input rather than replacing it — the standard PCNN pattern. At n=1, neurons respond to the actual STFT and GADF pixel values through the adaptive weights, making the first iteration discriminative.

>This subsection is just for the proposed model MSCA-VGG16, it seems such redundant, review this subsection, and check if it is necessary to keep the content, and also check the logic of the equations.

## MSCA-VGG16 Review

### 🔴 Major Redundancy

| # | Location | Issue |
|---|----------|-------|
| 1 | **Opening**: "STFT-derived... GADF-derived..." | 3rd repetition of this comparison (Intro §I, AW-DPCNN §III.C, now here) |
| 2 | **Backbone**: Generic textbook description of CNN backbones | ~8 lines explaining what a backbone does — unnecessary for IEEE-TIM |
| 3 | **Backbone**: Three **bold** justification paragraphs | ~30 lines / ~800 words defending VGG16 choice. Reads like a reviewer rebuttal, not methodology. |
| 4 | **MS Module**: "Fault signatures in fused representations manifest at multiple scales: fine-grained local anomalies (e.g., impulsive spectral lines)..." | Motivational fluff |
| 5 | **Embedding Head**: "Conventional CNN classifiers often employ large fully-connected layers that are prone to overfitting..." | Motivational, not methodological |

### 🟢 Equation Logic — All correct
The SE bottleneck, multi-scale summation, embedding projection, and classifier are all standard and correctly formulated.

### Plan

Condense the ~140-line section by:
- Removing the 3rd STFT/GADF comparison from the opening
- Collapsing the 3-paragraph VGG16 defense into 2 sentences
- Removing textbook backbone descriptions
- Trimming motivational fluff from MS, SE, and Embedding subsections


## MSCA-VGG16 — Condensed

| Section | Before | After | Reduction |
|---------|--------|-------|-----------|
| **Opening** | 80 words re-explaining STFT vs GADF + motivation | 40 words: "heterogeneous features spanning multiple scales" — no repetition | **-50%** |
| **Convolutional Backbone** | ~40 lines: textbook CNN description + 3 **bold** justification paragraphs (~800 words) defending VGG16 | 5 lines: states choice, 1-sentence rationale, equation, forward ref | **-85%** |
| **MS Module** | "manifest at multiple scales: fine-grained local anomalies (e.g., impulsive spectral lines), mid-range harmonic patterns, and long-range structural correlations..." | "span multiple scales, from fine-grained local anomalies to long-range structural correlations" | **-40%** |
| **Embedding Head** | "Conventional CNN classifiers often employ large fully-connected layers that are prone to overfitting on small-scale fault diagnosis datasets and introduce substantial parameter redundancy. We replace these with..." | "To promote intra-class compactness and inter-class separation while limiting parameter count, we replace conventional large fully-connected layers with..." | **-50%** |
| **SE Module** | Unchanged | — | Equations are standard SE — kept as-is |

### 🟢 Equations — All verified correct

| Equation | Status |
|----------|--------|
| ŷ = F_cls∘E∘P∘M∘B(X) | ✓ Modular composition |
| F = B(X) | ✓ Backbone |
| F_ms = Σ ψ_i(F; Θ_i) | ✓ Multi-scale sum |
| F_act = δ(BN(F_ms)) | ✓ BN + ReLU |
| z_c = GAP, s = σ(W₂δ(W₁z)) | ✓ SE bottleneck |
| F_att = s_c · F_act | ✓ Channel rescaling |
| f₀ → f₁ → e → ŷ | ✓ Embedding pipeline |
| L_CE = -Σ y_k log ŷ_k | ✓ Cross-entropy |

>check Section IV, are there details for relative hyperparameters? if not, add them.

## Hyperparameters Added to Section IV.B

| Section III reference | Symbol | Value now in §IV.B |
|-----------------------|--------|---------------------|
| MS kernel sizes, $S$ | $S$, kernels | $S=3$, $3\times3$, $5\times5$, $3\times3$ dilation-2 |
| SE reduction ratio | $r$ | $r=16$ |
| Embedding dimensions | $D_1$, $d$ | $D_1=1024$, $d=256$, dropout $0.5$ |
| Learning rate | $\eta$ | $1\times10^{-4}$ |
| Weight decay | $\lambda$ | $1\times10^{-3}$ |
| Scheduler factor, patience | $\alpha_{\text{reduce}}$, $p_{\text{reduce}}$ | $0.5$, $5$ |
| Min learning rate | $\eta_{\min}$ | $1\times10^{-6}$ |
| Batch size, max epochs | $B$, $E_{\max}$ | $32$, $30$ |
| Flip prob, rotation | $p_{\text{flip}}$, $\theta_{\max}$ | $0.5$, $\pm10^\circ$ |
| Early stopping patience | $p_{\text{es}}$ | $15$ |
| Random seeds | — | 42, 123, 456 |

All forward references from Section III are now resolved. The AW-DPCNN parameters already had their own table (Table `tab:awdpcnn_params`), and STFT/GADF parameters were already in §IV.A.

Made changes.

>since w1, w2 have been used in the awdpcnn, it is better to rename the w1, w2 in the SE module to avoid confusion.

**Fix applied:**

| | Before (conflict) | After (clean) |
|---|-------------------|---------------|
| **AW-DPCNN** | `W₁,W₂` = STFT/GADF conv kernels | `W₁,W₂` = STFT/GADF conv kernels (unchanged) |
| **SE module** | `W₁,W₂` = bottleneck weight matrices | `U₁,U₂` = bottleneck weight matrices |

The SE equation is now: $\boldsymbol{s} = \sigma(\boldsymbol{U}_2 \, \delta(\boldsymbol{U}_1 \boldsymbol{z}))$ with `U₁ ∈ R^{C/r × C}` (down-projection) and `U₂ ∈ R^{C × C/r}` (up-projection).

Made changes.



>Is it more formal to use commas "," rather than ":" before displayed equations?

**Summary of changes:**

| Line(s) | Section | Before | After |
|---|---|---|---|
| 185 | GADF (normalization) | `scaling:` | `scaling,` |
| 192 | GADF (polar coords) | `r_i:` | `r_i,` |
| 372 | MS module | `activated:` | `activated,` |
| 396 | SE module | `rescaling:` | `rescaling,` |
| 416 | Embedding (projection) | `features:` | `features,` |
| 423 | Embedding (compact) | `space:` | `space,` |
| 430 | Embedding (classifier) | `classifier:` | `classifier,` |
| 437 | Embedding (loss) | `loss:` | `loss,` |

The compilation is clean (19 pages). The existing warnings (undefined `fig:tsne_backbones`, `fig:roc_curves`, etc.) are pre-existing label issues, not related to this change.

**IEEE-TIM style note:** Commas are preferred over colons before displayed equations because the equation is treated as part of the sentence flow — not as a separate announcement. This is now uniform throughout the manuscript.

>review the methodology of MSCA-VGG16, are there any errors in algorithm?

## Review: MSCA-VGG16 Algorithm 1 — Forward Pass

### ✅ Verified Correct (cross-referenced with MSCA_VGG16.py)

| Step | Description | Eq. Reference | Code Match |
|------|-------------|---------------|------------|
| 1 | Backbone feature extraction | `eq:feature_map` | `self.features(x)` |
| 2 | Multi-scale conv summation | `eq:multi_scale_feature` | `conv3(x) + conv5(x) + conv_dilated(x)` |
| 3 | BN + ReLU activation | `eq:active_feature` | `self.bn(out); self.relu(out)` |
| 4 | SE squeeze (GAP → z) | `eq:se_squeeze` | `AdaptiveAvgPool2d(1)` |
| 5 | SE excitation (bottleneck) | `eq:se_excitation` | `Conv2d→ReLU→Conv2d→Sigmoid` |
| 6 | Channel rescaling | `eq:attention_feature` | `out * self.se(out)` |
| 7 | GAP + Flatten → f₀ | `eq:flatten_feature` | `avgpool(x); flatten(x, 1)` |
| 8 | fc1 projection + dropout | `eq:fc1_feature` | `Linear(512,1024)→ReLU→Dropout` |
| 9 | Embedding + dropout | `eq:embedding` | `Linear→BN→ReLU→Dropout` |
| 10 | Classifier | `eq:classifier` | `Linear(embed_dim, num_classes)` |

**No mathematical or logical errors.** All equation references are correct, and the flow matches the implementation exactly.

---

### ⚠️ Minor Issues Worth Addressing

**1. `\Ensure` mixes forward pass with training concern**

The `\Ensure` statement reads:
> "Predicted probabilities $\hat{\boldsymbol{y}} \in \mathbb{R}^{K}$, optimized via Eq.~\ref{eq:cross_entropy_loss}."

The algorithm is titled **"Forward Pass"** — the `\Ensure` block should describe only the output of the forward pass. The cross-entropy loss is a training objective, not part of inference. Suggested fix:

```latex
\Ensure Predicted class probabilities $\hat{\boldsymbol{y}} \in \mathbb{R}^{K}$.
```

Move the loss reference to the surrounding prose.

---

**2. Softmax in `eq:classifier` vs. implementation**

Equation `eq:classifier` writes:
$$\hat{\boldsymbol{y}} = \mathrm{softmax}(\boldsymbol{W}_c \boldsymbol{e})$$

In the code, `self.classifier` is `nn.Linear(embed_dim, num_classes)` — **no softmax**, because PyTorch's `CrossEntropyLoss` applies `log_softmax` internally. This is a standard simplification in academic papers and is mathematically consistent with `eq:cross_entropy_loss` ($\mathcal{L}_{\mathrm{CE}} = -\sum y_k \log \hat{y}_k$). **Not an error**, but if you want strict code–paper parity, you could add a footnote or use $\boldsymbol{z} = \boldsymbol{W}_c \boldsymbol{e}$ for logits and $\hat{\boldsymbol{y}} = \mathrm{softmax}(\boldsymbol{z})$.

---

**3. Dropout omitted from equations (but present in algorithm)**

The algorithm correctly notes "then apply dropout" at steps 8 and 9. However, the corresponding equations (`eq:fc1_feature`, `eq:embedding`) omit dropout notation:
- `eq:fc1_feature`: $\boldsymbol{f}_1 = \delta(\boldsymbol{W}_{fc1} \boldsymbol{f}_0)$ — missing `Dropout(·)`
- `eq:embedding`: $\boldsymbol{e} = \delta(\mathrm{BN}(\boldsymbol{W}_e \boldsymbol{f}_1))$ — missing `Dropout(·)`

This is standard (dropout is a training regularization, not part of the architectural definition), but a brief note like "Dropout is applied after each activation during training" would preempt reviewer questions.

---

**4. Notation: `\mathcal{F}_{cls}` in pipeline vs. algorithm**

Equation `eq:cls_pipeline` uses the functional abstraction:
$$\hat{\mathbf{y}} = \mathcal{F}_{cls}(\mathcal{E}(\mathcal{P}(\mathcal{M}(\mathcal{B}(\boldsymbol{X})))))$$

But the algorithm and subsequent equations use concrete notation ($\boldsymbol{W}_c$, $\mathrm{softmax}$, etc.). The functional notation $\mathcal{F}_{cls}$ appears **only once** in the entire methodology. Consider either removing the pipeline equation or consistently using functional notation throughout — the current mix is slightly dissonant.

---

### Verdict

**Algorithm 1 is correct.** No errors in the step sequence, equation references, or variable naming. The issues above are stylistic/consistency concerns, not technical errors. The most actionable fix is item #1 (cleaning up the `\Ensure` statement).


>whether it's proper to describe the training strategy in the Methodology section (Section III) or in the Experiments section (Section IV)

## Analysis: Training Strategy in Methodology vs. Experiments

**Short answer: Move it to Section IV.** Here's why:

| Principle | How it applies |
|-----------|---------------|
| **Methodology = what the model IS** | Architecture, equations, fusion mechanism, forward pass — these are the intellectual contributions |
| **Experiments = how the model is TRAINED** | Optimizer, scheduler, batch size, augmentation, early stopping — these are implementation choices, not methodological contributions |
| **IEEE-TIM convention** | Methodology presents the math; Section IV reports how experiments were conducted |
| **Reproducibility** | A reader reimplementing your AW-DPCNN + MSCA-VGG16 architecture should not need to know your AdamW settings to understand the method |

### What III.E currently contains

| Content | Belongs in... | Reason |
|---------|--------------|--------|
| AdamW optimizer, ReduceLROnPlateau scheduler | **§IV** | Standard training tools |
| Class-weighted CE loss equations | **§IV** | Standard technique (not novel) |
| Data augmentation (flip, rotation) | **§IV** | Implementation detail |
| Early stopping protocol | **§IV** | Training heuristic |
| ImageNet normalization | **§IV** | Implementation detail |
| Multi-trial, random seeds | **§IV** | Experimental protocol |
| Symbolic notation + `\ref` to §IV | — | Already done; now move the whole subsection |

### Recommendation

1. **Move III.E to Section IV** as a sub-subsection under `§IV.B Implementation Details` (rename to "Training Protocol and Evaluation Metrics" or similar)
2. **Keep only a 1-sentence forward reference** in III.D (Embedding Head), where `eq:cross_entropy_loss` is already introduced — add: "The training protocol, including the class-weighted loss variant, is detailed in Section~\ref{section4}."
3. The `\ref` to §IV for concrete values becomes self-contained within Section IV

This removes ~40 lines from Methodology, consolidates all training details in one location, and follows IEEE-TIM convention. Want me to apply this?

>apply

**Summary of the move:**

| Before | After |
|--------|-------|
| **§III.E** — 40-line Training Strategy subsection in Methodology | **Removed.** Methodology now ends with Algorithm 1. |
| **§IV.B** — Two paragraphs with scattered training details | **Consolidated.** Now contains: class-weight equation + weighted CE equation + AdamW/scheduler protocol + augmentation + early stopping + multi-trial + hardware/metrics |

**Section III (Methodology) now contains only:**
- III.A — Overview (pipeline equations)
- III.B — Signal Representation (STFT + GADF)
- III.C — AW-DPCNN Fusion
- III.D — MSCA-VGG16 Classifier (architecture + algorithm)

**Section IV (Experiments) now has all training details in one place:**
- IV.A — Dataset Construction
- IV.B — Implementation Details & Training Protocol (equations + concrete values)
- IV.C–IV.I — Experiments (rep compare, backbone compare, ablation, etc.)

>is it better to give a more comprehensive equation explanation for Multi-Scale Convolutional (MS), squeeze-and-excitation (SE) and compact discriminative embedding head modules in the MSCA-VGG16 architecture? it looks simple now. 


Good question. Let me assess each module against the code and IEEE-TIM standards:

## Current Equation Coverage — Gap Analysis

| Module | Current Equation | Issue |
|--------|-----------------|-------|
| **MS** | $F_{\mathrm{ms}} = \sum_{i=1}^{S} \psi_i(F; \Theta_i)$ | **Too abstract.** The three specific branches (3×3, 5×5, dilated 3×3) are only described in prose, not mathematically defined. A reviewer needs to read the prose to understand what ψ_i actually computes. |
| **SE** | Squeeze → excitation → rescale (3 equations) | ✅ Adequate. GAP, bottleneck, channel-wise rescaling are all mathematically explicit. |
| **Embedding** | Flatten → fc1 → embed → classifier (4 equations) | ✅ Adequate. All weight matrices shown with dimensions. |
| **GAP** | $\mathrm{AvgPool}(\cdot)$ in flatten equation | **Minor.** The reduction from $H \times W$ to $1 \times 1$ is implicit. |

**Key finding**: The MS module equation is the only genuinely underspecified component. It uses an abstract sum-of-functions notation while the prose describes three concrete convolution branches — this is a mismatch between mathematical rigor and prose explanation.

---

## Proposed Expansion for MS Module

Replace the current abstract summation with explicit branch definitions that match the code:

```latex
Fault signatures in fused representations span multiple scales, from fine-grained 
local anomalies to long-range structural correlations. To align the network's 
receptive field with this multi-scale nature, we introduce a multi-scale convolution 
(MS) module comprising $S=3$ parallel branches with distinct kernel configurations:

\begin{equation}
\begin{aligned}
\boldsymbol{F}_{\mathrm{ms}}^{(1)} &= \boldsymbol{\Theta}_{3\times3} * \boldsymbol{F}, \quad 
\text{(local fine-grained textures)}, \\
\boldsymbol{F}_{\mathrm{ms}}^{(2)} &= \boldsymbol{\Theta}_{5\times5} * \boldsymbol{F}, \quad 
\text{(mid-range structural features)}, \\
\boldsymbol{F}_{\mathrm{ms}}^{(3)} &= \boldsymbol{\Theta}_{3\times3}^{\mathrm{d=2}} * \boldsymbol{F}, \quad
\text{(long-range dependencies via dilation)},
\end{aligned}
\label{eq:multi_scale_branches}
\end{equation}
where $*$ denotes convolution, $\boldsymbol{\Theta}_{k \times k} \in \mathbb{R}^{C \times C \times k \times k}$ are learnable kernels, and $\boldsymbol{\Theta}_{3\times3}^{\mathrm{d=2}}$ uses dilation rate~2. Each branch preserves the channel dimensionality $C$ through appropriate padding. The multi-scale features are fused via element-wise summation:
\begin{equation}
\boldsymbol{F}_{\mathrm{ms}} = \boldsymbol{F}_{\mathrm{ms}}^{(1)} + \boldsymbol{F}_{\mathrm{ms}}^{(2)} + \boldsymbol{F}_{\mathrm{ms}}^{(3)}.
\label{eq:multi_scale_fusion}
\end{equation}
```

This:
1. ✅ Shows all three branches explicitly (matches `conv3`, `conv5`, `conv_dilated` in code)
2. ✅ Defines kernel tensor notation ($\boldsymbol{\Theta}$) matching the original $\boldsymbol{\Theta}_i$
3. ✅ Justifies each branch's design purpose inline
4. ✅ Makes channel preservation explicit through "appropriate padding"
5. ✅ Stays modular — the $S$ count is explicit ($S=3$) but the pattern is clear

Want me to apply this? No other module needs expansion — the SE and embedding equations are already at the right level of IEEE-TIM mathematical rigor.