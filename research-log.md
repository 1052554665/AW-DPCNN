## Time-Frequency Analysis Methods

>An Open-Set Semi-Supervised Contrastive Learning  for Bearing Fault Diagnosis

Specifically, STFT performs time-frequency analysis by applying the Fourier transform to segmented portions of the vibration signal. It generates a spectrogram that captures both time and frequency information, making it suitable for analyzing bearing vibration signals under normal conditions or mild faults, defined as

$$f_{\text{STFT}}[x(t)](t, f) = \int_{-\infty}^{\infty} x(s) g(s - t) e^{-j2\pi f s} ds \quad (1)$$

where $x(t)$ is the segmented vibration signal, $t$ is the time coordinate, $s$ is the integration variable, $f$ is the sampling rate, $g(·)$ is the window function centered around $t$, and $e^{-j2\pi f s}$ represents the complex exponential function.

However, due to its segmentation mechanism, STFT may be less effective for short-duration vibration signals. For example, when a bearing fault causes an impact signal with a very short duration, in contrast, CWT applies wavelet functions across the entire time range, enabling analysis of vibration signals of arbitrary length with multiscale resolution, defined as

$$f_{\text{CWT}}[x(t)](a, b) = \int_{-\infty}^{\infty} x(t) \psi^{*} \left(\frac{t - b}{a}\right) dt \quad (2)$$

where $\psi(·)$ is the wavelet function, $a$ is the scale parameter controlling the dilation of the wavelet, and $b$ is the translation parameter determining the shift of the wavelet.

However, both STFT and CWT may introduce information loss and fail to fully preserve the original time-domain structure. To address this limitation, TDC directly transforms the 1-D vibration signal into a 2-D grayscale image, preserving the original time-series structure and minimizing information loss, defined as

$$f_{2D}(x(t)) = o_{i,j}^{2D}(k_1, k_2) = x(j) \quad (3)$$

where $(k_1, k_2)$ represent the horizontal and vertical coordinates of a pixel in the transformed 2-D image, $o_{i,j}^{2D}$, $k_1 = \lfloor j/h \rfloor$ and $k_2 = \text{modulo}(j/h)$ represent the horizontal and vertical coordinates of $o_{i}^{1}$, respectively. $h$ represents the width of $o_i$, and $x(j)$ refers to the value of the $j$-th point in the sample $x(t)$.

-----

Refer to the above format of descriptions, Review and revise this section`\section{Acoustic Signal Representation}` in paper `paper/tim.tex`, the reconstructed section is as follows:
- Firstly, define the Mel spectrogram and its mathematical representation, explaining how it captures the frequency content of acoustic signals over time. Then, discuss the disadvantages of the Mel spectrogram, such as its sensitivity to noise and limited frequency resolution.
- Next, introduce the GAF (Gramian Angular Field) representation, providing its mathematical formulation and explaining how it encodes temporal correlations in acoustic signals. Highlight the advantages of GAF, such as its ability to capture global patterns and its robustness to noise.


------------

### Revised Structure

**§2.1 Mel Spectrogram** — follows the pattern: **define → suitable for → However (limitations)**

| Element | Content |
|---|---|
| **Definition** | Three-step construction: STFT → Mel filter bank → log compression, with the Mel frequency formula |
| **What it captures** | Spectral energy distribution and temporal evolution, suitable for stationary/slowly-varying conditions |
| **Limitations ("However...")** | (1) Log compression amplifies low-level background noise → noise sensitivity; (2) Mel-scale nonlinear compression reduces high-frequency spectral resolution; (3) Does **not** encode global temporal correlation structure — fault patterns manifesting as long-range dependencies may be missed |

**§2.2 Gramian Angular Field** — follows the pattern: **addresses limitations → define → advantages**

| Element | Content |
|---|---|
| **Motivation** | Opens with "To address the limitations of the Mel spectrogram in encoding global temporal structures..." |
| **Definition** | Full mathematical formulation: min–max normalization → polar encoding (bijective mapping) → GASF/GADF variants |
| **Advantages** | (1) Captures **global** temporal correlation structures beyond STFT's local receptive field; (2) Min–max normalization suppresses absolute amplitude → **robust to noise** and signal gain variations; (3) Polar encoding provides bijective mapping → no information loss |
| **GADF choice** | Angular differences (vs. GASF summation) → higher sensitivity to abrupt temporal variations and asymmetric patterns |

### Key Improvements
- The **"However"** paragraph for Mel spectrogram now explicitly details three concrete limitations (noise sensitivity, frequency resolution, missing temporal correlations) rather than a single vague transition sentence
- The **GAF advantages** paragraph explicitly ties each advantage back to how it addresses a Mel limitation
- The narrative flow mirrors the reference paper's pattern: Method A → limitation → Method B addresses it → advantages

Made changes.