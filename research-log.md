The methodology in this paper `paper/tim.tex` seems weak and lacks sufficient detail, referring to relative scripts to rewrite this part. All the subsection should be expanded to provide a clearer understanding of the research process, and the principles behind the chosen methods should be explained thoroughly and coherently. The following formats can be referred to for rewriting this section.

-----------


## A. Overview

Our proposed open-set semi-supervised contrastive learning (OSCL) framework comprises three stages: data preparation, pretraining, and model fine-tuning, as illustrated in Fig. 2. In the data preparation stage, three signal analysis methods are applied to extract representative features from time-series vibration signals. During pretraining, a contrastive learning approach is employed to enable the model to learn domain-specific features from large volumes of unlabeled data.

In the fine-tuning stage, to achieve high-quality few-shot bearing fault diagnosis under both closed-set and open-set conditions, we introduce three types of classifiers: a closed-set classifier, an open-set classifier, and a multibinary classifier. Specifically, the closed-set classifier consists of two fully connected layers (FCs) and is responsible for classifying input data into the known $n$ categories (i.e., the closed-set setting). Similarly, the open-set classifier (OC) classifies input data into $n+1$ categories, thereby extending the model's capability to recognize unknown classes in the open-set setting.

Considering that most of the data are unlabeled, we adopt pseudo-labeling techniques to effectively leverage unlabeled data, enhancing the open-set classifier. However, the closed-set classifier can only assign unlabeled samples to known categories and is incapable of recognizing unknown classes. To address this limitation, we introduce a multibinary classifier to assist the pseudo-label generation. The multibinary classifier is composed of $n + 1$ independent binary classifiers, where the $k$th binary classifier predicts the probability that a sample belongs to the $k$th class, including the probability estimation for the $(n+1)$th class (unknown). By integrating the predictions from the closed-set classifier and the multibinary classifier, we can generate more accurate pseudo-labels for the unlabeled data, including those belonging to unknown classes, thereby enhancing the performance of the open-set classifier.

Given a vibration signal $L$ collected from various sensors, time-series signals are interpreted from a 2-D image perspective, where the horizontal axis represents time and the vertical axis corresponds to signal amplitude. A vibration image dataset $\mathcal{D} = [D_l, D_u]$ is generated using overlapping data segmentation with a fixed length $L$. The labeled dataset $D_l = \{X_l, y_l\}$ contains $N_l$ labeled samples, where $X_l \in \mathbb{R}^{d \times w \times h \times N_l}$ represents the vibration images and $y_l \in \{1, ..., C\}$ denotes their corresponding fault diagnosis labels from $C$ seen classes. Here, $d$ is the sample dimension and $w$ and $h$ are the width and height of the vibration image, respectively. The unlabeled dataset $D_u = \{X_u\} = \{x_i^u\}_{i=1}^{N_u}$ consists of $N_u$ unlabeled samples $X_u \in \mathbb{R}^{d \times w \times h \times N_u}$, which may include samples from unseen classes. The open-set bearing fault diagnosis can be formulated as a multiclass classification problem. The model assigns samples to one of the known fault types or an additional class $(C+1)$ representing unknown fault types.

---

## B. Data Preparation

Bearing faults are typically monitored via vibration signals, which are inherently nonlinear, nonstationary, and often contaminated with noise [41]. Previous studies [23], [42] have shown that STFT, CWT, and TDC are effective for fault-sensitive signal processing. Accordingly, we employ these methods to extract features from raw signals. Under complex operating conditions or varying loads, their complementary characteristics enhance fault representation and diagnostic performance.

Specifically, STFT performs time-frequency analysis by applying the Fourier transform to segmented portions of the vibration signal. It generates a spectrogram that captures both time and frequency information, making it suitable for analyzing bearing vibration signals under normal conditions or mild faults, defined as

$$f_{\text{STFT}}[x(t)](t, f) = \int_{-\infty}^{\infty} x(s) g(s - t) e^{-j2\pi f s} ds \quad (1)$$

where $x(t)$ is the segmented vibration signal, $t$ is the time coordinate, $s$ is the integration variable, $f$ is the sampling rate, $g(·)$ is the window function centered around $t$, and $e^{-j2\pi f s}$ represents the complex exponential function.

However, due to its segmentation mechanism, STFT may be less effective for short-duration vibration signals. For example, when a bearing fault causes an impact signal with a very short duration. In contrast, CWT applies wavelet functions across the entire time range, enabling analysis of vibration signals of arbitrary length with multiscale resolution, defined as

$$f_{\text{CWT}}[x(t)](a, b) = \int_{-\infty}^{\infty} x(t) \psi^{*} \left(\frac{t - b}{a}\right) dt \quad (2)$$

where $\psi(·)$ is the wavelet function, $a$ is the scale parameter controlling the dilation of the wavelet, and $b$ is the translation parameter determining the shift of the wavelet.

However, both STFT and CWT may introduce information loss and fail to fully preserve the original time-domain structure. To address this limitation, TDC directly transforms the 1-D vibration signal into a 2-D grayscale image, preserving the original time-series structure and minimizing information loss, defined as

$$f_{2D}(x(t)) = o_{t}^{2D}(k_1, k_2) = x(j) \quad (3)$$

where $(k_1, k_2)$ represent the horizontal and vertical coordinates of a pixel in the transformed 2-D image $o_{t}^{2D}$, $k_1 = \lfloor j/h \rfloor$ and $k_2 = \text{modulo}(j/h)$ represent the horizontal and vertical coordinates of $o_t^{1D}$, respectively. $h$ represents the width of $o_t$, and $x(j)$ refers to the value of the $j$th point in the sample $x(t)$.

Therefore, by combining STFT, CWT, and TDC, we integrate their respective advantages, retain comprehensive signal information, and enhance feature diversity for improved bearing fault diagnosis. As shown in Fig. 3, the raw signals are divided into overlapping 1024-length segments, processed into 2-D grayscale images, and then concatenated into a three-channel input. Following previous studies [23], [43], the STFT sampling frequency $f$ is set to 12000 Hz, the window function $g(·)$ is defined as a *Hann* window with a window size of 64, and an overlap length of 32. Additionally, the CWT scale parameter $a$ is set to 1025, and the wavelet function $\psi(·)$ is set as the *Morlet* wavelet. The TDC method directly transforms raw vibration signals into 2-D images [44], with the resulting image width $o_t^{2D}$ set to 32.

---

## C. Pretraining

In the pretraining stage, we adopt the self-supervised learning framework BYOL to pretrain the feature extractor for learning domain-specific features for fault diagnosis. As shown in Fig. 4, BYOL consists of two branches: an online branch and a target branch, forming a student–teacher architecture, where ResNet50 is selected as the backbone for feature extraction. The online branch acts as the student and actively learns from the input data, while the target branch serves as the teacher and provides stable supervision signals. The online network is optimized by maximizing the consistency between its output and that of the target network. The parameters of the target branch are updated via exponential moving average (EMA) from the online branch, rather than through gradient backpropagation. This update mechanism stabilizes target representations, ensuring consistent optimization and preventing model collapse.

After the pretraining, the feature extractor is frozen for the subsequent fine-tuning stage.

Specifically, input data undergoes image augmentations to generate two views $v$ and $v'$. Both views are sequentially passed through the feature extractor and a projector (implemented as an MLP). View $v$ is processed by the online network, producing representations $y$ and $z$, while view $v'$ is processed by the target network, generating corresponding representations $y'$ and $z'$. The online network further processes $z$ through a predictor (also an MLP), yielding a prediction $q_u(z)$. The similarity loss between $q_u(z)$ and $sg(z')$, where $sg(·)$ denotes the stop-gradient operator, is computed to train the model, as defined

$$\mathcal{L} = \left\| \frac{q_u(z)}{\|q_u(z)\|_2} - z' \right\|_2^2 = 2 - 2 \cdot \frac{\langle q_u(z, z') \rangle}{\|q_u(z)\|_2 \cdot \|z'\|_2} \quad (4)$$

$$\mathcal{L}^{\text{BYOL}} = \mathcal{L} + \bar{\mathcal{L}} \quad (5)$$

where $\bar{\mathcal{L}}$ is computed by feeding $v'$ to the online network and $v$ to the target network, $\langle \cdot \rangle$ represents the dot product of two vectors, and $\| \cdot \|_2$ is the L2-norm.

During the training process, the parameters $\theta$ of the online network are updated iteratively via gradient descent. The parameters $\xi$ of the target network are updated using EMAs of the parameters $\theta$, defined as

$$\xi \leftarrow \tau\xi + (1 - \tau)\theta \quad (6)$$

where $\eta$ is the learning rate and $\tau \in [0, 1]$ is the target decay rate.

---

## D. Fine-Tuning

We train both the closed-set classifier and the open-set classifier using a small number of labeled samples combined with a large volume of unlabeled data, which includes both known and unknown classes. As illustrated in Fig. 4, weak augmentation $T_w(·)$ is applied to the labeled data $x_l^i$ before passing it through the pretrained feature extractor $f(·)$. The extracted features $h_i^l = f(T_w(x_i^l))$ are fed into both the closed-set classifier $C(·)$ and multibinary classifier $M_k(·)$, which are trained in a supervised manner using the losses $\mathcal{L}_s$ and $\mathcal{L}_{mb}$, defined as

$$\mathcal{L}_s(X^l) = \frac{1}{B} \sum_{i=1}^{B} H(y_i, p_i) \quad (7)$$

$$\mathcal{L}_{mb}(X^l) = \frac{1}{B} \sum_{i=1}^{B} \left( -\log(o_{i,i}) - \min_{k=y_i} \log\left(1 - o_{i,k}\right) \right) \quad (8)$$

where $y_i$ is the ground truth, $B$ is the number of labeled samples, $H(·)$ denotes the cross-entropy loss, $p_i = C(h_i^l)$ is the output of closed-set classifier, and $o_{i,k} = M_k(g(h_i^l))$ is the output of the $k$th binary classifier, with $g(·)$ as the projection head.

For unlabeled data $x_i^u$, $T_s(·)$ are applied to obtain $h_i^u = f(T_s(x_i^u))$, which is then fed into the closed-set and multibinary classifiers to generate pseudo-labels $\tilde{q}_{i,k}$, defined as

$$\tilde{q}_{i,k} = \begin{cases}
\tilde{p}_{i,k} \cdot o_{i,k}^{u} & \text{if } 1 \leq k \leq K \\
1 - \sum_{k=1}^{K} \tilde{p}_{i,j} \cdot o_{i,k}^{u} & \text{if } k = K + 1
\end{cases} \quad (9)$$

where $\tilde{p}_i = C(h_i^u)$, $o_i^u = M_k(g(h_i^u))$ and $K$ is the number of known classes. The probability that $x_i^u$ belongs to a known class is given by $\tilde{q}_{i,k} = \tilde{p}_{i,k} \cdot o_{i,k}^u$. Thus, the probability that $x_i^u$ belongs to an unknown class is $1 - \sum_{k=1}^{K} \tilde{p}_{i,j} \cdot o_{i,k}^u$.

For open-set recognition, strong augmentation $T_s(·)$ is applied to $x_i^u$ to obtain $h_i^u = f(T_s(x_i^u))$, which is then fed into the open-set classifier $O(·)$. The pseudo-labels $\tilde{q}_{i,k}$ serve as the supervision signals for training open-set classifier, and the optimization objective $\mathcal{L}_{op}$ is defined as

$$\mathcal{L}_{op}(U) = \frac{1}{uB} \sum_{i=1}^{uB} \mathbb{1}\left(\max(\tilde{q}_{i,k}) > \tau_q\right) \cdot H(\tilde{q}_{i,k}, q_i^l) \quad (10)$$

where $q_i^s = O(g(h_i^u))$, $uB$ is the number of unlabeled samples, $\mathbb{1}(·)$ is the indicator function, and $\tau_q$ is a confidence threshold. While optimizing the open-set classifier, the projection head $g(·)$ is fine-tuned to generate more discriminative features, further enhancing the multibinary classifier's performance. Meanwhile, a dual-filtering strategy selects high-quality pseudo-labels for known classes, optimizing the closed-set classifier by minimizing $\mathcal{L}_{ui}$, and the overall optimization objective $\mathcal{L}_{\text{overall}}$ is defined as

$$\mathcal{L}_{ui}(U) = \frac{1}{uB} \sum_{i=1}^{uB} \mathcal{F}(x_i^u) \cdot H(\bar{p}_i, p_i^l) \quad (11)$$

$$\mathcal{L}_{\text{overall}} = \mathcal{L}_s + \lambda_{mb}\mathcal{L}_{mb} + \lambda_{op}\mathcal{L}_{op} + \lambda_{ui}\mathcal{L}_{ui} \quad (12)$$

where $\bar{p}_i = C(h_i^u)$, $\mathcal{F}(·)$ is the filtering function, $\mathcal{F}(x_i^u) = \mathbb{1}\left(\max(\bar{p}_{i,k}) > \tau_p\right) \cdot \mathbb{1}\left(\left|1 - \sum_{j=1}^{K}\bar{p}_{i,j} \cdot o_{i,k}^u\right| < 0.5\right)$ where the first term ensures the predicted confidence exceeds the confidence threshold $\tau_p$ and the second term guarantees the trustworthiness of closed-set pseudo-labels. $\lambda_{mb}$, $\lambda_{op}$, and $\lambda_{ui}$ are the weights of the corresponding losses.

---

## Algorithm 1: Optimize OSCL Fine-Tuning in Each Iteration

**Input:** $\{(x_i^l, y_i)\}_{i=1}^{B}$ and $\{x_i^u\}_{i=1}^{uB}$

1. Calculate closed-set supervised loss $\mathcal{L}_s$ using Eq. 7.
2. Calculate multibinary supervised loss $\mathcal{L}_{mb}$ using Eq. 8.
3. Generate pseudo-labels $\tilde{q}_{i,k}$ using Eq. 9.
4. Calculate open-set loss $\mathcal{L}_{op}$ using Eq. 10, where $\tilde{q}_{i,k}$ serves as supervision.
5. Calculate $\mathcal{L}_{ui}$ using Eq. 11 to further optimize closed-set classifier.

**Output:** Calculate $\mathcal{L}_{\text{overall}}$ using Eq. 12 to update the OSCL fine-tuning network parameters.

---

As shown in Fig. 4, $\mathcal{L}_s$ and $\mathcal{L}_{mb}$ optimize the closed-set and multibinary classifiers, respectively, through supervised learning with ground-truth labels. $\mathcal{L}_{op}$ is employed to optimize the open-set classifier using generated open-set pseudo-labels, while $\mathcal{L}_{ui}$ further improves the performance of the closed-set classifier by leveraging filtered high-quality known class pseudo-labels. All three classifiers are updated concurrently during training. In Algorithm 1, we present the detailed optimization procedure for each training iteration. Unlike traditional open-set SSL methods, OSCL performs open-set training in a single stage, updating all network parameters simultaneously. Optimizing the open-set classifier also enhances the closed-set classifier, enabling effective handling of both tasks in bearing fault diagnosis.


# Final Methodology Structure (§III)

| Subsection | Status | Key Additions |
|---|---|---|
| **3.1 Overview** | ✨ New | Pipeline formalization with operator notation $\mathcal{T}_{\mathrm{Mel}}$, $\mathcal{T}_{\mathrm{GADF}}$, $\mathcal{F}_{\mathrm{AW}}$, $\mathcal{C}$; two-stage description with equations |
| **3.2 Acoustic Signal Rep.** | 🔧 Expanded | Mel: $n_{\mathrm{fft}}=2048$, hop=256, $n_{\mathrm{mels}}=128$, $f_{\mathrm{max}}=8000$ Hz, Viridis colormap, $224\times224$; GADF: sequence length $n=224$, difference variant, Viridis colormap |
| **3.3 AW-DPCNN** | 🔧 Expanded | Three innovations enumerated (dual-channel stimuli, contrast-guided weighting, iterative dynamics); explicit ablation reference |
| **3.4 MSCA-VGG16** | 🔧 Expanded | Three design principles; VGG16-BN justification with citation; $r=16$ SE reduction; $3\times3$, $5\times5$, dilated $3\times3$ (dilation=2) convs; $512\to1024\to256$ dimensions; dropout=0.5 |
| **3.5 Training Strategy** | ✨ New | AdamW ($\eta=10^{-4}$, $\lambda=10^{-3}$); ReduceLROnPlateau (factor=0.5, patience=5); class-weighted CE loss with formula; augmentation (flip, $\pm10^\circ$ rotation); ImageNet norm; early stopping (patience=15, monitor val-F1); 3 trials (seeds 42,123,456); PyTorch 2.11 + `torch.compile` |

### Total New Content Added
- **~120 lines** of new LaTeX across two new subsections (Overview + Training Strategy)
- **~40 lines** of parameter-specific enhancements across Mel, GADF, AW-DPCNN, and MSCA-VGG16 subsections
- All parameter values verified against default.yaml, build_cwru_dataset.py, and MSCA_VGG16.py

Made changes.