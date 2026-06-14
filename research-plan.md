# Research plan
For the project `AW-DPCNN`, the research plan is divided into two main sections: supervised learning and unspervised learning. Each section will outline the steps and methodologies to be followed for dataset preparation, model training, and evaluation.

## for supervised learning

### CWRU Dataset Processing Strategy

Reviewing and revising the script `build_fused_dataset.py`, the script is responsible for building the fused dataset by combining the Mel spectrogram and GADF image using AW-DPCNN. The script likely performs the following steps:
1. Add `.mat` reading support directly to `build_fused_dataset.py`
2. Preprocess the data: Split at file level rather than segment level (e.g., 2 files train / 1 val / 1 test per class)
3. For each .mat file:
   a. Load `DE_time` signal by searching for keys containing `DE_time`
   b. Sliding-window segmentation, and segment each split independently (sampling rate: 12 kHz, window length: 2048 samples, hop length: 1024 samples)
   c. For each window:
      - Generate Mel spectrogram image, the parameters:
        - n_fft: 1024
        - n_mels: 128
        - fmax: 6000 Hz
        - image size: `224*224`
      - Generate GADF image, the parameters are as follows:
        - method: `difference`
        - image size: `224*224`
        - normalization range: `[-1, 1]`
        - colormap: `viridis`
      - Fuse via AW-DPCNN (γ=4, N=20)
      - Save as .png in split-appropriate class subfolder
4. Save `metadata.csv` (filename, class, source_file, window_idx, split)
5. Verify: per-class segment counts, JS divergence, feature-space MMD
6. The output is an ImageFolder-compatible directory tree (no further splitting needed)


The Normal (N) class has ~3.5× more windows than the fault classes because its recordings are longer. This is real-world class imbalance — your existing class-weighted loss in the training pipeline should handle it, but be aware of it when interpreting per-class metrics.
