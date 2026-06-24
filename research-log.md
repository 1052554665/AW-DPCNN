create a LaTeX table from the noise robustness CSV data for inclusion in the paper.

### Key Results

| Model | Clean | 5 dB | 10 dB | 15 dB+ |
|---|---|---|---|---|
| **MSCA-VGG16** | 95.1 | **92.9** | 93.9 | ~95 |
| VGG16 | 94.7 | 83.5 | 93.7 | ~95 |
| EfficientNet-B0 | 91.0 | **14.1** | 75.3 | ~91 |
| MobileNetV3 | 84.7 | 56.9 | 72.6 | ~84 |

- **MSCA-VGG16 drops only ~2% at 5 dB** — best noise resilience
- **VGG16 drops ~11% at 5 dB** — the MSCA enhancements provide ~9 pp noise robustness gain over the baseline
- **EfficientNet-B0 collapses at 5 dB** (14.1%) — extremely noise-sensitive
- **ConvNeXt-Tiny** is noise-indifferent but consistently poor (76-78%)