# representation comparison experiment

>Add **representation comparison experiment**, the purpose is to compare different time–frequency representations and temporal encoding methods under the same classifier, to justify the choice of Mel spectrogram and GADF in our method. The experiment design is as follows:
  - Compare Mel vs. STFT spectrogram vs. CWT scalogram (time–frequency domain)
  - Compare GADF vs. GASF vs. Markov Transition Field vs. Recurrence Plot (temporal encoding)
  - Report accuracy under same classifier
  - This justifies the Mel+GADF combination empirically


## Script: representation_comparison.py

### What it does
Generates parallel datasets for all 12 combinations of **3 time-frequency × 4 temporal encoding** methods, using the **same file-level split** for strictly fair comparison. Each combination is an ImageFolder-compatible dataset ready for training.

### Representation methods

| Domain | Method | Generator |
|---|---|---|
| **Time-Frequency** | Mel spectrogram | `librosa.feature.melspectrogram` |
| | STFT spectrogram | `librosa.stft` → dB |
| | CWT scalogram | `pywt.cwt` (Morlet) |
| **Temporal encoding** | GADF | `pyts.image.GramianAngularField` (difference) |
| | GASF | `pyts.image.GramianAngularField` (summation) |
| | MTF | `pyts.image.MarkovTransitionField` |
| | RP | `pyts.image.RecurrencePlot` |

### Output structure
```
datasets/rep_compare/
├── mel_gadf/      mel_gasf/      mel_mtf/      mel_rp/
├── stft_gadf/     stft_gasf/     stft_mtf/     stft_rp/
├── cwt_gadf/      cwt_gasf/      cwt_mtf/      cwt_rp/
    └── {train,val,test}/{Class}/*.png  +  metadata.csv
```

### Usage

```bash
# All 12 combinations (5,859 images each = 70,308 total)
python scripts/representation_comparison.py --workers 16

# Single combination for quick test
python scripts/representation_comparison.py --tf stft --temporal gasf --workers 8

# Dry-run to see the plan
python scripts/representation_comparison.py --dry-run
```

### How to train for comparison
Point `dataset.root_dir` in the config to each variant, e.g.:
```yaml
dataset:
  root_dir: ./datasets/rep_compare/stft_gasf
```
Then run `python scripts/train.py     --config configs/default.yaml     --exp-config experiments/exp1/VGG16.yaml`

The existing transformer-five already serves as the **Mel × GADF** reference.