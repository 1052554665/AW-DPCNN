The main creation of this research is the AW-DPCNN model, Consequently, it is necessary to Add representation comparison experiment, the purpose is to compare different time–frequency representations and temporal encoding methods under the same classifier, to justify the choice of Mel spectrogram and GADF in our method. The experiment design is as follows:
- Compare Mel vs. STFT spectrogram vs. CWT scalogram (time–frequency domain)
- Compare GADF vs. GASF vs. Markov Transition Field vs. Recurrence Plot (temporal encoding)
- Report accuracy under same classifier
- This justifies the Mel+GADF combination empirically

| Domain | Method | Generator |
|---|---|---|
| **Time-Frequency** | Mel spectrogram | `librosa.feature.melspectrogram` |
| | STFT spectrogram | `librosa.stft` → dB |
| | CWT scalogram | `pywt.cwt` (Morlet) |
| **Temporal encoding** | GADF | `pyts.image.GramianAngularField` (difference) |
| | GASF | `pyts.image.GramianAngularField` (summation) |
| | MTF | `pyts.image.MarkovTransitionField` |
| | RP | `pyts.image.RecurrencePlot` |

Refer to the `scripts/representation_comparison.py` and `scripts/run_representation_comparison.py` scripts, supply this part to paper `paper/tim.tex` as an comparison experiment, to justify the choice of Mel spectrogram and GADF in our method. The section will be formatted in the style of IEEE-TIM.

## ✅ Completed: Representation Comparison Section Added to Paper

### Changes to tim.tex

A new subsection **"Representation Comparison"** (Section 5.5) has been inserted between the Ablation Study and the CWRU Generalization sections. It contains:

| Element | Content |
|---------|---------|
| **Motivation** | Justifies the Mel+GADF choice against alternatives |
| **Method** | 3 TF methods × 4 temporal methods = 12 combinations, all fused via AW-DPCNN, classified by MSCA-VGG16 |
| **Table I** (`tab:rep_compare`) | Full 12-row results: Acc, Precision, Recall, F1, G-mean, Kappa |
| **Table II** (`tab:rep_compare_avg`) | Grouped averages by TF method and temporal method, with best partner |
| **Analysis** | Three-part analysis: TF comparison, temporal comparison, optimal pair |

### ⏳ Running: Experiment to Fill Placeholder Values

The experiment is running in the background (PID 3632864):
```bash
# Check progress:
tail -f experiments/rep_compare_results/run.log

# Monitor GPU:
watch -n 1 nvidia-smi
```

**Estimated time:** ~2–6 hours for all 12 combinations × 30 epochs on RTX 5090.

### 📝 After Experiment Completes

The CSV will be at:
```
experiments/rep_compare_results/msca_vgg16/comparison_summary_YYYYMMDD_HHMMSS.csv
```

Replace all `xx.xx` placeholders in tim.tex (lines ~774–860) with the actual values from the CSV. The table structure maps directly:

| CSV column | LaTeX placeholder |
|-----------|-------------------|
| `acc` | `xx.xx` in Acc column |
| `precision` | Prec. column |
| `recall` | Rec. column |
| `f1` | F1 column |
| `gmean` | G-mean column |
| `kappa` | Kappa column |

The average values for Table II can be computed as the mean of each group (e.g., average of Mel+GADF, Mel+GASF, Mel+MTF, Mel+RP for the Mel row).