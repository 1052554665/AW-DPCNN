>Review and revise script `build_ablation_datasets.py` and `run_ablation_experiments`, redesign the ablation study. Both AW-DPCNN and MSCA-VGG16 should be included in the ablation study. The ablation study should include the following components:
- AW-DPCNN components
- MSCA-VGG16 components
The data is located in the `raw-data/CWRU-dataset/12k_Drive-End-Bearing-Dataset` directory. The ablation study should be designed to evaluate the impact of different components on the performance of the models. The results should be saved in a structured format for further analysis.

### Ablation Matrix (9 experiments, all on CWRU 12k DE)

| Exp | Dataset | Model | AW-DPCNN | MS | CA | EH |
|---|---|---|---|---|---|---|
| **B0** | mel_only | VGG16 | ✗ | ✗ | ✗ | ✗ |
| **B1** | gadf_only | VGG16 | ✗ | ✗ | ✗ | ✗ |
| **B2** | concat | VGG16 | ✗ (naive avg) | ✗ | ✗ | ✗ |
| **B3** | awdpcnn_gamma1 | VGG16 | γ=1 | ✗ | ✗ | ✗ |
| **B4** | awdpcnn_full | VGG16 | γ=10 ✓ | ✗ | ✗ | ✗ |
| **B5** | awdpcnn_full | MSCA-VGG16 | γ=10 ✓ | ✓ | ✗ | ✗ |
| **B6** | awdpcnn_full | MSCA-VGG16 | γ=10 ✓ | ✗ | ✓ | ✗ |
| **B7** | awdpcnn_full | MSCA-VGG16 | γ=10 ✓ | ✗ | ✗ | ✓ |
| **B8** | awdpcnn_full | MSCA-VGG16 | γ=10 ✓ | ✓ | ✓ | ✓ |

### Key changes

- **Data source**: 12k_Drive_End_Bearing_Fault_Data (40 `.mat` files, 10 classes)
- **Unified split**: All 5 datasets share the same file-level split (seed=42, 60/20/20)
- **Windows pre-segmented once**: Shared across all variants for efficiency
- **B4 dataset doubles as B5–B8 input**: No redundant builds

### Usage

```bash
# Step 1 — Build datasets (~10 min)
python scripts/build_ablation_datasets.py --workers 16

# Step 2 — Run all 9 experiments (~45 min)
python scripts/run_ablation_experiments.py --epochs 30

# Single experiments
python scripts/run_ablation_experiments.py --exp-ids B0,B4,B8
```

Made changes.
