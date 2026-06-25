create scripts to build the following three datasets for representation comparison, 
- 12k_Drive_End_Bearing_Fault_Data
- 12k_Fan_End_Bearing_Fault_Data
- 48k_Drive_End_Bearing_Fault_Data

the following method for data selection can be refered


Builds AW-DPCNN fused images from the **12 kHz drive-end** bearing fault
data of the CWRU dataset.  The output is a 10‑class ImageFolder‑compatible
directory.

Target classes
--------------
  BF007   BF014   BF021      (ball faults    — 0.007", 0.014", 0.021")
  IF007   IF014   IF021      (inner race     — 0.007", 0.014", 0.021")
  OF007   OF014   OF021      (outer race @6  — 0.007", 0.014", 0.021")
  Normal

Data source
-----------
  raw-data/CWRU-dataset/12k_Drive_End_Bearing_Fault_Data/
    B/{007,014,021}/           (ball fault .mat files)
    IR/{007,014,021}/          (inner race .mat files)
    OR/{007,014,021}/@6/       (outer race @ 6 o'clock)
  raw-data/CWRU-dataset/Normal/ (normal baseline .mat files)


### Before → After

| Aspect | Before | After |
|---|---|---|
| Representations | Mel + GADF only | All 12: mel/stft/cwt × gadf/gasf/mtf/rp |
| Generators | Inline | Imported from representation_comparison.py |
| Output | `raw-data/cwru_rep_12k_de/{Class}/` | `raw-data/rep_compare_12k_de/{tf}_{temporal}/{Class}/` |
| Window segmentation | Per-combination | Once, shared across all combinations |

### Output structure

```
raw-data/rep_compare_12k_de/
├── mel_gadf/       ← 5,886 images verified
├── mel_gasf/
├── mel_mtf/
├── mel_rp/
├── stft_gadf/
├── stft_gasf/
├── stft_mtf/
├── stft_rp/
├── cwt_gadf/
├── cwt_gasf/
├── cwt_mtf/
└── cwt_rp/
    └── {BF007,...,Normal}/  +  metadata.csv
```

### Usage

```bash
# All 12 combos, one dataset
python scripts/build_cwru_rep_datasets.py --dataset 12k_de --workers 32

# Single combo
python scripts/build_cwru_rep_datasets.py --dataset 12k_fe --tf stft --temporal gasf

# All three datasets, all 12 combos (36 total)
python scripts/build_cwru_rep_datasets.py --all --workers 32
```