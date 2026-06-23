For the CWRU dataset `raw-data/CWRU-dataset`, there are three types of fault data, i.e., 12kHz drive end bearing fault data, 12kHz fan end bearing fault data, and 48kHz drive end bearing fault data.
the dataset includes three fault types: ball faults, inner race faults, and outer race faults, each with three defect sizes: 0.007, 0.014, and 0.021 inches. In total, the dataset consists of ten states: one normal state and nine fault states.

Creating three scripts to make the dataset. Selecting 12kHz drive end bearing fault data, 12kHz fan end bearing fault data, and 48kHz drive end bearing fault data, separately. Referencing `build_cwru_dataset.py`. The fault types and sizes are as follows:

each dataset includes ten types: BF007, IF007, and OF007; BF014, IF014, and OF014; BF021, IF021, and OF021; and Normal.


### Scripts Created

| Script | Data Source | SR | Key Pattern | Files | Classes |
|---|---|---|---|---|---|
| build_cwru_de.py | 12kHz Drive End | 12 kHz | `DE_time` | 40 | 10/10 ✓ |
| build_cwru_fe.py | 12kHz Fan End | 12 kHz | `FE_time` | 34 | 10/10 ⚠️ |
| build_cwru_48k_de.py | 48kHz Drive End | 48 kHz | `DE_time` | 40 | 10/10 ✓ |

### 10-Class Mapping

| Fault Type | 0.007" | 0.014" | 0.021" |
|---|---|---|---|
| Ball | BF007 | BF014 | BF021 |
| Inner Race | IF007 | IF014 | IF021 |
| Outer Race (@6) | OF007 | OF014 | OF021 |
| Normal | — | — | — |

### Usage

```bash
# 12kHz Drive End
python scripts/build_cwru_de.py --output-dir ./datasets/cwru_de \
    --file-split 60,20,20 --metadata --verify --workers 16

# 12kHz Fan End
python scripts/build_cwru_fe.py --output-dir ./datasets/cwru_fe \
    --file-split 60,20,20 --metadata --verify --workers 16

# 48kHz Drive End (scaled window/FFT params for 4× sample rate)
python scripts/build_cwru_48k_de.py --output-dir ./datasets/cwru_48k_de \
    --file-split 60,20,20 --metadata --verify --workers 16
```

### Key Design Decisions

- **OR load position**: defaults to `@6` (6 o'clock, centered in load zone) — the CWRU benchmark standard
- **48kHz defaults**: window=8192, hop=4096, n_fft=4096, fmax=20000 (scaled 4× vs 12kHz defaults)
- **Normal data**: shared from `Normal/` directory; for 48kHz script, normal files use 12kHz (no 48kHz normal baseline exists)
- **Fan-end limitation**: OF014 and OF021 have only 1 file each in the FE dataset (known CWRU limitation)

Made changes.



