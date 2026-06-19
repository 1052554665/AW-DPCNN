>how to ensure if there is a class imbalance and solve it?

>is there a way to control the GAF images generation? the parameter maybe called `sequence_length`? if so, add this parameter to script `build_group2_4_harmonic.py` and `build_ablation_datasets.py`

The `sequence_length` parameter controls how many time points from the signal are used before GAF computation — in `_time_series_to_image` this is currently hardcoded to `img_size * 4`, and in `generate_gadf_image` the full signal is used without resampling.

### Files modified

| File | Changes |
|---|---|
| representation_comparison.py | Added `sequence_length` to `_time_series_to_image` (default=`img_size * 4`) and `generate_gadf` |
| build_CWRU_dataset.py | Added `sequence_length` to `generate_gadf_image` (default=`None`=full signal), `process_one_window`, `_append_window_tasks`, `_collect_tasks`, and CLI |
| build_group2_4_harmonic.py | Added `--sequence-length` CLI arg, threaded through `_collect_window_tasks` → task tuples → `process_one_window` |
| build_ablation_datasets.py | Added `--sequence-length` CLI arg, threaded through all 3 `_build_*` / `_process_*` pipelines |

### Usage examples

```bash
# Harmonic dataset with GAF limited to 896 time-steps
python scripts/build_group2_4_harmonic.py --sequence-length 896

# Ablation datasets with GAF limited to 512 time-steps
python scripts/build_ablation_datasets.py --sequence-length 512 --workers 16
```

### Default behavior (backward compatible)

- **build_group2_4_harmonic.py** / **build_CWRU_dataset.py**: `None` → uses the full window signal (same as before)
- **`build_ablation_datasets.py`** (via representation_comparison.py): `None` → uses `img_size * 4` (same hardcoded `224*4=896` as before)