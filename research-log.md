>add a function to support the specification of output and the dataset directory.

### train.py
```bash
--dataset {12k_de,12k_fe,48k_de}   # overrides dataset.root_dir
--output-root PATH                  # overrides output.root_dir
```

### run_repeated_trials.py
```bash
--dataset {12k_de,12k_fe,48k_de}   # passed through to train.py
--output-root PATH                  # passed through + used for aggregation dir naming
```

### Usage examples

```bash
# 3 trials on 48k DE, results saved to experiments/runs_48k/
python scripts/run_repeated_trials.py \
    --config configs/default.yaml \
    --exp-config experiments/exp1/MSCA_VGG16.yaml \
    --dataset 48k_de --output-root experiments/runs_48k \
    --num-runs 3

# Batch: all exp1 configs on all 3 datasets
python scripts/run_repeated_trials.py \
    --config configs/default.yaml \
    --exp-dir experiments/exp1 \
    --datasets 12k_de 12k_fe 48k_de \
    --num-runs 3

# Dry-run to preview commands
python scripts/run_repeated_trials.py \
    --config configs/default.yaml \
    --exp-config experiments/exp1/vgg16.yaml \
    --dataset 12k_fe --dry-run
```

Made changes.

>three dataset are located in `./datasets/cwru_de`, `./datasets/cwru_fe`, and `./datasets/cwru_de_48k`. The default dataset is `12k_de` (i.e., `./datasets/cwru_de`).


>create a demo to plot waveform of ten classes, the raw data is located in `raw-data/CWRU-dataset/12k_Drive_End_Bearing_Fault_Data`, and save the ten plots separately.
