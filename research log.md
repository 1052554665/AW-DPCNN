# report mean ± std
>report **mean ± std** for all metrics.

##  Weakness 5 — Implemented

Here's a summary of all changes made to address the reviewer comment about repeated trials:

### Files Modified (4)

| File | Change |
|------|--------|
| train.py | Added `--seed` CLI argument; fixed `sys.path` for standalone execution; propagates seed to config & run_dir naming |
| experiment.py | **`set_seed()`** now enables cudnn determinism (`benchmark=False`, `deterministic=True`, `use_deterministic_algorithms`). Added **`seed_worker()`** for DataLoader. **`prepare_run_dir()`** uses deterministic `{exp_name}/trial_seed{seed}/` naming when seed is set |
| workflow.py | `train_and_evaluate()` accepts optional `seed` parameter and passes it to `build_dataloaders()` |
| image_classification.py | `build_dataloaders()` accepts optional `seed`; uses `torch.Generator` + `worker_init_fn=seed_worker` for reproducible train-set shuffling |

### Files Created (2)

| File | Purpose |
|------|---------|
| run_repeated_trials.py | Multi-run orchestrator — runs N independent trials with different seeds, then aggregates results |
| aggregation.py | Aggregation utilities — computes mean ± std (sample), min/max, outputs CSV + JSON + Markdown tables |

### Key Design Decisions

1. **3 independent runs** (not 5-fold CV) — simpler to implement on existing fixed splits; randomness comes from weight initialization + data augmentation
2. **Seeds**: `[42, 123, 456, 789, 1024]` by default, well-separated and reproducible
3. **Deterministic directory structure**: `{output.root_dir}/{exp_name}/trial_seed{seed}/` — no timestamp search needed
4. **Backward compatible**: single runs without `--seed` still use timestamped directories and non-deterministic mode

### Usage

```bash
# Single experiment, 3 runs (default)
python scripts/run_repeated_trials.py --exp-config experiments/exp1/vgg16.yaml

# 5 runs
python scripts/run_repeated_trials.py --exp-config experiments/exp1/MSCA_VGG16.yaml --num-runs 5

# Batch mode: all experiments in a directory
python scripts/run_repeated_trials.py --exp-dir experiments/exp1 --num-runs 3

# Re-aggregate without re-training
python scripts/run_repeated_trials.py --exp-config experiments/exp1/vgg16.yaml --aggregate-only
```

### Output Example (`aggregated_metrics.md`)

```
| Metric         | Mean ± Std      | Min    | Max    |
|----------------|-----------------|--------|--------|
| Accuracy       | 98.52 ± 0.34    | 98.10  | 98.85  |
| Precision      | 98.55 ± 0.31    | 98.18  | 98.82  |
| Recall         | 98.48 ± 0.36    | 98.05  | 98.80  |
| F1-score       | 98.50 ± 0.33    | 98.10  | 98.78  |
| G-Mean         | 98.47 ± 0.37    | 98.03  | 98.79  |
| ROC-AUC        | 99.82 ± 0.08    | 99.72  | 99.88  |
```