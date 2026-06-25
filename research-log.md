for `run_rep_compare`, add new repeated Independent Trials Runner
====================================
Run 3 independent training trials with different random seeds and aggregate results into **mean ± std** format for publication.

This directly addresses the reviewer comment:
  "Run 3 independent runs with different random seeds;
   report mean ± std for all metrics."


### What changed in run_rep_compare.py

| Feature | Before | After |
|---|---|---|
| **Trial mode** | Single seed only (42) | Single (`--num-trials 1`) or repeated (`--num-trials 3`) |
| **Seeds** | Hardcoded 42 | `--num-trials N` auto-selects [42, 123, 456, …]; `--seeds` for explicit list |
| **Aggregation** | None | Per-combo `aggregated_metrics.{json,csv,md}` via `src.utils.aggregation` |
| **Cross-combo table** | None | `rep_compare_summary_mean_std.{json,csv,md}` comparing all 12 combos |
| **Error handling** | Stops on first failure | `--continue-on-error` to keep running |
| **Re-aggregate** | N/A | `--aggregate-only` to re-aggregate without re-training |

### Usage examples

```bash
# Single trial (current behavior, backward compatible)
python scripts/run_rep_compare.py --num-workers 32

# 3 independent trials with mean ± std aggregation (reviewer-ready)
python scripts/run_rep_compare.py --num-workers 32 --num-trials 3

# Single combo, 3 trials
python scripts/run_rep_compare.py --combo mel_gadf --num-workers 32 --num-trials 3

# Aggregate only (if trials already completed)
python scripts/run_rep_compare.py --aggregate-only --num-trials 3
```

### Output structure

```
experiments/experiment_result/rep_compare/
├── mel_gadf/
│   └── rep_compare_mel_gadf/
│       ├── trial_seed42/results/test_metrics.json
│       ├── trial_seed123/results/test_metrics.json
│       ├── trial_seed456/results/test_metrics.json
│       └── aggregated/
│           ├── aggregated_metrics.json   # {mean, std, min, max, trials}
│           ├── aggregated_metrics.csv
│           └── aggregated_metrics.md     # Publication-ready table
├── rep_compare_summary_mean_std.json     # All 12 combos side-by-side
├── rep_compare_summary_mean_std.csv
└── rep_compare_summary_mean_std.md       # IEEE-style comparison table
```

The cross-combo summary auto-generates a Markdown table with all 12 combos in **mean ± std** format across Acc, Prec, Rec, F1, G-mean, κ, and AUC — ready for direct insertion into the paper