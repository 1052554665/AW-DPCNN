>add trial seed selection to the `plot_tsne_all.py` script.


```bash
python scripts/plot_tsne_all.py --mode models --trial trial_seed456
python scripts/plot_tsne_all.py --mode models --trial trial_seed42   # default
```

Changes made:
- Renamed `TRIAL` → `DEFAULT_TRIAL` (default `"trial_seed42"`)
- `find_model_runs(trial=...)` — accepts trial name instead of using global
- `generate_model_tsne(..., trial=...)` — passes trial through
- `--trial` CLI argument added with help text showing the default