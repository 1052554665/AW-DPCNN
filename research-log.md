remove the following model and relative configuration files.
- cnn_lstm
- harmonic_cnn


## Removed Files

| File | Description |
|------|-------------|
| `src/models/cnn_lstm.py` | CNN+LSTM hybrid model implementation |
| `src/models/harmonic_cnn.py` | Harmonic CNN model implementation |
| `experiments/exp1/cnn_lstm.yaml` | Experiment config |
| `experiments/exp1/harmonic_cnn.yaml` | Experiment config |

## Updated Files

| File | Change |
|------|--------|
| registry.py | Removed 2 imports + 2 registration blocks (lines 4, 6, 31–48) |
| run_representation_comparison.py | Removed from docstring model list |

Verified: `from src.models import build_model` imports cleanly with no errors. No other files in the codebase reference these models.

Made changes.