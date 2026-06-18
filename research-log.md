
## CNN-LSTM Hybrid Model (2026-06-18)

Created `src/models/cnn_lstm.py` — a lightweight CNN‑BiLSTM hybrid for spectrogram‑based fault diagnosis.

### Architecture
```
Input [B, 3, 224, 224]
  │
  ├─ Frequency‑aware stem (7×1 + 3×3 conv, 32 ch)
  ├─ Block2 (32→64, /2) ── 2× Conv-BN-ReLU + MaxPool
  ├─ Block3 (64→128, /2) ── 2× Conv-BN-ReLU + MaxPool
  ├─ Block4 (128→128, /2) ── 2× Conv-BN-ReLU + MaxPool
  │                           → [B, 128, 14, 14]
  ├─ Temporal pool (mean over freq axis) → [B, 128, 14]
  ├─ Permute → [B, 14, 128]
  ├─ BiLSTM (hidden=256, 1 layer) → [B, 14, 512]
  ├─ Last timestep → [B, 512]
  └─ FC (512→128) + Dropout + FC (128→num_classes)
```

### Key specs
| Property | Value |
|----------|-------|
| Parameters | **1.44M** (lightweight) |
| CNN channels | 32→64→128→128 |
| LSTM | 1‑layer BiLSTM, hidden=256 |
| Bidirectional output | 512 dim |
| Weight init | Kaiming (CNN) + default (LSTM) |

### Design rationale
- CNN extracts local spectral features per time column
- Average-pooling over frequency axis preserves temporal structure
- BiLSTM explicitly models time‑axis dependencies — physically meaningful for
  vibration/acoustic signals where harmonic patterns evolve over time
- Much lighter than VGG16 (134M) or ConvNeXt‑Tiny (27.8M)

### Files created
- `src/models/cnn_lstm.py` — model definition
- `experiments/exp1/cnn_lstm.yaml` — experiment config
- Model registered in `src/models/registry.py` as `cnn_lstm` / `cnn-lstm` / `cnnlstm`

### Usage
```bash
python scripts/train.py --config configs/default.yaml --exp-config experiments/exp1/cnn_lstm.yaml
```