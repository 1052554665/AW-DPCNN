"""
CNN-LSTM Hybrid Model
======================
A lightweight spectrogram-native CNN‑LSTM for fault diagnosis.

Design rationale
----------------
- **CNN stem** (frequency‑aware 7×1 conv) → extracts local spectral features
  from each time column of the spectrogram.
- **CNN backbone** → progressively abstracts spatial‑frequency patterns
  (3×3 convs, BatchNorm, ReLU, MaxPool).
- **Temporal pooling** → average‑pools the frequency axis, treating each
  remaining spatial column as a time step → sequence of feature vectors.
- **BiLSTM** → models long‑range temporal dependencies across the sequence.
- **Classifier head** → FC layers for final prediction.

Why this matters for spectrograms
----------------------------------
A plain CNN treats the spectrogram as a spatial image and loses explicit
temporal ordering. The LSTM explicitly models time‑axis dependencies
(e.g. how harmonic patterns evolve), which is physically meaningful for
vibration/acoustic signals.

Parameter budget
-----------------
~1.1M parameters — comparable to HarmonicCNN (~1.2M), much lighter than
VGG16 (134M) or ConvNeXt‑Tiny (27.8M).

Input
-----
[B, 3, H, W]  AW‑DPCNN fused RGB images (typically 224×224)

Output
------
[B, num_classes] logits
"""

import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    """CNN‑LSTM hybrid for spectrogram‑based fault classification.

    Args:
        num_classes:      number of output classes.
        in_channels:      input image channels (default 3 for RGB).
        cnn_out_channels: output channels of the CNN backbone (default 128).
        lstm_hidden:      LSTM hidden state size (default 256).
        lstm_layers:      number of stacked LSTM layers (default 1).
        bidirectional:    use bidirectional LSTM (default True).
        dropout:          dropout rate in classifier and LSTM (default 0.5).
    """

    def __init__(
        self,
        num_classes: int = 9,
        in_channels: int = 3,
        cnn_out_channels: int = 128,
        lstm_hidden: int = 256,
        lstm_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.5,
    ):
        super().__init__()

        # ═══════════════════════════════════════════════════════════════
        #  Frequency‑aware stem
        # ═══════════════════════════════════════════════════════════════
        # 7×1 conv scans along the frequency axis to detect narrow harmonic
        # bands BEFORE spatial mixing.  Followed by a 3×3 conv for local
        # spatial‑frequency refinement.
        self.stem_freq = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(7, 1), padding=(3, 0), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.stem_spatial = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2)  # 32 × 112 × 112

        # ═══════════════════════════════════════════════════════════════
        #  CNN backbone
        # ═══════════════════════════════════════════════════════════════
        # Block 2:  32 →  64,   112 → 56
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2)  # 64 × 56 × 56

        # Block 3:  64 → 128,   56 → 28
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, cnn_out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cnn_out_channels),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(2)  # cnn_out × 28 × 28

        # Block 4:  keep → cnn_out,   28 → 14
        self.block4 = nn.Sequential(
            nn.Conv2d(cnn_out_channels, cnn_out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cnn_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(cnn_out_channels, cnn_out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cnn_out_channels),
            nn.ReLU(inplace=True),
        )
        self.pool4 = nn.MaxPool2d(2)  # cnn_out × 14 × 14

        # ═══════════════════════════════════════════════════════════════
        #  BiLSTM temporal encoder
        # ═══════════════════════════════════════════════════════════════
        # Input:  [B, W', cnn_out_channels] = [B, 14, 128]
        # Output: [B, W', lstm_hidden * (2 if bidirectional else 1)]
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        lstm_out_dim = lstm_hidden * 2 if bidirectional else lstm_hidden

        # ═══════════════════════════════════════════════════════════════
        #  Classifier head
        # ═══════════════════════════════════════════════════════════════
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        # LSTM weights use default PyTorch init (uniform), which is fine.

    def forward(self, x: torch.Tensor, return_feat: bool = False):
        """Forward pass.

        Args:
            x:           input tensor  [B, C, H, W]
            return_feat: if True, return (features, logits) for t‑SNE.

        Returns:
            logits, or (features, logits) if return_feat=True.
        """
        # ── Stem ──
        x = self.stem_freq(x)
        x = self.stem_spatial(x)
        x = self.pool1(x)          # [B, 32, 112, 112]

        # ── CNN backbone ──
        x = self.block2(x)
        x = self.pool2(x)          # [B, 64, 56, 56]

        x = self.block3(x)
        x = self.pool3(x)          # [B, cnn_out, 28, 28]

        x = self.block4(x)
        x = self.pool4(x)          # [B, cnn_out, 14, 14]

        # ── Temporal pooling ──
        # Average-pool over frequency axis (H), keep time axis (W).
        # [B, C, H, W] → mean(dim=2) → [B, C, W]
        x = x.mean(dim=2)          # [B, cnn_out, 14]

        # Permute to sequence-first: [B, W, C] = [B, 14, cnn_out]
        x = x.permute(0, 2, 1)

        # ── BiLSTM ──
        lstm_out, _ = self.lstm(x)         # [B, 14, lstm_out_dim]

        # Use the last time step's output (contains both directions)
        feat = lstm_out[:, -1, :]          # [B, lstm_out_dim]

        # ── Classifier ──
        out = self.classifier(feat)        # [B, num_classes]

        if return_feat:
            return feat, out
        return out


def build_cnn_lstm(
    num_classes: int = 9,
    in_channels: int = 3,
    cnn_out_channels: int = 128,
    lstm_hidden: int = 256,
    lstm_layers: int = 1,
    bidirectional: bool = True,
    dropout: float = 0.5,
):
    """Factory function for CNNLSTM."""
    return CNNLSTM(
        num_classes=num_classes,
        in_channels=in_channels,
        cnn_out_channels=cnn_out_channels,
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
        bidirectional=bidirectional,
        dropout=dropout,
    )
