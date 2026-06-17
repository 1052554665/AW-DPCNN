"""
HarmonicCNN — A lightweight spectrogram-native CNN for harmonic fault classification.

Design principles (cf. ConvNeXT-Tiny mismatch analysis):
  - ~1.2M parameters (vs 27.8M) → 23× fewer, resists file-level memorization
  - 3×3 kernels throughout → preserves fine frequency-local features
  - BatchNorm → respects per-frequency-bin statistics (unlike LayerNorm)
  - ReLU → sparse activations preserve energy peaks (unlike GELU)
  - Frequency-aware first layer → 7×1 conv along frequency axis to
    explicitly model harmonic bands before spatial mixing
  - No ImageNet pretraining → learns spectrogram-native features from scratch
  - Progressive channel expansion: 32→64→128→256

Expected parameter count: ~1.17M
"""

import torch
import torch.nn as nn


class HarmonicCNN(nn.Module):
    """Lightweight spectrogram-native CNN optimized for harmonic classification.

    Input:  [B, 3, H, W]   (AW-DPCNN fused RGB images, typically 224×224)
    Output: [B, num_classes] logits
    """

    def __init__(self, num_classes: int = 9, in_channels: int = 3, dropout: float = 0.5):
        super().__init__()

        # ── Block 1: Frequency-aware stem ──
        # 7×1 kernel scans along the frequency axis to detect narrow harmonic
        # bands BEFORE mixing spatially.  Followed by a 3×3 conv for local
        # spatial-frequency refinement.
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

        # ── Block 2 ──
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2)  # 64 × 56 × 56

        # ── Block 3 ──
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(2)  # 128 × 28 × 28

        # ── Block 4 ──
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.pool4 = nn.MaxPool2d(2)  # 256 × 14 × 14

        # ── Global pooling & classifier ──
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 256 × 1 × 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, num_classes)

        # Weight init
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

    def forward(self, x: torch.Tensor, return_feat: bool = False):
        """Forward pass.

        Args:
            x: input tensor [B, 3, H, W]
            return_feat: if True, return (features, logits) for t-SNE.
        """
        # Block 1 — frequency-aware
        x = self.stem_freq(x)
        x = self.stem_spatial(x)
        x = self.pool1(x)

        # Block 2
        x = self.block2(x)
        x = self.pool2(x)

        # Block 3
        x = self.block3(x)
        x = self.pool3(x)

        # Block 4
        x = self.block4(x)
        x = self.pool4(x)

        # Global pooling + classifier
        x = self.global_pool(x)
        feat = torch.flatten(x, 1)
        feat = self.dropout(feat)
        out = self.fc(feat)

        if return_feat:
            return feat, out
        return out


def build_harmonic_cnn(num_classes: int = 9, in_channels: int = 3, dropout: float = 0.5):
    """Factory function for HarmonicCNN."""
    return HarmonicCNN(num_classes=num_classes, in_channels=in_channels, dropout=dropout)
