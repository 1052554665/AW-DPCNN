import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


class MSCA_Block(nn.Module):
    """Multi-scale convolution + channel attention block.

    Adapted from MSCA_VGG16 — uses the same multi-branch design
    with configurable multi-scale convolutions and SE-style channel
    attention, but generalized for any number of input channels.

    Args:
        in_channels: number of input channels.
        use_ms:      enable multi-scale conv branches (3×3, 5×5, dilated 3×3).
                     If False, falls back to a single 3×3 conv.
        use_ca:      enable SE-style channel attention (reduction=16).
    """

    def __init__(self, in_channels: int, use_ms: bool = True, use_ca: bool = True):
        super().__init__()
        self.use_ms = use_ms
        self.use_ca = use_ca

        if use_ms:
            self.conv3 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
            self.conv5 = nn.Conv2d(in_channels, in_channels, 5, padding=2)
            self.conv_dilated = nn.Conv2d(in_channels, in_channels, 3, padding=2, dilation=2)
        else:
            self.conv_single = nn.Conv2d(in_channels, in_channels, 3, padding=1)

        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

        if use_ca:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, max(in_channels // 16, 8), 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(max(in_channels // 16, 8), in_channels, 1),
                nn.Sigmoid(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_ms:
            out = self.conv3(x) + self.conv5(x) + self.conv_dilated(x)
        else:
            out = self.conv_single(x)
        out = self.bn(out)
        out = self.relu(out)
        if self.use_ca:
            out = out * self.se(out)
        return out


class MSCA_ResNet50(nn.Module):
    """MSCA-enhanced ResNet50 for fused acoustic representation classification.

    Replaces the VGG16 backbone of MSCA-VGG16 with ResNet50 to leverage
    residual learning and deeper feature hierarchies.  The multi-scale
    convolution block and SE-style channel attention are retained and
    adapted to the ResNet50 output channel dimensionality (2048-d).

    Architecture overview
    ---------------------
    ::

        Input (224×224×3)
          │
          ▼
        ResNet50 conv layers (stem + layer1–4)
          │  2048 × 7 × 7
          ▼
        MSCA Block  ── multi-scale convs (3×3, 5×5, dil. 3×3)
          │            + SE channel attention (r=16)
          │  2048 × 7 × 7
          ▼
        AdaptiveAvgPool2d  ── 2048 × 1 × 1
          │
          ▼
        FC1  ── 2048 → 1024, ReLU, Dropout(0.5)
          │
          ▼
        Embedding Head  ── 1024 → 256, BN, ReLU, Dropout(0.5)
          │
          ▼
        Classifier  ── 256 → num_classes

    Args:
        num_classes: number of output classes.
        pretrained:  load ImageNet-1K pre-trained weights (default True).
        embed_dim:   embedding dimension (default 256).
        dropout:     dropout rate in fc / embed layers (default 0.5).
        use_ms:      enable multi-scale convolution (default True).
        use_ca:      enable SE-style channel attention (default True).
        use_eh:      enable embedding head; if False, fc1 → classifier
                     directly (default True).
    """

    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        embed_dim: int = 256,
        dropout: float = 0.5,
        use_ms: bool = True,
        use_ca: bool = True,
        use_eh: bool = True,
    ):
        super().__init__()

        # ── ResNet50 backbone (strip fc + avgpool) ──
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)

        # Keep everything except the final fully-connected and avgpool layers.
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )  # → 64 × 56 × 56
        self.layer1 = backbone.layer1    # → 256 × 56 × 56
        self.layer2 = backbone.layer2    # → 512 × 28 × 28
        self.layer3 = backbone.layer3    # → 1024 × 14 × 14
        self.layer4 = backbone.layer4    # → 2048 × 7 × 7
        self.out_channels = 2048

        # ── MSCA block ──
        self.use_ms = use_ms
        self.use_ca = use_ca
        self.use_eh = use_eh

        if use_ms or use_ca:
            self.msca = MSCA_Block(self.out_channels, use_ms=use_ms, use_ca=use_ca)
        else:
            self.msca = None

        # ── Pooling ──
        self.avgpool = nn.AdaptiveAvgPool2d(1)   # → 2048 × 1 × 1

        # ── Fully-connected projection ──
        self.fc1 = nn.Sequential(
            nn.Linear(self.out_channels, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # ── Embedding head (optional) ──
        if use_eh:
            self.embed = nn.Sequential(
                nn.Linear(1024, embed_dim),
                nn.BatchNorm1d(embed_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            self.classifier = nn.Linear(embed_dim, num_classes)
        else:
            self.embed = None
            self.classifier = nn.Linear(1024, num_classes)

        # ── Weight initialisation for newly added layers ──
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor, return_feat: bool = False):
        # ── ResNet50 backbone ──
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)               # → 2048 × 7 × 7

        # ── MSCA enhancement ──
        if self.msca is not None:
            x = self.msca(x)             # → 2048 × 7 × 7

        # ── Pooling & flatten ──
        x = self.avgpool(x)              # → 2048 × 1 × 1
        x = torch.flatten(x, 1)          # → 2048

        # ── FC projection ──
        x = self.fc1(x)                  # → 1024

        # ── Embedding + classify ──
        if self.use_eh:
            feat = self.embed(x)         # → embed_dim (256)
        else:
            feat = x                     # → 1024
        out = self.classifier(feat)

        if return_feat:
            return feat, out
        return out


# ── Quick sanity check ──
if __name__ == "__main__":
    model = MSCA_ResNet50(num_classes=10, pretrained=True,
                          use_ms=True, use_ca=True, use_eh=True)
    dummy = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        feats, logits = model(dummy, return_feat=True)
    print(f"Input  : {dummy.shape}")
    print(f"Feat   : {feats.shape}   (expected: [2, 256])")
    print(f"Logits : {logits.shape}  (expected: [2, 10])")

    # Parameter count
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params : {total / 1e6:.2f} M total  |  {trainable / 1e6:.2f} M trainable")

    # Ablation variants
    for ms, ca, eh in [(True, True, True), (False, True, True),
                        (True, False, True), (True, True, False),
                        (False, False, True)]:
        m = MSCA_ResNet50(num_classes=10, pretrained=False,
                          use_ms=ms, use_ca=ca, use_eh=eh)
        p = sum(p.numel() for p in m.parameters())
        print(f"  MS={ms} CA={ca} EH={eh}  →  {p / 1e6:.2f} M params")
