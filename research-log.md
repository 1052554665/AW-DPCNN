Create a new model architecture by replacing the backbone with ResNet50 while keeping the multi-scale convolutional layers. Additionally, enable an embedding head for feature representation and incorporate SE-style channel attention to enhance the model's performance.


### New Model: `MSCA_ResNet50`

**File**: MSCA_ResNet50.py  
**Registry name**: `msca_resnet50` (or `msca_resnet`)

| Component | Specification |
|---|---|
| **Backbone** | ResNet50 (ImageNet-1K V2 pretrained), stripped of FC/avgpool |
| **Feature dims** | Stem → layer1 → layer2 → layer3 → layer4 → **2048 × 7 × 7** |
| **MSCA Block** | Multi-scale: 3×3, 5×5, dilated 3×3 (d=2) + SE channel attention (r=16) |
| **FC Projection** | 2048 → 1024, ReLU, Dropout(0.5) |
| **Embedding Head** | 1024 → 256, BN, ReLU, Dropout(0.5) |
| **Classifier** | 256 → `num_classes` |
| **Total params** | 206.76M (full), 64.15M (MS off) |

### Ablation flags (matching MSCA_VGG16 convention)

| Flag | Off effect | Params |
|---|---|---|
| `use_ms=False` | Single 3×3 conv replaces multi-scale | 64.15M |
| `use_ca=False` | Removes SE channel attention | 206.24M |
| `use_eh=False` | fc1 → classifier directly (1024 → K) | 206.51M |

### Usage in experiment YAML

```yaml
model:
  name: msca_resnet50
  num_classes: 9
  pretrained: true
  use_ms: true
  use_ca: true
  use_eh: true
  embed_dim: 256
  dropout: 0.5
```

### Key design differences vs. MSCA_VGG16

| Aspect | MSCA_VGG16 | MSCA_ResNet50 |
|---|---|---|
| Backbone channels | 512 | **2048** |
| Receptive field | Progressive 3×3 stack | Residual blocks + skip connections |
| Pretrained weights | VGG16-BN ImageNet-1K V1 | ResNet50 ImageNet-1K V2 |
| Parameter efficiency | Lightweight (~15M) | Heavier (~207M) due to 2048-d MSCA convs |

Made changes.