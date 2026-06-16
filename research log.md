# Lightening MSCA-VGG16

>The parameters of the model `MSCA-VGG16` is too large to train on the current dataset, which may lead to overfitting. Therefore, lightening the model is necessary.

Here's what changed in `MSCA_VGG16`:

| Component | Before | After | Reduction |
|---|---|---|---|
| Pooling | `AdaptiveAvgPool2d(7,7)` → 25,088 | `AdaptiveAvgPool2d(1)` → **512** | 49× |
| `fc1` | 25,088 → 4096 = **102.8M** | 512 → 1024 = **0.52M** | **197×** |
| `embed` | 4096 → 1024 = 4.2M | 1024 → 256 = 0.26M | 16× |
| `embed_dim` default | 1024 | **256** | 4× |
| `num_classes` default | 10 | **4** | — |
| **Total params** | ~126M | **~26.8M** | **4.7×** |

The key fix: global average pooling (`AdaptiveAvgPool2d(1)`) replaces the old `7×7` pool, shrinking the flattened feature from 25,088 down to 512 before the fully connected layers. This eliminates the 103M-parameter `fc1` bottleneck that was the primary cause of overfitting on small/medium fault diagnosis datasets.
