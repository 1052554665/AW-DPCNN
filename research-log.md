reduce the model scale of vgg16, the vgg16 is same as MSCA_VGG16 but remove the multi-scale convs, channel attention and compact embedding head. The vgg16 is a baseline model for comparison with MSCA_VGG16.


# Reduced VGG16 baseline
vgg = VGG16(num_classes=10, pretrained=False)
x = torch.randn(2, 3, 224, 224)
feat, out = vgg(x, return_feat=True)
print('=== Reduced VGG16 (Baseline) ===')
print(f'Input : {x.shape}')
print(f'Feat  : {feat.shape}  (expected [2, 1024])')
print(f'Logits: {out.shape}  (expected [2, 10])')
p_vgg = sum(p.numel() for p in vgg.parameters()) / 1e6
print(f'Params: {p_vgg:.2f}M')

print()

# MSCA-VGG16 full
msca = MSCA_VGG16(num_classes=10, pretrained=False, use_ms=True, use_ca=True, use_eh=True)
feat2, out2 = msca(x, return_feat=True)
print('=== MSCA-VGG16 (Full) ===')
print(f'Feat  : {feat2.shape}  (expected [2, 256])')
print(f'Logits: {out2.shape}  (expected [2, 10])')
p_msca = sum(p.numel() for p in msca.parameters()) / 1e6
print(f'Params: {p_msca:.2f}M')

print()

# MSCA-VGG16 with all enhancements OFF (= should be same structure as reduced VGG16)
msca_off = MSCA_VGG16(num_classes=10, pretrained=False, use_ms=False, use_ca=False, use_eh=False)
feat3, out3 = msca_off(x, return_feat=True)
print('=== MSCA-VGG16 (MS=False, CA=False, EH=False) ===')
print(f'Feat  : {feat3.shape}  (expected [2, 1024])')
print(f'Logits: {out3.shape}  (expected [2, 10])')
p_off = sum(p.numel() for p in msca_off.parameters()) / 1e6
print(f'Params: {p_off:.2f}M')
print(f'Match : p_vgg == p_off? {abs(p_vgg - p_off) < 0.1}')
"

The reduced VGG16 baseline is verified. Here's a summary:

---

### VGG16 Baseline — Reduced

**File**: vgg16.py

| Property | Old VGG16 | New VGG16 | MSCA-VGG16 (full) |
|---|---|---|---|
| Backbone | VGG16-BN | VGG16-BN | VGG16-BN |
| Feature dim | 7×7×512 | 7×7×512 | 7×7×512 |
| Pooling | AdaptiveAvgPool2d | AdaptiveAvgPool2d | AdaptiveAvgPool2d |
| FC projection | 25088→4096→4096 | **512→1024** | 512→1024 |
| Multi-scale convs | — | — | 3×3, 5×5, dilated |
| Channel attention | — | — | SE (r=16) |
| Embedding head | — | — | 1024→256, BN |
| Classifier | 4096→K | 1024→K | 256→K |
| **Params** | ~134M | **15.26M** | 26.82M |
| t-SNE feature | 4096-d | **1024-d** | 256-d |

**Equivalence**: `VGG16(num_classes=K)` ≡ `MSCA_VGG16(num_classes=K, use_ms=False, use_ca=False, use_eh=False)` — verified identical parameter count (15.26M).

Made changes.