# revision plan
>This paper has been rejected by many journals, there list some reviewers' comments. Some issues in this paper have been fixed, but some are still under discussion and improvement. Refering to the reviewers' comments and this project, the paper, make a detailed plan to improve the paper and corresponding experiments. The plan should include the following aspects:
1. The main contributions of the paper, and how to highlight them in the revised version.
2. The main weaknesses of the paper, and how to address them in the revised version.
3. The main experiments that need to be added or improved, and how to design and conduct them.
4. The main related work that needs to be cited or discussed, and how to integrate them into the paper.
5. The main writing issues that need to be fixed, and how to improve the clarity and readability of the paper.
6. The main future work that can be proposed, and how to outline the potential directions for further research. 
Make sure to provide specific and actionable steps for each aspect, and to prioritize the most important ones. The plan should be comprehensive and feasible, and should demonstrate a clear understanding of the reviewers' comments and the research topic.



## Top 5 Critical Actions (P0)

| # | What | Why (Reviewer) |
|---|---|---|
| 1 | **Full 9-row component decomposition ablation** (B0–B8) | Directly refutes "package-style innovation" — isolates every component's marginal gain |
| 2 | **Add 4 SOTA baselines** (EfficientNet, ViT, CE-ViT, MobileNetV3) | All reviewers demanded modern comparisons; configs mostly ready |
| 3 | **Model complexity table** (params, FLOPs, inference time) | IEEE Sensors R2; code already implemented, just needs compilation |
| 4 | **Representation justification experiment** (Mel vs STFT vs CWT; GADF vs GASF vs MTF) | IEEE Sensors R1 + R2 — empirically justify why these two representations |
| 5 | **Rewriting contributions** from "we combined X, Y, Z" → "we solved gap A via mechanism B, yielding gain C" | Measurement R2, R5 — novelty perception |

## Quick Wins Already Ready (from codebase)
- ✅ transformer-five dataset (5,859 images, 5-class acoustic) — ready for generalization validation
- ✅ Group2_4 dataset (7,497 images, 8-class acoustic) — ready for generalization validation
- ✅ File-level split with JS divergence verification — addresses leakage concerns
- ✅ ROC curves + AUC for all models
- ✅ Params/FLOPs computation integrated into training pipeline


# SOTA Baselines
>Review and add 4 SOTA baselines: EfficientNet, ViT, CE-ViT, MobileNetV3,  corresponding configuration files are also needed.

## 4 SOTA Baselines — Status

| Model | Params | Config | Model File | Registry | Training Ready |
|---|---|---|---|---|---|
| **EfficientNet-B0** | 4.01M | efficientnet_b0.yaml ✅ | `efficientnet.py` ✅ | ✅ | ✅ (lr fixed: 1e-6→1e-4) |
| **ViT** | 11.02M | vit.yaml ✅ | `vit.py` ✅ | ✅ | ✅ |
| **CE-ViT** | 20.78M | CE_ViT.yaml ✅ | `CE_ViT.py` ✅ | ✅ | ✅ (lr fixed: 1e-6→1e-4) |
| **MobileNetV3-Small** | 1.52M | `mobilenetv3_small.yaml` 🆕 | `mobilenetv3.py` 🆕 | 🆕 | ✅ |

### What was done

1. **Created** mobilenetv3.py — MobileNetV3-Small wrapper with ImageNet pretrained weights, configurable dropout, replaceable classifier head.

2. **Created** mobilenetv3_small.yaml — config with standard settings: 30 epochs, lr=1e-4, AdamW, ReduceLROnPlateau.

3. **Registered** `mobilenetv3_small` (also `mobilenetv3`, `mobilenet_v3`) in registry.py.

4. **Fixed** efficientnet_b0.yaml and CE_ViT.yaml — learning rate was `1e-6` (fine-tuning level), corrected to `1e-4` (initial training).

### Usage

```bash
# Run all 4 SOTA baselines
python scripts/run_exp1_all.py --config configs/default.yaml \
    --pattern "efficientnet_b0.yaml|vit.yaml|CE_ViT.yaml|mobilenetv3_small.yaml"

# Or individually
python scripts/train.py --config configs/default.yaml \
    --exp-config experiments/exp1/mobilenetv3_small.yaml
```
