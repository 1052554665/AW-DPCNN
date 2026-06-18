# Research Log — AW-DPCNN

## Issues Identified
- On CWRU dataset, the accuracy all reaches 1.0, which is suspicious. Need to check if the data split is correct and if there is any data leakage.
- On Transformer dataset, the accuracy is below 0.5.

## Raw Input t-SNE Analysis

> Since the registered models expose learned features, it is necessary to create a new visualization that shows t-SNE of raw input features, to check if the raw input features are already linearly separable. If they are, then the high accuracy on CWRU dataset might be due to the fact that the model is just learning to separate the raw features, rather than learning any meaningful patterns.


### Created: raw_input_tsne.py

A standalone script that loads raw image pixels (Resize + ToTensor only — **no normalization**, no model) and runs PCA → t-SNE to visualize whether raw input features are already linearly separable.

- `scripts/raw_input_tsne.py` — loads raw pixels (no normalisation, only Resize + ToTensor), applies PCA→50 dims, then t-SNE.

### Key Findings

| Dataset | Raw Pixel Separability | Interpretation |
|---|---|---|
| **CWRU** | ✅ Well-separated clusters | Input already discriminative — explains the ~1.0 accuracy ceiling. The AW-DPCNN fused images contain inherently separable patterns. |
| **Transformer** | ❌ Heavily overlapped | Genuinely difficult — the ~0.5 accuracy reflects real challenge. Models must learn non-trivial features. |
| **Ablation variants** (Transformer) | ❌ All overlapped | Consistent across mel_only, gadf_only, concat, awdpcnn_gamma1 — none separate at raw pixel level. |

>save the t-SNE output in `experiment/tsne_raw_input/`.

>the t-SNE feature of the raw input can not be separated. After training, the t-SNE feature of the learned features can be separated, what is the result means? 

!!!! two notes


 which indicates that the high accuracy on CWRU dataset is not due to the raw input features being linearly separable. This suggests that the model is learning meaningful patterns from the data, rather than just memorizing the raw input features. 