Refer to `raw_input_tsne.py` to implement a section in this paper to analyse the raw input t-SNE, and compare it with the t-SNE of the fused representation. This will help to visually demonstrate the effectiveness of the fusion in terms of class separability.


## Added: Raw Input t-SNE Analysis (Section 5.5)

### What was added to tim.tex

A new subsection **"Raw Input Feature Visualization"** was inserted between the Ablation Study and the Representation Comparison sections. It contains:

| Element | Content |
|---------|---------|
| **Motivation** | Diagnostic check: if raw pixels are already separable, high accuracy is trivial — the section proves this is NOT the case |
| **Method** | PCA (150,528D → 50D) → t-SNE on raw pixels without normalization |
| **Fig. a** (`fig:raw_tsne_transformer`) | Raw input t-SNE for transformer dataset — severely overlapping clusters |
| **Fig. b** (`fig:raw_tsne_cwru`) | Raw input t-SNE for CWRU dataset — substantial inter-class overlap |
| **Analysis** | Contrasts raw input chaos with fused representation clarity from existing Figs. 7 & 9 |

### Key visual narrative

```
Raw pixels (Fig. a/b)  →  Fused AW-DPCNN (Fig. 7d/9d)
    scattered,                 compact, well-isolated
    overlapping                clusters
         ↓                           ↓
    NOT separable            Highly discriminative
```

This directly supports the paper's claim that AW-DPCNN fusion is the **cause** of improved class separability — not an artifact of the input data.

### Files created

| File | Description |
|------|-------------|
| tsne_Group2_4_harmonic_raw_input.png | 9-class transformer raw input t-SNE |
| tsne_cwru_within_raw_input.png | 4-class CWRU raw input t-SNE |

The `\graphicspath` was updated to include `{raw_tsne/}`.
