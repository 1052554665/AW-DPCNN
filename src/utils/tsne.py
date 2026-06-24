from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def _extract_feature_tensor(model_output):
    if isinstance(model_output, tuple):
        candidates = [t for t in model_output if isinstance(t, torch.Tensor) and t.ndim == 2]
        if len(candidates) < 2:
            raise ValueError("Expected (logits, feat) or (feat, logits) when return_feat=True.")
        # In image classification, feature dimension is usually larger than num_classes.
        return max(candidates, key=lambda t: t.shape[1])
    if isinstance(model_output, torch.Tensor):
        return model_output
    raise ValueError("Unsupported model output type for feature extraction.")

@torch.no_grad()
def extract_features(model, loader, device):
    model.eval()
    features = []
    labels = []

    for x, y in loader:
        x = x.to(device)
        feat = _extract_feature_tensor(model(x, return_feat=True))
        features.append(feat.cpu().numpy())
        labels.append(y.numpy())

    return np.vstack(features), np.hstack(labels)


def plot_tsne(
    features,
    labels,
    class_names,
    title="t-SNE Visualization",
    save_path="results/tsne.png",
):
    # 1. t-SNE（参数固定，便于复现）
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        max_iter=1000,
        init="pca",
        random_state=42,
    )
    emb = tsne.fit_transform(features)

    # 2. IEEE 风格设置
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        # "font.serif": ["Times New Roman"],
        "font.size": 13,  # 全局默认字号
        "axes.labelsize": 13,  # 坐标轴标签
        "axes.titlesize": 15,  # 图表标题
        "xtick.labelsize": 14,  # X轴刻度（略小于标签，避免拥挤）
        "ytick.labelsize": 14,  # Y轴刻度（与X轴一致）
        # "legend.fontsize": 14,  # 图例字号
        "legend.fontsize": 10,  # 图例字号
        "figure.dpi": 300,
        "axes.linewidth": 0.8,  # 坐标轴线条粗细
    })

    # 3. 绘图（单栏尺寸友好）
    fig, ax = plt.subplots(figsize=(7, 6))

    for i, name in enumerate(class_names):
        idx = labels == i
        ax.scatter(
            emb[idx, 0],
            emb[idx, 1],
            s=30,              # IEEE 推荐小点
            alpha=0.8,
            label=name,
            edgecolors="none"
        )

    # 4. 轴与标题
    # ax.set_title(title)
    # ax.set_xlabel("t-SNE Dimension 1")
    # ax.set_ylabel("t-SNE Dimension 2")

    # 5. 图例（不遮挡主体）
    ax.legend(
        loc="best",
        frameon=False,
        markerscale=1.5
    )

    ax.grid(False)
    plt.tight_layout()

    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # 6. 保存
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
