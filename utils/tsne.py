import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

@torch.no_grad()
def extract_features(model, loader, device):
    model.eval()
    features = []
    labels = []

    for x, y in loader:
        x = x.to(device)
        feat, _ = model(x, return_feat=True)
        features.append(feat.cpu().numpy())
        labels.append(y.numpy())

    return np.vstack(features), np.hstack(labels)


def plot_tsne(
    features,
    labels,
    class_names,
    title="t-SNE Visualization",
    # save_path="tsne_baseline_cwru.png"
    # save_path = "tsne_convnext_tiny_cwru.png"
    save_path = "tsne_vgg16_test=val.png"
    # save_path = "tsne_resnet18_cwru.png"
    # save_path = "tsne_resnet18_se_bs16_test=val.png"
    # save_path = "tsne_ConvNeXt_test=val.png"
    # save_path = "tsne_alexnet_se_cwru.png"
    # save_path="tsne_ResNet50_CBAM_bs16_test=val.png"
    # save_path = "tsne_vit_test=val.png"
    # save_path = "tsne_CE_vit_test=val.png"
    # save_path = "tsne_VGG16_cwru_gadf.png"
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
        "font.size": 15,  # 全局默认字号
        "axes.labelsize": 15,  # 坐标轴标签
        "axes.titlesize": 15,  # 图表标题
        "xtick.labelsize": 14,  # X轴刻度（略小于标签，避免拥挤）
        "ytick.labelsize": 14,  # Y轴刻度（与X轴一致）
        "legend.fontsize": 14,  # 图例字号
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
            s=8,              # IEEE 推荐小点
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

    # 6. 保存
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
