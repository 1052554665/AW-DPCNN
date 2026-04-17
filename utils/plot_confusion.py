import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion(
    y_true,
    y_pred,
    class_names,
    # save_path="confusion_matrix_baseline_cwru.png",
    # save_path="confusion_matrix_convnext_test=val.png",
    # save_path="confusion_matrix_convnext_tiny_cwru.png",
    save_path="confusion_matrix_test=val_vgg16.png",
    # save_path="confusion_matrix_resnet18_cwru.png",
    # save_path="confusion_matrix_resnet18_se_bs16_test=val.png",
    # save_path="confusion_matrix_alexnet_se_cwru.png",
    # save_path="confusion_matrix_ResNet50_CBAM_bs16_test=val.png",
    # save_path="confusion_matrix_vit_test=val.png",
    # save_path="confusion_matrix_CE_vit_test=val.png",
    # save_path="confusion_matrix_MSCA_VGG16_cwru.png",
    # save_path="confusion_matrix_VGG16_cwru_gadf.png",

):
    # 1. 计算归一化混淆矩阵
    cm = confusion_matrix(y_true, y_pred, normalize='true')

    # 2. IEEE 风格全局设置
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "stix",
        # "font.serif": ["Times New Roman"],
        "font.size": 15,    # 全局默认字号
        "axes.labelsize": 15,   # 坐标轴标签
        "axes.titlesize": 15,   # 图表标题
        "xtick.labelsize": 14,   # X轴刻度（略小于标签，避免拥挤）
        "ytick.labelsize": 14,   # Y轴刻度（与X轴一致）
        "legend.fontsize": 14,   # 图例字号
        "figure.dpi": 300,
        "axes.linewidth": 0.8,  # 坐标轴线条粗细
    })

    # 3. 绘图
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names,
        square=True,
        linewidths=0.5,
        ax=ax
    )

    # 4. 坐标轴与标题
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    # ax.set_title("Confusion Matrix (Test Set)")

    # 5. 横坐标旋转 60°
    plt.setp(
        ax.get_xticklabels(),
        rotation=60,
        ha="right",
        rotation_mode="anchor"
    )

    plt.tight_layout()

    # 6. 保存（IEEE 推荐：bbox_inches='tight'）
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
