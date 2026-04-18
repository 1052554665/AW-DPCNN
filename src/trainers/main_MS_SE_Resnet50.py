
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.MS_SE_Resnet50 import ResNet50_MSSE

# from cnn_audio_classification.train_eval import train_one_epoch, evaluate, save_logs
from utils.plot_confusion import plot_confusion

# ========= 配置 =========
# data_root = "datasets"
# data_root = "data5_mel"
# data_root = "/home/220242215063/pycharm_project_legion/data5_awpcnn1"
# data_root = "/home/220242215063/data5_awpcnn2"
data_root = "/home/220242215063/pycharm_project_legion/AW-DPCNN/data4"

num_workers = 32
num_classes = 10
batch_size = 32
epochs = 30
lr = 1e-4
# for ConvNeXt
# lr = 3e-4


# ========= 设备检测（强制） =========
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("⚠️ CUDA not available, using CPU")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class_names = sorted(
    datasets.ImageFolder(f"{data_root}/train").classes
)

# ========= 数据 =========
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485,0.456,0.406],
    std=[0.229,0.224,0.225])
    # mean = [0.5, 0.5, 0.5],
    # std = [0.5, 0.5, 0.5])
])

train_set = datasets.ImageFolder(f"{data_root}/train", transform)
val_set   = datasets.ImageFolder(f"{data_root}/val", transform)
test_set  = datasets.ImageFolder(f"{data_root}/test", transform)

train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=num_workers,
        # persistent_workers=True,    # worker 不反复重启,epoch 多时强烈建议
        # pin_memory=True,    # CPU→GPU 拷贝更快,GPU 训练必开
        # prefetch_factor=4,  # 提前预取：每个 worker 预取 batch
)

val_loader = DataLoader(val_set, batch_size, shuffle=False,
                        num_workers=num_workers,
                        # persistent_workers=True
                )

test_loader  = val_loader
# test_loader = DataLoader(test_set, batch_size, shuffle=False,
#                          num_workers=num_workers,
#                          )
# 确保标签映射无误
print("Train classes:", train_set.class_to_idx)
print("Val   classes:", val_set.class_to_idx)
print("Test  classes:", test_set.class_to_idx)


# ========= 模型 =========
# model = ResNet18(num_classes, pretrained=True)
model = ResNet50_MSSE(num_classes)
# model = VGG16(num_classes)
# model = ResNet18_SE(num_classes)
# model = ConvNeXt(num_classes)
# model = ConvNeXtTiny(num_classes,pretrained=True)
# model = SE_AlexNet(num_classes)
model.to(device)

# optimizer = torch.optim.AdamW(model.parameters(), lr=lr).
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=lr,
    weight_decay=1e-2
)
criterion = nn.CrossEntropyLoss()

logs = []

# ========= train =========
for epoch in range(epochs):

    train_loss, train_acc = train_one_epoch(
        model, train_loader, optimizer, criterion, device
    )

    val_loss, metrics, _, _ = evaluate(
        model, val_loader, criterion, device
    )

    # 验证集指标
    acc, prec, rec, f1, gmean, bal_acc, kappa = metrics
    logs.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "gmean": gmean,
        "val_bal_acc": bal_acc,
        "val_kappa": kappa

    })

    # 打印训练轮次、训练准确率、验证准确率、F1分数
    print(
        f"[{epoch}] "
        f"TrainAcc={train_acc:.4f} "
        f"ValAcc={acc:.4f} "
        f"ValF1={f1:.4f}"
        f"Val_kappa={kappa:.4f}"
    )

save_logs(logs)

# ========= test =========
test_loss, test_metrics, y_true, y_pred = evaluate(
    model, test_loader, criterion, device
)

test_acc, test_prec, test_rec, test_f1, test_gmean, bal_acc, kappa = test_metrics

print("\n===== Test Results =====")
print(f"Test Acc     : {test_acc:.4f}")
print(f"Test Precision: {test_prec:.4f}")
print(f"Test Recall   : {test_rec:.4f}")
print(f"Test F1       : {test_f1:.4f}")
print(f"Test G-Mean   : {test_gmean:.4f}")

plot_confusion(y_true, y_pred, class_names)


from utils.tsne import extract_features, plot_tsne

# ======== t-SNE 可视化（Test Set）========
features, labels = extract_features(
    model, test_loader, device
)

plot_tsne(
    features,
    labels,
    class_names,
    title="t-SNE Visualization on Test Set"
)

# 单独存一个 test 结果文件
import pandas as pd

test_log = {
    "test_acc": test_acc,
    "test_precision": test_prec,
    "test_recall": test_rec,
    "test_f1": test_f1,
    "test_gmean": test_gmean,
    "test_bal_acc": bal_acc,
    "test_kappa": kappa
}

df_test = pd.DataFrame([test_log])
# df_test.to_csv("logs/test_metrics_ConvNeXtTiny_test=val.csv", index=False)
df_test.to_csv("logs/test_metrics_ResNet50_MSSE_test=val.csv", index=False)

