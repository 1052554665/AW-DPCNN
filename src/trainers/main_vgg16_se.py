
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.vgg16_se import VGG16_SE

from cnn_audio_classification.train_eval import train_one_epoch, evaluate, save_logs
from utils.plot_confusion import plot_confusion

# ========= 配置 =========
# data_root = "datasets"
# data_root = "data5_mel"
# data_root = "/home/220242215063/pycharm_project_legion/data5_awpcnn1"
# data_root = "/home/220242215063/data5_awpcnn2"
data_root = "/home/220242215063/pycharm_project_legion/AW-DPCNN/data4"

num_workers = 32
batch_size = 16
epochs = 30
lr = 1e-4

# ========= 设备检测（强制） =========
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("⚠️ CUDA not available, using CPU")


# ========= 数据 =========
transform = transforms.Compose([
    # transforms.Resize(224),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485,0.456,0.406],
    std=[0.229,0.224,0.225])
    # mean = [0.5, 0.5, 0.5],
    # std = [0.5, 0.5, 0.5])
])

train_set = datasets.ImageFolder(f"{data_root}/train", transform)
class_names = train_set.classes
num_classes = len(class_names)
print("Classes:", class_names)
print("Num classes:", num_classes)

val_set   = datasets.ImageFolder(f"{data_root}/val", transform)
class_names = val_set.classes
num_classes = len(class_names)
print("Classes:", class_names)
print("Num classes:", num_classes)

test_set  = datasets.ImageFolder(f"{data_root}/test", transform)
class_names = test_set.classes
num_classes = len(class_names)
print("Classes:", class_names)
print("Num classes:", num_classes)

# 确保标签映射无误
assert train_set.class_to_idx == val_set.class_to_idx
assert train_set.class_to_idx == test_set.class_to_idx

print("Train classes:", train_set.class_to_idx)
print("Val   classes:", val_set.class_to_idx)
print("Test  classes:", test_set.class_to_idx)

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


# ========= 模型 =========
model = VGG16_SE(num_classes)
model.to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=lr,
    weight_decay=1e-3
)

#=======Weighted CrossEntropyLoss===========
# 统计训练集类别分布
from collections import Counter

targets = [y for _, y in train_set.samples]
counter = Counter(targets)

print(counter)

# 计算类别权重
num_samples = len(targets)
class_weights = []

for i in range(num_classes):
    class_weights.append(num_samples / (num_classes * counter[i]))

class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

print("Class weights:", class_weights)

criterion = nn.CrossEntropyLoss(weight=class_weights)

# criterion = nn.CrossEntropyLoss()

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
        f"ValF1={f1:.4f} "
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
df_test.to_csv("logs/test=val_metrics_VGG16_se_test=bal.csv", index=False)
