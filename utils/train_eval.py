# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import pandas as pd
from utils.metrics import compute_metrics
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        preds = out.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    train_acc = correct / total
    return running_loss / len(loader), train_acc



@torch.no_grad()
def evaluate(model, loader, criterion, device): # loader 是谁，指标就属于谁
    model.eval()
    losses = []
    y_true, y_pred = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        losses.append(loss.item())

        preds = out.argmax(dim=1)
        y_true.extend(y.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

    metrics = compute_metrics(y_true, y_pred)
    return sum(losses)/len(losses), metrics, y_true, y_pred


# def save_logs(logs, path="logs/training_log_ResNet50_CBAM_bs16_test=val.csv"):
# def save_logs(logs, path="logs/training_log_vit_test=val.csv"):
# def save_logs(logs, path="logs/training_log_CE_vit_test=val.csv"):

# def save_logs(logs, path="logs/training_log_alexnet_se_bs16_cwru.csv"):
# def save_logs(logs, path="logs/training_log_baseline_cwru.csv"):
# def save_logs(logs, path="logs/training_log_convnext_test=val.csv"):
# def save_logs(logs, path="logs/training_log_convnext_tiny_cwru.csv"):
def save_logs(logs, path="logs/training_log_VGG16_cwru_gadf.csv"):
# def save_logs(logs, path="logs/training_log_resnet18_cwru.csv"):
# def save_logs(logs, path="logs/training_log_resnet18_se_bs=16_test=val.csv"):
# def save_logs(logs, path="logs/training_log_ConvNeXt.csv"):

    df = pd.DataFrame(logs)
    df.to_csv(path, index=False)
