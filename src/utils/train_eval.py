from pathlib import Path

import torch
import pandas as pd
from src.utils.metrics import compute_metrics


def _extract_logits(output):
    if isinstance(output, tuple):
        candidates = [tensor for tensor in output if isinstance(tensor, torch.Tensor) and tensor.ndim == 2]
        if not candidates:
            raise ValueError("Model returned tuple output but no logits tensor was found.")
        # Logits usually have the smallest channel dimension (num_classes).
        return min(candidates, key=lambda t: t.shape[1])
    return output


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = _extract_logits(model(x))
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
        out = _extract_logits(model(x))
        loss = criterion(out, y)
        losses.append(loss.item())

        preds = out.argmax(dim=1)
        y_true.extend(y.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

    metrics = compute_metrics(y_true, y_pred)
    return sum(losses)/len(losses), metrics, y_true, y_pred


def save_logs(logs, path="logs/training_log.csv"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(logs)
    df.to_csv(path, index=False)
