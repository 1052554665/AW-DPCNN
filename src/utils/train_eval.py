from contextlib import nullcontext
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from src.utils.metrics import compute_metrics, compute_roc_auc


def _get_autocast_context(device: torch.device):
    """Return an autocast context manager for the given device.

    Prefers bfloat16 (native on Blackwell / H100 / A100) because it
    requires no gradient scaling and has the same dynamic range as FP32.
    Falls back to float16 autocast + GradScaler if bfloat16 is not
    supported by the GPU, and to a no-op on CPU.
    """
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.amp.autocast("cuda", dtype=torch.bfloat16)
        else:
            return torch.amp.autocast("cuda", dtype=torch.float16)
    return nullcontext()


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
    autocast_ctx = _get_autocast_context(device)

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        with autocast_ctx:
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
def evaluate(model, loader, criterion, device):
    """Evaluate model on a dataloader.

    Returns:
        loss: average loss.
        metrics: (acc, precision, recall, f1, gmean, bal_acc, kappa).
        y_true: ground-truth labels.
        y_pred: predicted labels.
        y_score: predicted class probabilities (softmax), shape [N, num_classes].
    """
    model.eval()
    losses = []
    y_true, y_pred, y_score = [], [], []
    autocast_ctx = _get_autocast_context(device)

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with autocast_ctx:
            out = _extract_logits(model(x))
            loss = criterion(out, y)
        losses.append(loss.item())

        probs = torch.softmax(out, dim=1)
        preds = out.argmax(dim=1)

        y_true.extend(y.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_score.extend(probs.float().cpu().numpy())

    metrics = compute_metrics(y_true, y_pred)
    return sum(losses) / len(losses), metrics, y_true, y_pred, y_score


def calculate_roc_auc(y_true, y_score, num_classes: int):
    """Compute ROC-AUC from accumulated predictions."""
    if y_score is None or len(y_score) == 0:
        return 0.0
    y_score = np.asarray(y_score, dtype=float)
    return compute_roc_auc(y_true, y_score, num_classes, average="macro")


def check_label_leakage(y_true, y_pred, num_classes: int, num_shuffles: int = 5,
                        threshold: float = 0.10) -> dict:
    """Label-shuffling sanity check for data leakage detection.

    Shuffles the ground-truth labels and re-computes accuracy.  If the
    model performs substantially better than random chance on shuffled
    labels, the data pipeline almost certainly contains leakage.

    Parameters
    ----------
    y_true : array-like   Ground-truth labels.
    y_pred : array-like   Predicted labels.
    num_classes : int     Number of classes.
    num_shuffles : int    Number of shuffle trials (more → stabler estimate).
    threshold : float     Accuracy margin above random chance that triggers
                          a warning (default 0.10 = 10 percentage points).

    Returns
    -------
    dict with keys:
        shuffled_acc       – mean accuracy over shuffle trials.
        chance_level       – expected accuracy under random guessing (1/K).
        is_suspicious      – True if shuffled_acc exceeds chance_level
                             by more than *threshold*.
        warning            – human-readable message (empty string if OK).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    chance_level = 1.0 / max(num_classes, 1)

    shuffled_accs = []
    rng = np.random.RandomState(42)
    for _ in range(num_shuffles):
        y_shuffled = rng.permutation(y_true)
        shuffled_accs.append(accuracy_score(y_shuffled, y_pred))

    shuffled_acc = float(np.mean(shuffled_accs))
    margin = shuffled_acc - chance_level
    is_suspicious = margin > threshold

    if is_suspicious:
        warning = (
            f"[DATA LEAKAGE WARNING] Shuffled-label accuracy = {shuffled_acc:.4f} "
            f"(chance level for {num_classes} classes = {chance_level:.4f}). "
            f"Margin of {margin:.4f} exceeds threshold {threshold:.2f}. "
            f"Audit your dataset splitting pipeline for leakage."
        )
    else:
        warning = ""

    return {
        "shuffled_acc": shuffled_acc,
        "chance_level": chance_level,
        "is_suspicious": is_suspicious,
        "warning": warning,
    }


def save_logs(logs, path="logs/training_log.csv"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(logs)
    df.to_csv(path, index=False)
