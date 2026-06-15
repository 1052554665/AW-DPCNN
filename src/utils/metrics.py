# Accuracy / Precision / Recall / F1 / G-mean / ROC-AUC / Model Complexity
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    roc_auc_score,
)
from scipy.stats import gmean
import torch
import torch.nn as nn


def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    g_mean = gmean(recall_per_class + 1e-6)

    bal_acc = balanced_accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    return acc, precision, recall, f1, g_mean, bal_acc, kappa


def compute_roc_auc(y_true, y_score, num_classes: int, average: str = "macro"):
    """Compute ROC-AUC for multi-class classification.

    Args:
        y_true: 1-d array of ground-truth class indices.
        y_score: 2-d array of predicted probabilities (shape [N, num_classes]).
        num_classes: total number of classes.
        average: 'macro' (default), 'weighted', or None for per-class.

    Returns:
        If average is None: list of per-class AUC values.
        Otherwise: scalar AUC.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    if y_score.ndim != 2 or y_score.shape[1] != num_classes:
        raise ValueError(
            f"y_score shape must be [N, {num_classes}], got {y_score.shape}"
        )

    # One-hot encode for multi-class ROC-AUC
    n_samples = y_true.shape[0]
    y_true_onehot = np.zeros((n_samples, num_classes), dtype=int)
    y_true_onehot[np.arange(n_samples), y_true] = 1

    return roc_auc_score(
        y_true_onehot,
        y_score,
        average=average,
        multi_class="ovr",
    )


# ---------------------------------------------------------------------------
#  Model complexity utilities
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count the number of (trainable) parameters in a model.

    Args:
        model: PyTorch model.
        trainable_only: If True, only count parameters with requires_grad=True.

    Returns:
        Total number of parameters.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def _count_conv2d(m, x, y):
    """Hook: count MACs for Conv2d."""
    # x[0].shape = (N, C_in, H, W)
    # m.weight.shape = (C_out, C_in, kH, kW)
    cin = m.in_channels
    cout = m.out_channels
    kernel_ops = m.kernel_size[0] * m.kernel_size[1]
    # Output spatial size
    oh, ow = y.shape[2], y.shape[3]
    # MACs per output element = cin * kernel_ops (ignoring bias for FLOPs convention)
    macs = cout * oh * ow * cin * kernel_ops
    if m.bias is not None:
        macs += cout * oh * ow  # bias add
    m._total_macs = m._total_macs + macs * x[0].size(0)


def _count_linear(m, x, y):
    """Hook: count MACs for Linear."""
    # x[0].shape = (N, *, in_features)
    # y.shape = (N, *, out_features)
    macs_per_sample = m.in_features * m.out_features
    if m.bias is not None:
        macs_per_sample += m.out_features
    m._total_macs = m._total_macs + macs_per_sample * x[0].size(0)


def _count_bn(m, x, y):
    """Hook: count MACs for BatchNorm2d."""
    # BN: one multiply-add per element (scale + shift) ≈ 2 ops per element
    nelements = y.numel()
    m._total_macs = m._total_macs + nelements


def _count_relu(m, x, y):
    """Hook: count MACs for ReLU/activation."""
    m._total_macs = m._total_macs + y.numel()


def _count_pool(m, x, y):
    """Hook: count MACs for pooling layers."""
    # MaxPool2d / AvgPool2d / AdaptiveAvgPool2d
    nelements = y.numel()
    if isinstance(m, (nn.MaxPool2d, nn.AvgPool2d)):
        kernel_ops = m.kernel_size * m.kernel_size if isinstance(m.kernel_size, int) else m.kernel_size[0] * m.kernel_size[1]
        nelements *= kernel_ops
    elif isinstance(m, nn.AdaptiveAvgPool2d):
        # Rough estimate: one comparison/add per input element
        pass
    m._total_macs = m._total_macs + nelements


# Registry of layers we know how to count
_HOOK_REGISTRY = {
    nn.Conv2d: _count_conv2d,
    nn.Linear: _count_linear,
    nn.BatchNorm2d: _count_bn,
    nn.ReLU: _count_relu,
    nn.MaxPool2d: _count_pool,
    nn.AvgPool2d: _count_pool,
    nn.AdaptiveAvgPool2d: _count_pool,
}


def compute_flops(model: nn.Module, input_shape=(1, 3, 224, 224), device=None) -> int:
    """Estimate the number of multiply–accumulate operations (MACs) for a forward pass.

    Uses forward hooks on registered layer types.  Unsupported layers are
    silently skipped (their contribution is not counted).

    Args:
        model: PyTorch model.
        input_shape: (N, C, H, W) shape of a dummy input.
        device: torch.device (if None, uses model's parameter device).

    Returns:
        Estimated total MACs.
    """
    if device is None:
        param = next(model.parameters())
        device = param.device

    model.eval()
    hooks = []
    total_macs = [0]  # mutable container

    def _make_hook(layer_type):
        def hook(m, x, y):
            if not hasattr(m, '_total_macs'):
                m._total_macs = 0
            _HOOK_REGISTRY[layer_type](m, x, y)
        return hook

    for m in model.modules():
        for layer_type, hook_fn in _HOOK_REGISTRY.items():
            if isinstance(m, layer_type):
                hooks.append(m.register_forward_hook(hook_fn))
                m._total_macs = 0
                break  # register only one hook per module

    # Forward pass with a dummy input
    dummy = torch.randn(*input_shape).to(device)
    with torch.no_grad():
        try:
            _ = model(dummy)
        except Exception:
            # If forward fails (e.g. model expects a tuple), just skip FLOPs
            pass

    # Collect totals
    total = 0
    for m in model.modules():
        if hasattr(m, '_total_macs'):
            total += m._total_macs

    # Cleanup
    for h in hooks:
        h.remove()
    for m in model.modules():
        if hasattr(m, '_total_macs'):
            del m._total_macs

    return total

