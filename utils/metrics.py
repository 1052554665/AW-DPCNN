# Accuracy / Precision / Recall / F1 / G-mean
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    balanced_accuracy_score,
    cohen_kappa_score
)
from scipy.stats import gmean

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    recall_per_class = recall_score(y_true, y_pred, average=None)
    g_mean = gmean(recall_per_class + 1e-6)

    bal_acc = balanced_accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    return acc, precision, recall, f1, g_mean, bal_acc, kappa

