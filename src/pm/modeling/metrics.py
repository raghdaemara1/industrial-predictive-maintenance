from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray) -> Dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "per_class": classification_report(y_true, y_pred, labels=labels, output_dict=True),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def early_warning_rate(y_true_labels: pd.Series, y_pred_labels: pd.Series) -> float:
    mask = y_true_labels != "None"
    if mask.sum() == 0:
        return 0.0
    return float((y_true_labels[mask] == y_pred_labels[mask]).mean())
