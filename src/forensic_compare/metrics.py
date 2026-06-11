from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(
    y_true: np.ndarray | list[int],
    y_score: np.ndarray | list[float],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute binary metrics with y=1 as generated/fake."""

    y_true_arr = np.asarray(y_true, dtype=int)
    y_score_arr = np.asarray(y_score, dtype=float)
    y_pred = (y_score_arr >= threshold).astype(int)
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(y_true_arr, y_pred).tolist(),
        "n_samples": int(len(y_true_arr)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true_arr, y_score_arr))
    except ValueError:
        metrics["roc_auc"] = None
    return metrics
