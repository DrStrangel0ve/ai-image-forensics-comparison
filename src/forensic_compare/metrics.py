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


def threshold_candidates(scores: np.ndarray | list[float]) -> np.ndarray:
    finite_scores = np.unique(np.clip(np.asarray(scores, dtype=float), 0.0, 1.0))
    finite_scores = finite_scores[np.isfinite(finite_scores)]
    if len(finite_scores) == 0:
        return np.asarray([0.5], dtype=float)
    midpoint_candidates = (finite_scores[:-1] + finite_scores[1:]) / 2.0
    return np.unique(np.concatenate(([0.0, 0.5, 1.0], finite_scores, midpoint_candidates)))


def fast_threshold_metrics(
    y_true: np.ndarray | list[int],
    scores: np.ndarray | list[float],
    threshold: float,
) -> dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=int)
    score_arr = np.asarray(scores, dtype=float)
    y_pred = score_arr >= threshold
    y_true_bool = y_true_arr.astype(bool)
    tp = float(np.logical_and(y_pred, y_true_bool).sum())
    fp = float(np.logical_and(y_pred, ~y_true_bool).sum())
    tn = float(np.logical_and(~y_pred, ~y_true_bool).sum())
    fn = float(np.logical_and(~y_pred, y_true_bool).sum())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    return {
        "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1.0),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": float(threshold),
        "n_samples": int(len(y_true_arr)),
    }


def best_threshold(
    y_true: np.ndarray | list[int],
    scores: np.ndarray | list[float],
    objective: str = "accuracy",
) -> tuple[float, dict[str, float]]:
    if objective not in {"accuracy", "f1"}:
        raise ValueError(f"Unsupported threshold objective: {objective}")
    y_true_arr = np.asarray(y_true, dtype=int)
    score_arr = np.asarray(scores, dtype=float)
    best_threshold_value = 0.5
    best_metrics = fast_threshold_metrics(y_true_arr, score_arr, threshold=best_threshold_value)
    best_value = float(best_metrics[objective])
    for threshold in threshold_candidates(score_arr):
        metrics = fast_threshold_metrics(y_true_arr, score_arr, threshold=float(threshold))
        value = float(metrics[objective])
        if value > best_value or (
            value == best_value
            and abs(float(threshold) - 0.5) < abs(best_threshold_value - 0.5)
        ):
            best_threshold_value = float(threshold)
            best_metrics = metrics
            best_value = value
    return best_threshold_value, best_metrics
