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


def _binary_arrays(
    y_true: np.ndarray | list[int],
    y_score: np.ndarray | list[float],
) -> tuple[np.ndarray, np.ndarray]:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_score_arr = np.asarray(y_score, dtype=float)
    if y_true_arr.shape[0] != y_score_arr.shape[0]:
        message = (
            "y_true and y_score must have the same length, "
            f"got {len(y_true_arr)} and {len(y_score_arr)}"
        )
        raise ValueError(message)
    if len(y_true_arr) == 0:
        raise ValueError("Calibration metrics require at least one sample")
    if not np.isin(y_true_arr, [0, 1]).all():
        raise ValueError("Calibration metrics require binary labels encoded as 0/1")
    if not np.isfinite(y_score_arr).all():
        raise ValueError("Calibration metrics require finite scores")
    return y_true_arr, np.clip(y_score_arr, 0.0, 1.0)


def brier_score(
    y_true: np.ndarray | list[int],
    y_score: np.ndarray | list[float],
) -> float:
    """Return the Brier score for fake-class probabilities."""

    y_true_arr, y_score_arr = _binary_arrays(y_true, y_score)
    return float(np.mean((y_score_arr - y_true_arr) ** 2))


def calibration_bins(
    y_true: np.ndarray | list[int],
    y_score: np.ndarray | list[float],
    n_bins: int = 10,
) -> list[dict[str, float | int]]:
    """Summarize reliability bins for fake-class probabilities."""

    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    y_true_arr, y_score_arr = _binary_arrays(y_true, y_score)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_score_arr, edges[1:-1], right=False)
    rows: list[dict[str, float | int]] = []
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        count = int(mask.sum())
        if count == 0:
            confidence = 0.0
            accuracy = 0.0
            gap = 0.0
        else:
            confidence = float(y_score_arr[mask].mean())
            accuracy = float(y_true_arr[mask].mean())
            gap = abs(confidence - accuracy)
        rows.append(
            {
                "bin": bin_id,
                "bin_lower": float(edges[bin_id]),
                "bin_upper": float(edges[bin_id + 1]),
                "count": count,
                "weight": float(count / len(y_true_arr)),
                "confidence": confidence,
                "accuracy": accuracy,
                "abs_gap": gap,
            }
        )
    return rows


def expected_calibration_error(
    y_true: np.ndarray | list[int],
    y_score: np.ndarray | list[float],
    n_bins: int = 10,
) -> float:
    """Return equal-width expected calibration error."""

    rows = calibration_bins(y_true, y_score, n_bins=n_bins)
    return float(sum(row["weight"] * row["abs_gap"] for row in rows))


def maximum_calibration_error(
    y_true: np.ndarray | list[int],
    y_score: np.ndarray | list[float],
    n_bins: int = 10,
) -> float:
    """Return the largest non-empty-bin calibration gap."""

    nonempty_gaps = [
        float(row["abs_gap"])
        for row in calibration_bins(y_true, y_score, n_bins=n_bins)
        if int(row["count"]) > 0
    ]
    return max(nonempty_gaps, default=0.0)


def binary_metrics(
    y_true: np.ndarray | list[int],
    y_score: np.ndarray | list[float],
    threshold: float = 0.5,
    n_bins: int = 10,
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
        "brier_score": brier_score(y_true_arr, y_score_arr),
        "expected_calibration_error": expected_calibration_error(y_true_arr, y_score_arr, n_bins=n_bins),
        "maximum_calibration_error": maximum_calibration_error(y_true_arr, y_score_arr, n_bins=n_bins),
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
