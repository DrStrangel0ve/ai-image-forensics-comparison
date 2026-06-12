from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


CALIBRATORS = {
    "temperature",
    "temperature_balanced",
    "platt",
    "platt_balanced",
    "isotonic",
    "isotonic_balanced",
}
EPS = 1e-6


@dataclass
class FittedCalibrator:
    name: str
    model: Any
    temperature: float | None = None


def logit(scores: np.ndarray) -> np.ndarray:
    clipped = np.clip(scores.astype(float), EPS, 1.0 - EPS)
    return np.log(clipped / (1.0 - clipped))


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))


def class_balanced_weights(y_true: np.ndarray) -> np.ndarray:
    weights = np.ones(len(y_true), dtype=float)
    for label in np.unique(y_true):
        mask = y_true == label
        weights[mask] = len(y_true) / max(2.0 * float(mask.sum()), 1.0)
    return weights


def fit_temperature(
    y_true: np.ndarray,
    scores: np.ndarray,
    balanced: bool,
) -> FittedCalibrator:
    logits = logit(scores)
    weights = class_balanced_weights(y_true) if balanced else np.ones(len(y_true), dtype=float)

    def loss(temperature: float) -> float:
        probabilities = np.clip(sigmoid(logits / temperature), EPS, 1.0 - EPS)
        losses = -(y_true * np.log(probabilities) + (1 - y_true) * np.log(1.0 - probabilities))
        return float(np.average(losses, weights=weights))

    grid = np.geomspace(0.05, 20.0, 240)
    best_temperature = float(grid[int(np.argmin([loss(float(value)) for value in grid]))])
    lower = max(best_temperature / 2.0, 0.01)
    upper = min(best_temperature * 2.0, 100.0)
    fine_grid = np.geomspace(lower, upper, 240)
    best_temperature = float(fine_grid[int(np.argmin([loss(float(value)) for value in fine_grid]))])
    return FittedCalibrator(
        name="temperature_balanced" if balanced else "temperature",
        model=None,
        temperature=best_temperature,
    )


def fit_calibrator(name: str, y_true: np.ndarray, scores: np.ndarray) -> FittedCalibrator:
    if name not in CALIBRATORS:
        raise ValueError(f"Unsupported calibrator: {name}")
    balanced = name.endswith("_balanced")
    base_name = name.removesuffix("_balanced")
    if len(np.unique(y_true)) != 2:
        raise ValueError("Calibration split must contain both classes")

    if base_name == "temperature":
        return fit_temperature(y_true, scores, balanced=balanced)
    if base_name == "platt":
        model = LogisticRegression(
            class_weight="balanced" if balanced else None,
            max_iter=1000,
            random_state=0,
        )
        model.fit(logit(scores).reshape(-1, 1), y_true)
        return FittedCalibrator(name=name, model=model)
    if base_name == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        weights = class_balanced_weights(y_true) if balanced else None
        model.fit(np.clip(scores, 0.0, 1.0), y_true, sample_weight=weights)
        return FittedCalibrator(name=name, model=model)
    raise ValueError(f"Unsupported calibrator: {name}")


def predict_calibrated(calibrator: FittedCalibrator, scores: np.ndarray) -> np.ndarray:
    if calibrator.name.startswith("temperature"):
        if calibrator.temperature is None:
            raise ValueError("Temperature calibrator is missing fitted temperature")
        return sigmoid(logit(scores) / calibrator.temperature)
    if calibrator.name.startswith("platt"):
        return calibrator.model.predict_proba(logit(scores).reshape(-1, 1))[:, 1]
    if calibrator.name.startswith("isotonic"):
        return calibrator.model.predict(np.clip(scores, 0.0, 1.0))
    raise ValueError(f"Unsupported calibrator: {calibrator.name}")
