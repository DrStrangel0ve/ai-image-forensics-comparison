from __future__ import annotations

import numpy as np

from forensic_compare.metrics import best_threshold
from scripts.summarize_threshold_calibration import _best_threshold


def test_best_threshold_can_move_above_default() -> None:
    y_true = np.asarray([0, 0, 1, 1])
    scores = np.asarray([0.20, 0.55, 0.60, 0.80])

    threshold, metrics = _best_threshold(y_true, scores, objective="accuracy")

    assert threshold > 0.5
    assert metrics["accuracy"] == 1.0


def test_shared_best_threshold_matches_script_wrapper() -> None:
    y_true = np.asarray([0, 0, 1, 1])
    scores = np.asarray([0.20, 0.55, 0.60, 0.80])

    assert best_threshold(y_true, scores, objective="accuracy") == _best_threshold(
        y_true, scores, objective="accuracy"
    )
