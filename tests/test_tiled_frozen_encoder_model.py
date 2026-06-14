from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.evaluate_tiled_frozen_encoder_model import (
    aggregate_scores,
    patch_sklearn_predict_proba_compat,
    summarize_metrics,
    tile_boxes,
)


def test_tiled_frozen_encoder_tile_boxes_match_native_tiling_pattern() -> None:
    boxes = tile_boxes(width=1200, height=900, tile_size=512)

    assert len(boxes) == 9
    assert (0, 0, 512, 512) in boxes
    assert (344, 194, 856, 706) in boxes
    assert (688, 388, 1200, 900) in boxes


def test_tiled_frozen_encoder_aggregate_scores() -> None:
    scores = aggregate_scores(0.2, np.asarray([0.1, 0.6, 0.9], dtype=float))

    assert scores["global"] == 0.2
    assert scores["tile_mean"] == np.mean([0.1, 0.6, 0.9])
    assert scores["tile_max"] == 0.9
    assert scores["tile_top2_mean"] == np.mean([0.6, 0.9])
    assert scores["tile_std"] == np.std([0.1, 0.6, 0.9])


def test_tiled_frozen_encoder_summary_keeps_score_mode_order() -> None:
    metrics = pd.DataFrame(
        {
            "seed": [7, 7, 7, 7],
            "score_mode": ["tile_max", "global", "tile_top2_mean", "tile_mean"],
            "n_images": [10, 10, 10, 10],
            "mean_tiles": [4.0, 4.0, 4.0, 4.0],
            "accuracy": [0.7, 0.6, 0.75, 0.65],
            "roc_auc": [0.8, 0.7, 0.85, 0.75],
            "brier_score": [0.2, 0.3, 0.18, 0.25],
            "expected_calibration_error": [0.1, 0.2, 0.08, 0.15],
            "precision": [0.7, 0.6, 0.75, 0.65],
            "recall": [0.7, 0.6, 0.75, 0.65],
            "f1": [0.7, 0.6, 0.75, 0.65],
            "predicted_fake_rate": [0.5, 0.4, 0.55, 0.45],
        }
    )

    summary = summarize_metrics(metrics)

    assert summary["score_mode"].tolist() == ["global", "tile_mean", "tile_max", "tile_top2_mean"]
    assert summary.loc[summary["score_mode"] == "tile_top2_mean", "roc_auc_mean"].iloc[0] == 0.85


def test_sklearn_logistic_regression_compat_patch_recurses_into_pipeline_steps() -> None:
    logistic_cls = type("LogisticRegression", (), {})
    logistic = logistic_cls()
    pipeline = type("PipelineLike", (), {"steps": [("logreg", logistic)]})()

    patch_sklearn_predict_proba_compat(pipeline)

    assert logistic.multi_class == "auto"
