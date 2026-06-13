from __future__ import annotations

import numpy as np

from scripts.evaluate_tiled_feature_model import aggregate_scores, tile_boxes


def test_tile_boxes_cover_corners_and_center_for_large_image() -> None:
    boxes = tile_boxes(width=1200, height=900, tile_size=512)

    assert len(boxes) == 9
    assert (0, 0, 512, 512) in boxes
    assert (344, 194, 856, 706) in boxes
    assert (688, 388, 1200, 900) in boxes


def test_tile_boxes_use_whole_image_when_smaller_than_tile() -> None:
    assert tile_boxes(width=320, height=240, tile_size=512) == [(0, 0, 320, 240)]


def test_aggregate_scores_reports_global_and_tile_modes() -> None:
    scores = aggregate_scores(0.25, np.asarray([0.1, 0.4, 0.8], dtype=float))

    assert scores["global"] == 0.25
    assert scores["tile_mean"] == np.mean([0.1, 0.4, 0.8])
    assert scores["tile_max"] == 0.8
    assert scores["tile_top2_mean"] == np.mean([0.4, 0.8])
    assert scores["tile_std"] == np.std([0.1, 0.4, 0.8])
