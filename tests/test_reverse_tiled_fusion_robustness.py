from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from evaluate_reverse_tiled_fusion_robustness import (  # noqa: E402
    resolved_asset_prefix,
    run_tiled_robustness,
)
from tests.test_reverse_fusion_source_holdout_tuning import (  # noqa: E402
    _toy_reverse_fixture,
    _write_predictions,
)


def _selected_configs(path: Path) -> None:
    pd.DataFrame(
        {
            "constraint_policy": ["cap_0p5"],
            "seed": [7],
            "fusion_c": [0.1],
            "dropout_config": ["none"],
            "source_fake_rate_cap": [0.5],
            "selection_validation_utility_mean": [1.0],
            "selection_validation_utility_min": [0.5],
        }
    ).to_csv(path, index=False)


def _robust_predictions(root: Path) -> None:
    robust_root = root / "runs" / "robust"
    paths = [root / "robust_target" / f"target_{index}.jpg" for index in range(6)]
    labels = [0, 0, 0, 1, 1, 1]
    _write_predictions(
        robust_root / "seed7" / "combined_v3" / "predictions.csv",
        labels,
        [0.3, 0.35, 0.5, 0.65, 0.72, 0.86],
        paths,
    )
    _write_predictions(
        robust_root / "seed7" / "resnet18" / "predictions.csv",
        labels,
        [0.2, 0.25, 0.45, 0.72, 0.68, 0.9],
        paths,
    )


def _tile_detail(path: Path, root: Path) -> None:
    paths = [root / "robust_target" / f"target_{index}.jpg" for index in range(6)]
    pd.DataFrame(
        {
            "seed": [7] * 6,
            "path": [str(value) for value in paths],
            "y_true": [0, 0, 0, 1, 1, 1],
            "global_score": [0.3, 0.35, 0.5, 0.65, 0.72, 0.86],
            "tile_mean_score": [0.25, 0.34, 0.48, 0.68, 0.74, 0.88],
            "tile_max_score": [0.4, 0.52, 0.7, 0.9, 0.95, 0.98],
            "tile_top2_mean_score": [0.35, 0.45, 0.6, 0.82, 0.9, 0.95],
        }
    ).to_csv(path, index=False)


def test_run_tiled_robustness_on_toy_fixture(tmp_path: Path) -> None:
    metadata = _toy_reverse_fixture(tmp_path)
    selected_path = tmp_path / "selected.csv"
    detail_path = tmp_path / "tile_detail.csv"
    _selected_configs(selected_path)
    _robust_predictions(tmp_path)
    _tile_detail(detail_path, tmp_path)

    args = argparse.Namespace(
        run_root=str(tmp_path / "runs"),
        metadata=str(metadata),
        selected_configs=str(selected_path),
        robust_root=str(tmp_path / "runs" / "robust"),
        variant="toy_variant",
        constraint_policy="cap_0p5",
        tile_detail=str(detail_path),
        tile_branch="combined_v3",
        tile_score_modes=["global", "tile_top2_mean"],
        seeds=[7],
        methods=["combined_v3", "resnet18"],
        dropout_configs="none",
        threshold_tiebreak="higher",
        fake_detection_weight=1.0,
        real_clearance_weight=1.0,
        real_fpr_penalty=4.0,
        fake_miss_penalty=1.5,
        alignment_tolerance=1e-5,
        summary_dir=str(tmp_path / "assets"),
        asset_prefix=None,
        report_path=str(tmp_path / "report.md"),
    )

    detail, summary = run_tiled_robustness(args)

    assert resolved_asset_prefix(args) == "ms_cocoai_to_ishu_tuned_fusion_toy_variant_tiled_combined_v3"
    assert set(detail["score_mode"]) == {"global", "tile_top2_mean"}
    assert set(summary["score_mode"]) == {"global", "tile_top2_mean"}
    assert 0.0 <= summary["target_roc_auc_mean"].iloc[0] <= 1.0
