from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from sweep_reverse_tuned_fusion_constraints import (  # noqa: E402
    cap_label,
    run_sweep,
    summarize_by_cap,
)
from tests.test_reverse_fusion_source_holdout_tuning import _toy_reverse_fixture  # noqa: E402


def test_cap_label_formats_policy_names() -> None:
    assert cap_label(None) == "uncapped"
    assert cap_label(0.48) == "cap_0p48"


def test_summarize_by_cap_orders_frontier() -> None:
    selected = pd.DataFrame(
        {
            "constraint_policy": ["cap_0p48", "cap_0p48", "cap_0p45", "cap_0p45"],
            "seed": [1, 2, 1, 2],
            "fusion_c": [1.0, 1.0, 0.3, 0.3],
            "dropout_config": ["none", "none", "none", "none"],
            "source_fake_rate_cap": [0.48, 0.48, 0.45, 0.45],
            "target_accuracy": [0.7, 0.8, 0.6, 0.7],
            "target_roc_auc": [0.8, 0.9, 0.7, 0.8],
            "target_brier_score": [0.2, 0.2, 0.3, 0.3],
            "target_expected_calibration_error": [0.1, 0.1, 0.2, 0.2],
            "target_precision": [0.7, 0.8, 0.6, 0.7],
            "target_recall": [0.8, 0.9, 0.7, 0.8],
            "target_f1": [0.75, 0.85, 0.65, 0.75],
            "target_predicted_positive_rate": [0.6, 0.7, 0.5, 0.6],
            "source_predicted_positive_rate": [0.48, 0.47, 0.45, 0.44],
            "threshold_source_predicted_positive_rate": [0.48, 0.47, 0.45, 0.44],
            "selection_validation_utility_mean": [1.0, 1.1, 0.8, 0.9],
            "selection_validation_utility_min": [0.7, 0.8, 0.6, 0.7],
        }
    )

    summary = summarize_by_cap(selected)

    assert summary.iloc[0]["constraint_policy"] == "cap_0p48"
    assert summary.iloc[0]["target_accuracy_mean"] == 0.75
    assert "seed1" in summary.iloc[0]["selected_configs"]


def test_run_constraint_sweep_on_toy_fixture(tmp_path: Path) -> None:
    metadata = _toy_reverse_fixture(tmp_path)
    args = argparse.Namespace(
        run_root=str(tmp_path / "runs"),
        metadata=str(metadata),
        summary_dir=str(tmp_path / "assets"),
        report_path=str(tmp_path / "report.md"),
        seeds=[7],
        methods=["combined_v3", "resnet18"],
        fusion_cs="0.1",
        dropout_configs="none",
        source_fake_rate_caps="0.4,0.5",
        selection_score="min",
        real_validation_fraction=0.5,
        fake_detection_weight=1.0,
        real_clearance_weight=1.0,
        real_fpr_penalty=4.0,
        fake_miss_penalty=1.5,
        threshold_tiebreak="higher",
    )

    grid, selected, summary = run_sweep(args)

    assert set(summary["constraint_policy"]) == {"cap_0p4", "cap_0p5"}
    assert len(selected) == 2
    assert not grid.empty
