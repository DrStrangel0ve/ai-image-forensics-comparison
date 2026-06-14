from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_calibration_operating_modes_selects_objective_specific_winners(tmp_path: Path) -> None:
    core_results = tmp_path / "publication_core_results.csv"
    source_holdout = tmp_path / "score_fusion_source_holdout_calibration_summary.csv"
    tiled_choices = tmp_path / "tiled_dinov2_calibration_tradeoff_choices.csv"
    out_path = tmp_path / "calibration_operating_modes.md"
    csv_out = tmp_path / "calibration_operating_modes.csv"

    pd.DataFrame(
        [
            {
                "finding_id": "a",
                "setting": "Ishu -> source-balanced MS COCOAI",
                "method": "method accuracy",
                "accuracy": 0.70,
                "auc": 0.80,
                "brier": 0.30,
                "ece": 0.20,
                "predicted_fake_rate": 0.40,
                "source": "a.csv",
            },
            {
                "finding_id": "b",
                "setting": "Ishu -> source-balanced MS COCOAI",
                "method": "method auc",
                "accuracy": 0.68,
                "auc": 0.90,
                "brier": 0.29,
                "ece": 0.22,
                "predicted_fake_rate": 0.30,
                "source": "b.csv",
            },
            {
                "finding_id": "c",
                "setting": "Ishu -> source-balanced MS COCOAI",
                "method": "method calibrated",
                "accuracy": 0.62,
                "auc": 0.72,
                "brier": 0.21,
                "ece": 0.10,
                "predicted_fake_rate": 0.20,
                "source": "c.csv",
            },
        ]
    ).to_csv(core_results, index=False)

    pd.DataFrame(
        [
            {
                "method": "fusion a",
                "calibrator": "temperature",
                "mean_calibrated_accuracy": 0.74,
                "mean_calibrated_brier_score": 0.20,
                "mean_calibrated_ece": 0.14,
                "mean_calibrated_fake_detection": 0.24,
                "mean_calibrated_real_fpr": 0.05,
            },
            {
                "method": "fusion b",
                "calibrator": "balanced",
                "mean_calibrated_accuracy": 0.73,
                "mean_calibrated_brier_score": 0.18,
                "mean_calibrated_ece": 0.12,
                "mean_calibrated_fake_detection": 0.30,
                "mean_calibrated_real_fpr": 0.08,
            },
        ]
    ).to_csv(source_holdout, index=False)

    pd.DataFrame(
        {
            "variant": ["blur1", "jpeg30", "screenshot"],
            "best_accuracy_mode": ["tile_max", "tile_max", "global"],
            "best_accuracy_delta": [0.01, 0.02, 0.00],
            "best_auc_mode": ["tile_max", "tile_max", "tile_max"],
            "best_auc_delta": [0.02, 0.03, 0.01],
            "best_brier_mode": ["tile_mean", "tile_mean", "tile_max"],
            "best_brier_delta": [-0.01, -0.02, -0.01],
            "best_ece_mode": ["tile_mean", "global", "tile_mean"],
            "best_ece_delta": [-0.02, 0.00, -0.01],
        }
    ).to_csv(tiled_choices, index=False)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_calibration_operating_modes.py"),
            "--core-results",
            str(core_results),
            "--source-holdout-calibration",
            str(source_holdout),
            "--tiled-dino-choices",
            str(tiled_choices),
            "--out-path",
            str(out_path),
            "--csv-out",
            str(csv_out),
        ],
        cwd=ROOT,
        check=True,
    )

    text = out_path.read_text(encoding="utf-8")
    modes = pd.read_csv(csv_out)
    by_objective = modes.set_index("objective")

    assert "Calibration Operating Modes" in text
    assert by_objective.loc["threshold_accuracy", "selected_method"] == "method accuracy"
    assert by_objective.loc["ranking_auc", "selected_method"] == "method auc"
    assert by_objective.loc["probability_brier", "selected_method"] == "method calibrated"
    assert by_objective.loc["reliability_ece", "selected_method"] == "method calibrated"
    assert by_objective.loc["source_holdout_brier", "selected_method"] == "fusion b"
    assert by_objective.loc["source_holdout_fake_detection", "selected_method"] == "fusion b"
    assert by_objective.loc["tiled_dino_accuracy", "selected_mode"] == "tile_max"
    assert by_objective.loc["tiled_dino_brier", "selected_mode"] == "tile_mean"
    assert "mode_counts" in set(modes["secondary_metric"])
