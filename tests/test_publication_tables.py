from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_publication_table_builder_writes_core_csv_and_markdown(tmp_path: Path) -> None:
    physics_report = tmp_path / "physics.md"
    calibration_summary = tmp_path / "calibration.csv"
    clip_calibration = tmp_path / "clip_calibration.csv"
    clip_triage = tmp_path / "clip_triage.csv"
    reverse_all = tmp_path / "reverse_all.csv"
    reverse_regularization = tmp_path / "reverse_regularization.csv"
    reverse_cap = tmp_path / "reverse_cap.csv"
    reverse_utility = tmp_path / "reverse_utility.csv"
    reverse_holdout = tmp_path / "reverse_holdout.csv"
    reverse_tuned = tmp_path / "reverse_tuned.csv"
    reverse_constraint_sweep = tmp_path / "reverse_constraint_sweep.csv"
    reverse_jpeg70 = tmp_path / "reverse_jpeg70.csv"
    out_dir = tmp_path / "assets"

    physics_report.write_text(
        """
# Physics Report

## Where Physics Helps

| method | accuracy_mean | roc_auc_mean | accuracy_wins | auc_wins |
| --- | ---: | ---: | ---: | ---: |
| `combined_v3` | 0.8246 | 0.8942 | 1 | 1 |
| ResNet-18 | 0.8246 | 0.8927 | 0 | 0 |
| physics-guided ResNet-18 + `combined_v3` | 0.8450 | 0.9177 | 2 | 2 |
""".strip(),
        encoding="utf-8",
    )

    pd.DataFrame(
        {
            "method": ["combined_v3", "resnet18", "physics_guided", "convnext_tiny_frozen", "scp_fusion_v0"],
            "mean_accuracy": [0.55, 0.58, 0.61, 0.62, 0.59],
            "mean_roc_auc": [0.58, 0.65, 0.66, 0.71, 0.73],
            "mean_brier_score": [0.34, 0.35, 0.33, 0.32, 0.31],
            "mean_expected_calibration_error": [0.29, 0.34, 0.31, 0.32, 0.31],
            "mean_predicted_positive_rate": [0.17, 0.25, 0.27, 0.19, 0.14],
        }
    ).to_csv(calibration_summary, index=False)

    pd.DataFrame(
        {
            "method": [
                "scp_fusion_dinov2",
                "scp_fusion_clip",
                "scp_fusion_all_foundation",
                "clip_standalone",
            ],
            "mean_accuracy": [0.60, 0.62, 0.62, 0.64],
            "mean_roc_auc": [0.75, 0.79, 0.80, 0.86],
            "mean_brier_score": [0.31, 0.31, 0.31, 0.32],
            "mean_expected_calibration_error": [0.31, 0.33, 0.33, 0.34],
            "mean_predicted_positive_rate": [0.14, 0.13, 0.13, 0.16],
        }
    ).to_csv(clip_calibration, index=False)

    pd.DataFrame(
        {
            "method": ["scp_fusion_all_foundation", "clip_standalone"] * 2,
            "score_mode": ["raw", "raw", "temperature_balanced", "temperature_balanced"],
            "mean_test_coverage": [0.30, 0.47, 0.29, 0.46],
            "mean_test_triage_accuracy": [0.87, 0.93, 0.86, 0.92],
        }
    ).to_csv(clip_triage, index=False)

    pd.DataFrame(
        {
            "method": [
                "physics_guided_resnet18_combined_v3",
                "clip_vit_b_32",
                "score_fusion_all6_temp_balanced",
            ],
            "split": ["ms_cocoai_to_ishu_test"] * 3,
            "accuracy": [0.69, 0.62, 0.66],
            "auc": [0.74, 0.82, 0.83],
            "brier": [0.24, 0.33, 0.31],
            "ece": [0.19, 0.36, 0.33],
            "predicted_fake_rate": [0.53, 0.86, 0.81],
        }
    ).to_csv(reverse_all, index=False)

    pd.DataFrame(
        {
            "config": ["score_fusion_all6_dropout_mean_r35x8"],
            "split": ["ms_cocoai_to_ishu_test"],
            "accuracy": [0.65],
            "auc": [0.84],
            "brier": [0.31],
            "ece": [0.33],
            "predicted_fake_rate": [0.82],
        }
    ).to_csv(reverse_regularization, index=False)

    pd.DataFrame(
        {
            "config": ["cap_0p48"],
            "variant": ["ishu_test"],
            "accuracy": [0.72],
            "auc": [0.83],
            "brier": [0.22],
            "ece": [0.21],
            "predicted_fake_rate": [0.62],
        }
    ).to_csv(reverse_cap, index=False)

    pd.DataFrame(
        {
            "selection_policy": [
                "source_utility_unconstrained",
                "source_utility_cap_0p48",
            ],
            "target_accuracy_mean": [0.65, 0.72],
            "target_roc_auc_mean": [0.83, 0.83],
            "target_brier_score_mean": [0.29, 0.22],
            "target_expected_calibration_error_mean": [0.32, 0.21],
            "target_predicted_positive_rate_mean": [0.82, 0.62],
        }
    ).to_csv(reverse_utility, index=False)

    pd.DataFrame(
        {
            "selection_policy": [
                "source_holdout_mean_utility_unconstrained",
                "source_holdout_mean_utility_cap_0p48",
            ],
            "target_accuracy_mean": [0.65, 0.72],
            "target_roc_auc_mean": [0.83, 0.83],
            "target_brier_score_mean": [0.29, 0.22],
            "target_expected_calibration_error_mean": [0.32, 0.21],
            "target_predicted_positive_rate_mean": [0.82, 0.62],
        }
    ).to_csv(reverse_holdout, index=False)

    pd.DataFrame(
        {
            "selection_policy": ["source_holdout_tuned_fusion"],
            "target_accuracy_mean": [0.73],
            "target_roc_auc_mean": [0.84],
            "target_brier_score_mean": [0.27],
            "target_expected_calibration_error_mean": [0.29],
            "target_predicted_positive_rate_mean": [0.68],
        }
    ).to_csv(reverse_tuned, index=False)

    pd.DataFrame(
        {
            "constraint_policy": ["cap_0p4", "cap_0p48"],
            "target_accuracy_mean": [0.76, 0.73],
            "target_roc_auc_mean": [0.84, 0.83],
            "target_brier_score_mean": [0.28, 0.27],
            "target_expected_calibration_error_mean": [0.30, 0.29],
            "target_predicted_positive_rate_mean": [0.52, 0.68],
        }
    ).to_csv(reverse_constraint_sweep, index=False)

    pd.DataFrame(
        {
            "variant": ["jpeg70"],
            "variant_policy": ["cap_0p4"],
            "target_accuracy_mean": [0.77],
            "target_roc_auc_mean": [0.85],
            "target_brier_score_mean": [0.26],
            "target_expected_calibration_error_mean": [0.27],
            "target_predicted_positive_rate_mean": [0.47],
        }
    ).to_csv(reverse_jpeg70, index=False)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_publication_tables.py"),
            "--physics-guided-report",
            str(physics_report),
            "--calibration-summary",
            str(calibration_summary),
            "--clip-calibration-summary",
            str(clip_calibration),
            "--clip-triage-5pct",
            str(clip_triage),
            "--reverse-all-methods",
            str(reverse_all),
            "--reverse-fusion-regularization",
            str(reverse_regularization),
            "--reverse-threshold-cap",
            str(reverse_cap),
            "--reverse-model-utility-selection",
            str(reverse_utility),
            "--reverse-source-holdout-selection",
            str(reverse_holdout),
            "--reverse-source-holdout-tuned-fusion",
            str(reverse_tuned),
            "--reverse-tuned-fusion-constraint-sweep",
            str(reverse_constraint_sweep),
            "--reverse-tuned-fusion-jpeg70-robustness",
            str(reverse_jpeg70),
            "--out-dir",
            str(out_dir),
        ],
        cwd=ROOT,
        check=True,
    )

    csv_path = out_dir / "publication_core_results.csv"
    markdown_path = out_dir / "publication_core_results.md"
    assert csv_path.exists()
    assert markdown_path.exists()

    frame = pd.read_csv(csv_path)
    assert "ishu_same_physics_guided" in set(frame["finding_id"])
    assert "ishu_to_ms_triage5_clip_standalone" in set(frame["finding_id"])
    assert "ms_to_ishu_source_cap_accuracy" in set(frame["finding_id"])
    assert "ms_to_ishu_source_utility_unconstrained" in set(frame["finding_id"])
    assert "ms_to_ishu_source_utility_cap_0p48" in set(frame["finding_id"])
    assert "ms_to_ishu_source_holdout_mean_utility_unconstrained" in set(frame["finding_id"])
    assert "ms_to_ishu_source_holdout_mean_utility_cap_0p48" in set(frame["finding_id"])
    assert "ms_to_ishu_source_holdout_tuned_fusion" in set(frame["finding_id"])
    assert "ms_to_ishu_tuned_fusion_constraint_sweep_best" in set(frame["finding_id"])
    assert "ms_to_ishu_tuned_fusion_jpeg70" in set(frame["finding_id"])
    assert frame.loc[frame["finding_id"] == "ishu_same_physics_guided", "auc"].iloc[0] == 0.9177
    assert "Publication Core Results Table" in markdown_path.read_text(encoding="utf-8")
