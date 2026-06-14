from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_submission_result_table_builder_writes_compact_tables_and_deltas(tmp_path: Path) -> None:
    core_results = tmp_path / "publication_core_results.csv"
    source_stress = tmp_path / "source_holdout_generator_stress.csv"
    reconstruction_v2_probe = tmp_path / "reconstruction_v2_probe_mean_summary.csv"
    out_dir = tmp_path / "assets"
    report_out = tmp_path / "submission_result_tables.md"
    finding_ids = [
        "ishu_same_combined_v3",
        "ishu_same_resnet18",
        "ishu_same_physics_guided",
        "ishu_to_ms_combined_v3",
        "ishu_to_ms_resnet18",
        "ishu_to_ms_physics_guided",
        "ishu_to_ms_convnext_tiny_frozen",
        "ishu_to_ms_scp_fusion_v0",
        "ishu_to_ms_scp_fusion_dinov2",
        "ishu_to_ms_scp_fusion_all_foundation",
        "ishu_to_ms_clip_standalone",
        "ishu_to_ms_triage5_clip_standalone",
        "ms_to_ishu_physics_guided",
        "ms_to_ishu_clip_vit_b_32",
        "ms_to_ishu_score_fusion_all6_temp_balanced",
        "ms_to_ishu_branch_dropout_auc",
        "ms_to_ishu_source_cap_accuracy",
        "ms_to_ishu_source_holdout_tuned_fusion",
        "ms_to_ishu_tuned_fusion_constraint_sweep_best",
        "ms_to_ishu_tuned_fusion_native_tiling_best",
        "ms_to_ishu_tuned_fusion_jpeg70",
        "ms_to_ishu_tuned_fusion_jpeg50",
        "ms_to_ishu_tuned_fusion_jpeg30",
        "ms_to_ishu_tuned_fusion_noise3",
        "ms_to_ishu_tuned_fusion_social_square",
        "ms_to_ishu_tuned_fusion_social_720p",
        "ms_to_ishu_tuned_fusion_crop85",
        "ms_to_ishu_tuned_fusion_screenshot",
        "ms_to_ishu_tuned_fusion_blur1",
        "ms_to_ishu_tuned_fusion_resize_half",
    ]
    pd.DataFrame(
        {
            "finding_id": finding_ids,
            "setting": ["setting"] * len(finding_ids),
            "method": ["method"] * len(finding_ids),
            "accuracy": [0.70] * len(finding_ids),
            "auc": [0.80] * len(finding_ids),
            "brier": [0.20] * len(finding_ids),
            "ece": [0.10] * len(finding_ids),
            "predicted_fake_rate": [0.50] * len(finding_ids),
            "coverage": [pd.NA] * len(finding_ids),
            "decided_accuracy": [pd.NA] * len(finding_ids),
            "source": ["source"] * len(finding_ids),
            "interpretation": ["paper use"] * len(finding_ids),
        }
    ).assign(
        accuracy=lambda frame: frame["accuracy"].where(
            frame["finding_id"] != "ms_to_ishu_tuned_fusion_native_tiling_best", 0.75
        ),
        auc=lambda frame: frame["auc"].where(
            frame["finding_id"] != "ms_to_ishu_tuned_fusion_native_tiling_best", 0.85
        ),
    ).to_csv(core_results, index=False)
    pd.DataFrame(
        [
            {
                "selection_policy": "source_holdout_mean_utility_cap_0p48",
                "heldout_source_name": "sd3",
                "source_holdout_utility_mean": 1.42,
                "source_holdout_recall_mean": 0.80,
                "source_holdout_fake_miss_rate_mean": 0.20,
                "source_holdout_predicted_positive_rate_mean": 0.13,
            },
            {
                "selection_policy": "source_holdout_mean_utility_cap_0p48",
                "heldout_source_name": "sdxl",
                "source_holdout_utility_mean": 1.90,
                "source_holdout_recall_mean": 0.98,
                "source_holdout_fake_miss_rate_mean": 0.02,
                "source_holdout_predicted_positive_rate_mean": 0.18,
            },
        ]
    ).to_csv(source_stress, index=False)
    pd.DataFrame(
        [
            {
                "setting": "ishu_same_bounded",
                "candidate": "combined_v3_logreg",
                "accuracy_mean": 0.75,
                "roc_auc_mean": 0.81,
                "brier_score_mean": 0.18,
                "expected_calibration_error_mean": 0.14,
            },
            {
                "setting": "ishu_same_bounded",
                "candidate": "reconstruction_lite_logreg",
                "accuracy_mean": 0.69,
                "roc_auc_mean": 0.73,
                "brier_score_mean": 0.21,
                "expected_calibration_error_mean": 0.10,
            },
            {
                "setting": "ishu_same_bounded",
                "candidate": "reconstruction_v2_logreg",
                "accuracy_mean": 0.71,
                "roc_auc_mean": 0.76,
                "brier_score_mean": 0.20,
                "expected_calibration_error_mean": 0.12,
            },
            {
                "setting": "ishu_to_ms_cocoai_bounded",
                "candidate": "combined_v3_logreg",
                "accuracy_mean": 0.55,
                "roc_auc_mean": 0.58,
                "brier_score_mean": 0.33,
                "expected_calibration_error_mean": 0.28,
            },
            {
                "setting": "ishu_to_ms_cocoai_bounded",
                "candidate": "reconstruction_lite_logreg",
                "accuracy_mean": 0.60,
                "roc_auc_mean": 0.64,
                "brier_score_mean": 0.24,
                "expected_calibration_error_mean": 0.04,
            },
            {
                "setting": "ishu_to_ms_cocoai_bounded",
                "candidate": "reconstruction_v2_logreg",
                "accuracy_mean": 0.57,
                "roc_auc_mean": 0.60,
                "brier_score_mean": 0.26,
                "expected_calibration_error_mean": 0.10,
            },
        ]
    ).to_csv(reconstruction_v2_probe, index=False)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_submission_result_tables.py"),
            "--core-results",
            str(core_results),
            "--source-stress",
            str(source_stress),
            "--reconstruction-v2-probe",
            str(reconstruction_v2_probe),
            "--out-dir",
            str(out_dir),
            "--report-out",
            str(report_out),
        ],
        cwd=ROOT,
        check=True,
    )

    manifest = pd.read_csv(out_dir / "submission_result_table_manifest.csv")
    robustness = pd.read_csv(out_dir / "submission_table_robustness_stress.csv")
    source_table = pd.read_csv(out_dir / "submission_table_source_holdout_stress.csv")
    reconstruction = pd.read_csv(out_dir / "submission_table_reconstruction_ablation.csv")
    report = report_out.read_text(encoding="utf-8")

    assert set(manifest["table_id"]) == {
        "same_domain_anchor",
        "transfer_frontier",
        "reverse_operating_points",
        "robustness_stress",
        "source_holdout_stress",
        "reconstruction_ablation",
    }
    assert "Submission Result Tables" in report
    assert "Reverse Tuned-Fusion Robustness Stress" in report
    assert "Held-Out Generator Stress" in report
    assert "Reconstruction Residual Ablation" in report
    assert "delta_accuracy_vs_clean" in robustness.columns
    assert "delta_auc_vs_reconstruction_lite" in reconstruction.columns
    assert source_table["heldout_source"].tolist() == ["sd3", "sdxl"]
    assert reconstruction["method"].tolist() == [
        "combined_v3",
        "reconstruction_lite",
        "reconstruction_v2",
        "combined_v3",
        "reconstruction_lite",
        "reconstruction_v2",
    ]
    same_v2 = reconstruction[
        (reconstruction["setting"] == "ishu_same_bounded")
        & (reconstruction["method"] == "reconstruction_v2")
    ].iloc[0]
    transfer_v2 = reconstruction[
        (reconstruction["setting"] == "ishu_to_ms_cocoai_bounded")
        & (reconstruction["method"] == "reconstruction_v2")
    ].iloc[0]
    assert round(float(same_v2["delta_auc_vs_reconstruction_lite"]), 4) == 0.03
    assert round(float(transfer_v2["delta_auc_vs_reconstruction_lite"]), 4) == -0.04
    tiled = robustness[robustness["finding_id"] == "ms_to_ishu_tuned_fusion_native_tiling_best"].iloc[0]
    assert round(float(tiled["delta_accuracy_vs_clean"]), 4) == 0.05
    assert round(float(tiled["delta_auc_vs_clean"]), 4) == 0.05
