from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_submission_result_table_builder_writes_compact_tables_and_deltas(tmp_path: Path) -> None:
    core_results = tmp_path / "publication_core_results.csv"
    source_stress = tmp_path / "source_holdout_generator_stress.csv"
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

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_submission_result_tables.py"),
            "--core-results",
            str(core_results),
            "--source-stress",
            str(source_stress),
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
    report = report_out.read_text(encoding="utf-8")

    assert set(manifest["table_id"]) == {
        "same_domain_anchor",
        "transfer_frontier",
        "reverse_operating_points",
        "robustness_stress",
        "source_holdout_stress",
    }
    assert "Submission Result Tables" in report
    assert "Reverse Tuned-Fusion Robustness Stress" in report
    assert "Held-Out Generator Stress" in report
    assert "delta_accuracy_vs_clean" in robustness.columns
    assert source_table["heldout_source"].tolist() == ["sd3", "sdxl"]
    tiled = robustness[robustness["finding_id"] == "ms_to_ishu_tuned_fusion_native_tiling_best"].iloc[0]
    assert round(float(tiled["delta_accuracy_vs_clean"]), 4) == 0.05
    assert round(float(tiled["delta_auc_vs_clean"]), 4) == 0.05
