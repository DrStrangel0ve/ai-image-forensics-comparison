from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_submission_result_table_lint_validates_generated_tables(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    scripts_dir = repo_root / "scripts"
    assets_dir = repo_root / "reports" / "assets"
    scripts_dir.mkdir(parents=True)
    assets_dir.mkdir(parents=True)
    shutil.copyfile(
        ROOT / "scripts" / "build_submission_result_tables.py",
        scripts_dir / "build_submission_result_tables.py",
    )

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
    core_results = assets_dir / "publication_core_results.csv"
    pd.DataFrame(
        {
            "finding_id": finding_ids,
            "setting": [f"setting-{index}" for index in range(len(finding_ids))],
            "method": [f"method-{index}" for index in range(len(finding_ids))],
            "accuracy": [0.60 + index * 0.001 for index in range(len(finding_ids))],
            "auc": [0.70 + index * 0.001 for index in range(len(finding_ids))],
            "brier": [0.20 + index * 0.001 for index in range(len(finding_ids))],
            "ece": [0.10 + index * 0.001 for index in range(len(finding_ids))],
            "predicted_fake_rate": [0.30 + index * 0.001 for index in range(len(finding_ids))],
            "coverage": [pd.NA] * len(finding_ids),
            "decided_accuracy": [pd.NA] * len(finding_ids),
            "source": [f"source-{index}" for index in range(len(finding_ids))],
            "interpretation": [f"paper-use-{index}" for index in range(len(finding_ids))],
        }
    ).to_csv(core_results, index=False)

    claim_matrix = assets_dir / "claim_evidence_matrix.csv"
    pd.DataFrame(
        {
            "claim_id": ["claim"],
            "evidence_finding_ids": [",".join(finding_ids[:3])],
        }
    ).to_csv(claim_matrix, index=False)
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
    ).to_csv(assets_dir / "source_holdout_generator_stress.csv", index=False)
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
    ).to_csv(assets_dir / "reconstruction_v2_probe_mean_summary.csv", index=False)

    report_out = repo_root / "reports" / "submission_result_tables_2026_06_14.md"
    subprocess.run(
        [
            sys.executable,
            str(scripts_dir / "build_submission_result_tables.py"),
            "--core-results",
            "reports/assets/publication_core_results.csv",
            "--out-dir",
            "reports/assets",
            "--report-out",
            "reports/submission_result_tables_2026_06_14.md",
        ],
        cwd=repo_root,
        check=True,
    )

    out_path = tmp_path / "submission_result_tables_lint.md"
    checks_out = tmp_path / "submission_result_tables_lint.csv"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "lint_submission_result_tables.py"),
            "--repo-root",
            str(repo_root),
            "--core-results",
            str(core_results),
            "--claim-matrix",
            str(claim_matrix),
            "--manifest",
            str(assets_dir / "submission_result_table_manifest.csv"),
            "--report",
            str(report_out),
            "--out-path",
            str(out_path),
            "--checks-out",
            str(checks_out),
        ],
        cwd=ROOT,
        check=True,
    )

    report = out_path.read_text(encoding="utf-8")
    checks = pd.read_csv(checks_out)
    assert "Status: **PASS**" in report
    assert checks["passed"].all()
    assert "robustness deltas recompute from clean baseline" in checks["check"].tolist()
    assert "source_holdout_stress sorted by utility" in checks["check"].tolist()
    assert "reconstruction_ablation has expected methods" in checks["check"].tolist()
