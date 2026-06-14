from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_claim_evidence_matrix_validates_and_writes_artifacts(tmp_path: Path) -> None:
    core_results = tmp_path / "publication_core_results.csv"
    tiled_dino_tradeoff = tmp_path / "tiled_dinov2_calibration_tradeoff.csv"
    reconstruction_v2_probe = tmp_path / "reconstruction_v2_probe_mean_summary.csv"
    out_dir = tmp_path / "assets"
    finding_ids = [
        "ishu_same_combined_v3",
        "ishu_same_resnet18",
        "ishu_same_physics_guided",
        "ishu_to_ms_resnet18",
        "ishu_to_ms_physics_guided",
        "ishu_to_ms_scp_fusion_v0",
        "ishu_to_ms_scp_fusion_all_foundation",
        "ishu_to_ms_clip_standalone",
        "ishu_same_combined_v4_raw",
        "ishu_same_combined_v4_selectk60",
        "ishu_to_ms_combined_v4_raw",
        "ishu_to_ms_combined_v4_selectk60",
        "ishu_to_ms_triage5_scp_fusion_all_foundation",
        "ishu_to_ms_triage5_clip_standalone",
        "ms_to_ishu_physics_guided",
        "ms_to_ishu_clip_vit_b_32",
        "ms_to_ishu_score_fusion_all6_temp_balanced",
        "ms_to_ishu_branch_dropout_auc",
        "ms_to_ishu_source_cap_accuracy",
        "ms_to_ishu_source_utility_unconstrained",
        "ms_to_ishu_source_utility_cap_0p48",
        "ms_to_ishu_source_holdout_mean_utility_unconstrained",
        "ms_to_ishu_source_holdout_mean_utility_cap_0p48",
        "ms_to_ishu_source_holdout_tuned_fusion",
        "ms_to_ishu_tuned_fusion_constraint_sweep_best",
        "ms_to_ishu_tuned_fusion_native_tiling_best",
        "ms_to_ishu_tuned_fusion_jpeg70",
        "ms_to_ishu_tuned_fusion_blur1",
        "ms_to_ishu_tuned_fusion_resize_half",
        "ms_to_ishu_tuned_fusion_crop85",
        "ms_to_ishu_tuned_fusion_jpeg50",
        "ms_to_ishu_tuned_fusion_jpeg30",
        "ms_to_ishu_tuned_fusion_noise3",
        "ms_to_ishu_tuned_fusion_screenshot",
        "ms_to_ishu_tuned_fusion_social_square",
        "ms_to_ishu_tuned_fusion_social_720p",
    ]
    pd.DataFrame(
        {
            "finding_id": finding_ids,
            "setting": ["setting"] * len(finding_ids),
            "method": ["method"] * len(finding_ids),
            "accuracy": [0.60] * len(finding_ids),
            "auc": [0.80] * len(finding_ids),
            "brier": [0.30] * len(finding_ids),
            "ece": [0.20] * len(finding_ids),
            "predicted_fake_rate": [0.50] * len(finding_ids),
            "coverage": [pd.NA] * len(finding_ids),
            "decided_accuracy": [pd.NA] * len(finding_ids),
            "source": ["source"] * len(finding_ids),
            "interpretation": ["interpretation"] * len(finding_ids),
        }
    ).to_csv(core_results, index=False)
    pd.DataFrame(
        [
            {
                "variant": variant,
                "score_mode": score_mode,
                "target_accuracy_mean_delta_vs_global": acc_delta,
                "target_roc_auc_mean_delta_vs_global": auc_delta,
                "target_brier_score_mean_improved_vs_global": brier_improved,
                "target_expected_calibration_error_mean_improved_vs_global": ece_improved,
            }
            for variant in ["blur1", "jpeg30"]
            for score_mode, acc_delta, auc_delta, brier_improved, ece_improved in [
                ("global", 0.0, 0.0, False, False),
                ("tile_mean", 0.002, 0.004, True, True),
                ("tile_max", 0.014, 0.016, False, False),
            ]
        ]
    ).to_csv(tiled_dino_tradeoff, index=False)
    pd.DataFrame(
        [
            {
                "setting": "ishu_same_bounded",
                "candidate": "combined_v3_logreg",
                "accuracy_mean": 0.75,
                "roc_auc_mean": 0.81,
            },
            {
                "setting": "ishu_same_bounded",
                "candidate": "reconstruction_lite_logreg",
                "accuracy_mean": 0.69,
                "roc_auc_mean": 0.73,
            },
            {
                "setting": "ishu_same_bounded",
                "candidate": "reconstruction_v2_logreg",
                "accuracy_mean": 0.71,
                "roc_auc_mean": 0.76,
            },
            {
                "setting": "ishu_to_ms_cocoai_bounded",
                "candidate": "combined_v3_logreg",
                "accuracy_mean": 0.55,
                "roc_auc_mean": 0.58,
            },
            {
                "setting": "ishu_to_ms_cocoai_bounded",
                "candidate": "reconstruction_lite_logreg",
                "accuracy_mean": 0.60,
                "roc_auc_mean": 0.64,
            },
            {
                "setting": "ishu_to_ms_cocoai_bounded",
                "candidate": "reconstruction_v2_logreg",
                "accuracy_mean": 0.57,
                "roc_auc_mean": 0.60,
            },
        ]
    ).to_csv(reconstruction_v2_probe, index=False)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_claim_evidence_matrix.py"),
            "--core-results",
            str(core_results),
            "--tiled-dino-tradeoff",
            str(tiled_dino_tradeoff),
            "--reconstruction-v2-probe",
            str(reconstruction_v2_probe),
            "--out-dir",
            str(out_dir),
        ],
        cwd=ROOT,
        check=True,
    )

    csv_path = out_dir / "claim_evidence_matrix.csv"
    markdown_path = out_dir / "claim_evidence_matrix.md"
    assert csv_path.exists()
    assert markdown_path.exists()

    frame = pd.read_csv(csv_path)
    assert "clip_transfer_frontier" in set(frame["claim_id"])
    assert "combined_v4_is_ablation_candidate" in set(frame["claim_id"])
    assert "reconstruction_residuals_are_source_sensitive" in set(frame["claim_id"])
    assert "transform_stress_exposes_failure_modes" in set(frame["claim_id"])
    assert "tiled_dino_mode_tradeoff" in set(frame["claim_id"])
    v4_row = frame[frame["claim_id"] == "combined_v4_is_ablation_candidate"].iloc[0]
    assert v4_row["status"] == "ready_with_caveat"
    assert "ishu_to_ms_combined_v4_selectk60" in v4_row["evidence_finding_ids"]
    stress_row = frame[frame["claim_id"] == "transform_stress_exposes_failure_modes"].iloc[0]
    assert stress_row["primary_artifact"] == "reports/robustness_failure_ranking_2026_06_14.md"
    assert "ms_to_ishu_tuned_fusion_resize_half" in stress_row["evidence_finding_ids"]
    tiled_row = frame[frame["claim_id"] == "tiled_dino_mode_tradeoff"].iloc[0]
    assert tiled_row["primary_artifact"] == "reports/tiled_dinov2_calibration_tradeoff_2026_06_14.md"
    assert "tile_max: mean acc_delta=+0.0140" in tiled_row["evidence_summary"]
    assert "tile_mean: Brier improves on 2/2" in tiled_row["evidence_summary"]
    recon_row = frame[frame["claim_id"] == "reconstruction_residuals_are_source_sensitive"].iloc[0]
    assert recon_row["primary_artifact"] == "reports/reconstruction_v2_probe_2026_06_14.md"
    assert "same-domain AUC 0.7600 vs reconstruction_lite 0.7300" in recon_row["evidence_summary"]
    assert "Ishu to MS COCOAI AUC 0.6000 vs reconstruction_lite 0.6400" in recon_row["evidence_summary"]
    assert "ms_to_ishu_source_cap_accuracy" in frame["evidence_finding_ids"].str.cat(sep=",")
    assert "ms_to_ishu_tuned_fusion_native_tiling_best" in frame["evidence_finding_ids"].str.cat(sep=",")
    assert "Claim Evidence Matrix" in markdown_path.read_text(encoding="utf-8")
