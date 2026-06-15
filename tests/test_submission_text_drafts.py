from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_submission_text_drafts_use_core_metrics_and_write_word_counts(tmp_path: Path) -> None:
    core_results = tmp_path / "publication_core_results.csv"
    claim_matrix = tmp_path / "claim_evidence_matrix.csv"
    tiled_dino_tradeoff = tmp_path / "tiled_dinov2_calibration_tradeoff.csv"
    source_stress_summary = tmp_path / "source_stress.csv"
    calibration_operating_modes = tmp_path / "calibration_operating_modes.csv"
    out_path = tmp_path / "submission_text_drafts.md"
    counts_out = tmp_path / "word_counts.csv"

    finding_ids = [
        "ishu_same_combined_v3",
        "ishu_same_resnet18",
        "ishu_same_physics_guided",
        "ishu_to_ms_clip_standalone",
        "ishu_to_ms_scp_fusion_all_foundation",
        "ishu_to_ms_triage5_clip_standalone",
        "ms_to_ishu_tuned_fusion_constraint_sweep_best",
        "ms_to_ishu_tuned_fusion_native_tiling_best",
        "ms_to_ishu_tuned_fusion_jpeg30",
        "ms_to_ishu_tuned_fusion_blur1",
        "ms_to_ishu_tuned_fusion_social_720p",
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
            "coverage": [0.25] * len(finding_ids),
            "decided_accuracy": [0.90] * len(finding_ids),
            "source": ["source"] * len(finding_ids),
            "interpretation": ["interpretation"] * len(finding_ids),
        }
    ).to_csv(core_results, index=False)
    pd.DataFrame(
        {
            "claim_id": ["clip_transfer_frontier", "scp_fusion_is_diagnostic"],
            "claim": ["claim", "claim"],
            "submission_use": ["DFRWS headline", "DFF method framing"],
            "status": ["ready", "ready_with_caveat"],
            "evidence_finding_ids": ["ishu_to_ms_clip_standalone", "ms_to_ishu_tuned_fusion_native_tiling_best"],
            "evidence_summary": ["summary", "summary"],
            "primary_artifact": ["artifact", "artifact"],
            "risk_or_caveat": ["caveat", "caveat"],
            "next_action": ["next", "next"],
        }
    ).to_csv(claim_matrix, index=False)
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
                "selection_policy": "source_holdout_mean_utility_cap_0p48",
                "heldout_source_name": "sd3",
                "source_holdout_utility_mean": 1.42,
                "source_holdout_recall_mean": 0.7961,
                "source_holdout_fake_miss_rate_mean": 0.2039,
            }
        ]
    ).to_csv(source_stress_summary, index=False)
    pd.DataFrame(
        [
            {
                "objective": "ranking_auc",
                "selected_method": "Frozen CLIP",
                "selected_mode": "default_score",
                "metric": "auc",
                "metric_value": 0.8641,
            },
            {
                "objective": "probability_brier",
                "selected_method": "SCP-Fusion + CLIP",
                "selected_mode": "default_score",
                "metric": "brier",
                "metric_value": 0.3112,
            },
            {
                "objective": "reliability_ece",
                "selected_method": "combined_v4 select-k60",
                "selected_mode": "default_score",
                "metric": "ece",
                "metric_value": 0.2663,
            },
            {
                "objective": "source_holdout_ece",
                "selected_method": "source_calibrated",
                "selected_mode": "temperature_balanced",
                "metric": "mean_calibrated_ece",
                "metric_value": 0.1268,
            },
            {
                "objective": "tiled_dino_accuracy",
                "selected_method": "tiled DINOv2 reverse fusion",
                "selected_mode": "tile_max",
                "metric": "best_accuracy_delta",
                "metric_value": 0.0139,
            },
            {
                "objective": "tiled_dino_brier",
                "selected_method": "tiled DINOv2 reverse fusion",
                "selected_mode": "tile_mean",
                "metric": "best_brier_delta",
                "metric_value": -0.0058,
            },
        ]
    ).to_csv(calibration_operating_modes, index=False)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_submission_text_drafts.py"),
            "--core-results",
            str(core_results),
            "--claim-matrix",
            str(claim_matrix),
            "--tiled-dino-tradeoff",
            str(tiled_dino_tradeoff),
            "--source-stress-summary",
            str(source_stress_summary),
            "--calibration-operating-modes",
            str(calibration_operating_modes),
            "--out-path",
            str(out_path),
            "--word-counts-out",
            str(counts_out),
        ],
        cwd=ROOT,
        check=True,
    )

    text = out_path.read_text(encoding="utf-8")
    counts = pd.read_csv(counts_out)

    assert "Submission Text Drafts" in text
    assert "DFRWS Poster Abstract" in text
    assert "WIFS Six-Page Skeleton" in text
    assert "ms_to_ishu_tuned_fusion_native_tiling_best" in text
    assert "single-image proxy" in text
    assert "Tiled-DINO Mode Tradeoff" in text
    assert "tile_max" in text
    assert "tile_mean" in text
    assert "Held-Out Generator Stress" in text
    assert "sd3" in text
    assert "Calibration Operating Mode Guidance" in text
    assert "Frozen CLIP leads transfer AUC at 0.8641" in text
    assert "SCP-Fusion + CLIP gives the best transfer Brier score at 0.3112" in text
    assert "combined_v4 select-k60 gives the lowest transfer ECE at 0.2663" in text
    assert "source_calibrated with `temperature_balanced` anchors source-heldout ECE at 0.1268" in text
    assert "0.7961" in text
    assert "0.7000" in text
    assert set(counts["draft"]) == {
        "DFRWS poster abstract",
        "WIFS compact abstract",
        "DFF workshop abstract",
    }
    assert counts["word_count"].between(80, 260).all()
