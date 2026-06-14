from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_paper_section_drafts_builder_uses_metrics_and_caveats(tmp_path: Path) -> None:
    core_results = tmp_path / "publication_core_results.csv"
    claim_matrix = tmp_path / "claim_evidence_matrix.csv"
    literature_map = tmp_path / "literature_map.csv"
    tiled_dino_tradeoff = tmp_path / "tiled_dinov2_calibration_tradeoff.csv"
    reconstruction_ablation = tmp_path / "submission_table_reconstruction_ablation.csv"
    source_stress_summary = tmp_path / "source_stress.csv"
    out_path = tmp_path / "paper_section_drafts.md"
    manifest_out = tmp_path / "paper_section_draft_manifest.csv"

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
            "accuracy": [0.8, 0.8, 0.85, 0.64, 0.62, pd.NA, 0.76, 0.77, 0.71, 0.70, 0.75],
            "auc": [0.89, 0.88, 0.92, 0.86, 0.80, pd.NA, 0.84, 0.85, 0.82, 0.78, 0.85],
            "brier": [pd.NA] * len(finding_ids),
            "ece": [pd.NA] * len(finding_ids),
            "predicted_fake_rate": [pd.NA, pd.NA, pd.NA, 0.16, 0.13, pd.NA, 0.52, 0.55, 0.35, 0.56, 0.47],
            "coverage": [pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, 0.47, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA],
            "decided_accuracy": [pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, 0.93, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA],
            "source": ["source"] * len(finding_ids),
            "interpretation": ["interpretation"] * len(finding_ids),
        }
    ).to_csv(core_results, index=False)
    pd.DataFrame(
        {
            "claim_id": ["claim_a", "claim_b"],
            "claim": ["claim", "claim"],
            "submission_use": ["WIFS", "DFF"],
            "status": ["ready", "ready_with_caveat"],
            "evidence_finding_ids": ["a", "b"],
            "evidence_summary": ["summary", "summary"],
            "primary_artifact": ["artifact", "artifact"],
            "risk_or_caveat": ["caveat", "caveat"],
            "next_action": ["next", "next"],
        }
    ).to_csv(claim_matrix, index=False)
    pd.DataFrame(
        {
            "key": [
                "universal_fake_detectors_2023",
                "genimage_2023",
                "aide_chameleon_2025",
                "spectral_any_resolution_2025",
                "fake_or_jpeg_2024",
                "no_pixel_left_behind_2025",
                "photometric_faces_2023",
                "light2lie_2026",
            ],
            "theme": [
                "foundation generalization",
                "cross-generator benchmark",
                "multi-expert detection",
                "spectral learning",
                "compression bias",
                "high-resolution tiling",
                "physics-informed analysis",
                "reflectance physics",
            ],
        }
    ).to_csv(literature_map, index=False)
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
                "method": "reconstruction_v2",
                "delta_auc_vs_reconstruction_lite": 0.0215,
            },
            {
                "setting": "ishu_to_ms_cocoai_bounded",
                "method": "reconstruction_v2",
                "delta_auc_vs_reconstruction_lite": -0.0408,
            },
        ]
    ).to_csv(reconstruction_ablation, index=False)
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

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_paper_section_drafts.py"),
            "--core-results",
            str(core_results),
            "--claim-matrix",
            str(claim_matrix),
            "--literature-map",
            str(literature_map),
            "--tiled-dino-tradeoff",
            str(tiled_dino_tradeoff),
            "--reconstruction-ablation",
            str(reconstruction_ablation),
            "--source-stress-summary",
            str(source_stress_summary),
            "--out-path",
            str(out_path),
            "--manifest-out",
            str(manifest_out),
        ],
        cwd=ROOT,
        check=True,
    )

    report = out_path.read_text(encoding="utf-8")
    manifest = pd.read_csv(manifest_out)

    assert "Paper Section Drafts" in report
    assert "WIFS Introduction Draft" in report
    assert "0.8500 accuracy" in report
    assert "single-image physical/signal proxy" in report
    assert "does not universally beat frozen CLIP" in report
    assert "tile_max" in report
    assert "tile_mean" in report
    assert "reconstruction_v2" in report
    assert "+0.0215 same-domain" in report
    assert "-0.0408 under Ishu-to-MS transfer" in report
    assert "sd3" in report
    assert "0.2039 fake-miss rate" in report
    assert len(manifest) >= 6
    assert manifest["has_metric"].any()
    assert manifest["has_caveat"].any()
