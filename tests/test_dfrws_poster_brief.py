from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_dfrws_poster_brief_builder_writes_markdown_and_key_numbers(tmp_path: Path) -> None:
    core_results = tmp_path / "publication_core_results.csv"
    claim_matrix = tmp_path / "claim_evidence_matrix.csv"
    source_stress_summary = tmp_path / "source_stress.csv"
    out_path = tmp_path / "dfrws_poster_brief.md"
    key_numbers_out = tmp_path / "dfrws_poster_key_numbers.csv"
    finding_ids = [
        "ishu_same_combined_v3",
        "ishu_same_resnet18",
        "ishu_same_physics_guided",
        "ishu_to_ms_clip_standalone",
        "ishu_to_ms_scp_fusion_all_foundation",
        "ishu_to_ms_triage5_clip_standalone",
        "ms_to_ishu_branch_dropout_auc",
        "ms_to_ishu_tuned_fusion_constraint_sweep_best",
        "ms_to_ishu_tuned_fusion_native_tiling_best",
        "ms_to_ishu_tuned_fusion_jpeg70",
        "ms_to_ishu_tuned_fusion_jpeg50",
        "ms_to_ishu_tuned_fusion_jpeg30",
        "ms_to_ishu_tuned_fusion_blur1",
        "ms_to_ishu_tuned_fusion_resize_half",
        "ms_to_ishu_tuned_fusion_crop85",
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
            "accuracy": [0.70] * len(finding_ids),
            "auc": [0.80] * len(finding_ids),
            "brier": [0.25] * len(finding_ids),
            "ece": [0.20] * len(finding_ids),
            "predicted_fake_rate": [0.50] * len(finding_ids),
            "coverage": [pd.NA] * len(finding_ids),
            "decided_accuracy": [pd.NA] * len(finding_ids),
            "source": ["source"] * len(finding_ids),
            "interpretation": ["interpretation"] * len(finding_ids),
        }
    ).to_csv(core_results, index=False)
    pd.DataFrame(
        {
            "claim_id": ["clip_transfer_frontier", "combined_v4_is_ablation_candidate"],
            "claim": ["claim", "claim"],
            "submission_use": ["DFRWS poster lead", "appendix"],
            "status": ["ready", "needs_more_evidence"],
            "evidence_finding_ids": ["ishu_to_ms_clip_standalone", ""],
            "evidence_summary": ["summary", "summary"],
            "primary_artifact": ["artifact", "artifact"],
            "risk_or_caveat": ["caveat", "caveat"],
            "next_action": ["action", "action"],
        }
    ).to_csv(claim_matrix, index=False)
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
            str(ROOT / "scripts" / "build_dfrws_poster_brief.py"),
            "--core-results",
            str(core_results),
            "--claim-matrix",
            str(claim_matrix),
            "--source-stress-summary",
            str(source_stress_summary),
            "--out-path",
            str(out_path),
            "--key-numbers-out",
            str(key_numbers_out),
        ],
        cwd=ROOT,
        check=True,
    )

    text = out_path.read_text(encoding="utf-8")
    key_numbers = pd.read_csv(key_numbers_out)

    assert "DFRWS-USA 2026 Poster Brief" in text
    assert "Do Not Overclaim" in text
    assert "Held-Out Generator Stress" in text
    assert "sd3" in text
    assert "source_holdout_generator_stress.png" in text
    assert "seed-29 false-negative grid" in text
    assert "clip_transfer_frontier" in text
    assert "combined_v4_is_ablation_candidate" not in text
    assert "ms_to_ishu_tuned_fusion_social_720p" in set(key_numbers["finding"])
    assert "ms_to_ishu_tuned_fusion_native_tiling_best" in set(key_numbers["finding"])
