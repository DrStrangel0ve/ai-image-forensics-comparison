from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_claim_evidence_matrix_validates_and_writes_artifacts(tmp_path: Path) -> None:
    core_results = tmp_path / "publication_core_results.csv"
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
        "ms_to_ishu_tuned_fusion_jpeg70",
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

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_claim_evidence_matrix.py"),
            "--core-results",
            str(core_results),
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
    assert "needs_more_evidence" in set(frame["status"])
    assert "ms_to_ishu_source_cap_accuracy" in frame["evidence_finding_ids"].str.cat(sep=",")
    assert "Claim Evidence Matrix" in markdown_path.read_text(encoding="utf-8")
