from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_submission_text_drafts_use_core_metrics_and_write_word_counts(tmp_path: Path) -> None:
    core_results = tmp_path / "publication_core_results.csv"
    claim_matrix = tmp_path / "claim_evidence_matrix.csv"
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

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_submission_text_drafts.py"),
            "--core-results",
            str(core_results),
            "--claim-matrix",
            str(claim_matrix),
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
    assert "0.7000" in text
    assert set(counts["draft"]) == {
        "DFRWS poster abstract",
        "WIFS compact abstract",
        "DFF workshop abstract",
    }
    assert counts["word_count"].between(80, 260).all()
