from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_submission_paper_skeleton_builder_writes_wifs_and_dff_tex(tmp_path: Path) -> None:
    text_drafts = tmp_path / "submission_text_drafts.md"
    claim_matrix = tmp_path / "claim_evidence_matrix.csv"
    literature_map = tmp_path / "literature_map.csv"
    section_drafts = tmp_path / "paper_section_drafts.md"
    out_dir = tmp_path / "paper_skeletons"
    report_out = tmp_path / "paper_skeletons.md"
    text_drafts.write_text(
        "\n".join(
            [
                "# Drafts",
                "",
                "## WIFS Compact Abstract",
                "",
                "WIFS abstract with combined_v3 and SCP-Fusion wording.",
                "",
                "## DFF Workshop Abstract",
                "",
                "DFF abstract with CLIP and robustness wording.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "claim_id": "physics_guided_branch_helps",
                "claim": "Physics-guided fusion helps on the current evidence.",
                "submission_use": "WIFS/DFF ablation section.",
                "status": "ready_with_caveat",
                "evidence_finding_ids": "ishu_same_physics_guided",
                "evidence_summary": "supporting row",
                "primary_artifact": "reports/physics_guided_vs_resnet_2026_06_12.md",
                "risk_or_caveat": "single-image physical proxy, not true photometric stereo",
                "next_action": "keep comparative",
            },
            {
                "claim_id": "poster_only_claim",
                "claim": "Poster-only wording.",
                "submission_use": "DFRWS poster lead.",
                "status": "ready",
                "evidence_finding_ids": "poster_row",
                "evidence_summary": "supporting row",
                "primary_artifact": "reports/dfrws_poster_brief_2026_06_13.md",
                "risk_or_caveat": "poster only",
                "next_action": "none",
            },
        ]
    ).to_csv(claim_matrix, index=False)
    pd.DataFrame(
        {
            "key": [
                "universal_fake_detectors_2023",
                "genimage_2023",
                "aide_chameleon_2025",
                "realhd_2026",
                "bias_free_training_2025",
                "dire_2023",
                "aeroblade_2024",
                "fire_2025",
                "spectral_any_resolution_2025",
                "no_pixel_left_behind_2025",
                "fake_or_jpeg_2024",
                "photometric_faces_2023",
                "light2lie_2026",
            ]
        }
    ).to_csv(literature_map, index=False)
    section_drafts.write_text(
        "\n".join(
            [
                "# Paper Section Drafts",
                "",
                "## WIFS Introduction Draft",
                "",
                "Custom introduction with 0.8450 accuracy.",
                "",
                "## WIFS Data And Audit Draft",
                "",
                "Custom data and audit paragraph.",
                "",
                "## WIFS Methods Draft",
                "",
                "Custom methods paragraph with single-image physical proxy.",
                "",
                "## WIFS Results Draft",
                "",
                "Custom results paragraph with 0.8641 AUC.",
                "",
                "## DFF Expansion Draft",
                "",
                "Custom DFF expansion paragraph.",
                "",
                "## Limitations And Reproducibility Draft",
                "",
                "Custom limitations paragraph; SCP-Fusion does not universally beat frozen CLIP.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_submission_paper_skeletons.py"),
            "--text-drafts",
            str(text_drafts),
            "--claim-matrix",
            str(claim_matrix),
            "--literature-map",
            str(literature_map),
            "--section-drafts",
            str(section_drafts),
            "--out-dir",
            str(out_dir),
            "--report-out",
            str(report_out),
        ],
        cwd=ROOT,
        check=True,
    )

    manifest = pd.read_csv(out_dir / "submission_paper_skeleton_manifest.csv")
    wifs = (out_dir / "wifs_2026_paper_skeleton.tex").read_text(encoding="utf-8")
    dff = (out_dir / "dff_2026_workshop_skeleton.tex").read_text(encoding="utf-8")
    report = report_out.read_text(encoding="utf-8")

    assert set(manifest["paper_id"]) == {"wifs_2026", "dff_2026"}
    assert "\\documentclass[conference]{IEEEtran}" in wifs
    assert "\\documentclass[sigconf,review,anonymous]{acmart}" in dff
    assert "\\input{reports/assets/latex_tables/robustness_stress.tex}" in wifs
    assert "\\input{reports/assets/latex_tables/source_holdout_stress.tex}" in dff
    assert "reports/assets/source_holdout_generator_stress.png" in wifs
    assert "\\label{fig:source-stress}" in dff
    assert "\\ref{fig:source-stress}" in wifs
    assert "Custom introduction with 0.8450 accuracy." in wifs
    assert "Custom DFF expansion paragraph." in dff
    assert "TODO" not in wifs
    assert "\\cite{universal_fake_detectors_2023" in wifs
    assert "\\cite{photometric_faces_2023,light2lie_2026}" in dff
    assert "Claim-Evidence Checklist" in wifs
    assert "physics\\_guided\\_branch\\_helps" in wifs
    assert "poster\\_only\\_claim" not in wifs
    assert "single-image physical proxy" in dff
    assert set(manifest["claim_count"]) == {1}
    assert set(manifest["citation_count"]) == {3}
    assert set(manifest["todo_count"]) == {0}
    assert set(manifest["draft_section_count"]) == {5, 6}
    assert "Submission Paper Skeletons" in report
