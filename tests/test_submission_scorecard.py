from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_submission_scorecard_aggregates_manifest_lints_and_claims(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    scripts_dir = repo_root / "scripts"
    reports_dir = repo_root / "reports"
    assets_dir = reports_dir / "assets"
    scripts_dir.mkdir(parents=True)
    assets_dir.mkdir(parents=True)
    shutil.copyfile(
        ROOT / "scripts" / "build_submission_packet.py",
        scripts_dir / "build_submission_packet.py",
    )

    manifest = assets_dir / "submission_packet_manifest.csv"
    pd.DataFrame(
        [
            {
                "path": "README.md",
                "type": "repo",
                "venues": "DFRWS,WIFS,DFF",
                "purpose": "entry",
                "required": True,
                "exists": True,
                "size_bytes": 10,
            },
            {
                "path": "reports/assets/dfrws_poster_transfer_panel.png",
                "type": "figure",
                "venues": "DFRWS",
                "purpose": "poster panel",
                "required": True,
                "exists": True,
                "size_bytes": 10,
            },
            {
                "path": "reports/assets/paper_skeletons/wifs_2026_paper_skeleton.tex",
                "type": "paper-draft",
                "venues": "WIFS",
                "purpose": "wifs skeleton",
                "required": True,
                "exists": True,
                "size_bytes": 10,
            },
            {
                "path": "reports/assets/paper_skeletons/dff_2026_workshop_skeleton.tex",
                "type": "paper-draft",
                "venues": "DFF",
                "purpose": "dff skeleton",
                "required": True,
                "exists": True,
                "size_bytes": 10,
            },
            {
                "path": "reports/dfrws_poster_package_lint_2026_06_14.md",
                "type": "quality-control",
                "venues": "DFRWS",
                "purpose": "poster lint",
                "required": False,
                "exists": True,
                "size_bytes": 10,
            },
        ]
    ).to_csv(manifest, index=False)

    claim_matrix = assets_dir / "claim_evidence_matrix.csv"
    pd.DataFrame(
        {
            "claim_id": ["dfrws_ready", "wifs_caveated", "dff_ready"],
            "submission_use": [
                "DFRWS poster lead",
                "WIFS compact comparison",
                "DFF method framing",
            ],
            "status": ["ready", "ready_with_caveat", "ready"],
        }
    ).to_csv(claim_matrix, index=False)

    for relative, passed, total in [
        ("reports/dfrws_poster_package_lint_2026_06_14.md", 21, 21),
        ("reports/submission_result_tables_lint_2026_06_14.md", 31, 31),
        ("reports/claim_evidence_matrix_lint_2026_06_14.md", 12, 12),
        ("reports/paper_section_drafts_lint_2026_06_14.md", 59, 59),
        ("reports/paper_skeleton_lint_2026_06_14.md", 45, 45),
        ("reports/manuscript_drafts_lint_2026_06_14.md", 32, 32),
        ("reports/submission_package_lint_2026_06_14.md", 31, 31),
    ]:
        path = repo_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"# Lint\n\nStatus: **PASS** ({passed}/{total} checks passed).\n",
            encoding="utf-8",
        )

    out_path = tmp_path / "submission_scorecard.md"
    summary_out = tmp_path / "submission_scorecard.csv"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_submission_scorecard.py"),
            "--repo-root",
            str(repo_root),
            "--manifest",
            str(manifest),
            "--claim-matrix",
            str(claim_matrix),
            "--out-path",
            str(out_path),
            "--summary-out",
            str(summary_out),
        ],
        cwd=ROOT,
        check=True,
    )

    text = out_path.read_text(encoding="utf-8")
    scorecard = pd.read_csv(summary_out)
    dfrws = scorecard[scorecard["venue_key"] == "DFRWS"].iloc[0]
    wifs = scorecard[scorecard["venue_key"] == "WIFS"].iloc[0]

    assert "Submission Scorecard" in text
    assert "ready_to_polish" in text
    assert int(dfrws["days_until_deadline"]) == 23
    assert int(dfrws["lint_reports_passed"]) == int(dfrws["lint_reports_total"])
    assert int(wifs["caveated_claims"]) == 1
    assert set(scorecard["packet_status"]) == {"ready_to_polish"}
