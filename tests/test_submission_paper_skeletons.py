from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_submission_paper_skeleton_builder_writes_wifs_and_dff_tex(tmp_path: Path) -> None:
    text_drafts = tmp_path / "submission_text_drafts.md"
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

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_submission_paper_skeletons.py"),
            "--text-drafts",
            str(text_drafts),
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
    assert "single-image proxy" in dff
    assert "Submission Paper Skeletons" in report
