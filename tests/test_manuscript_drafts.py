from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_manuscript_drafts_assemble_wifs_and_dff_markdown(tmp_path: Path) -> None:
    manifest_out = tmp_path / "manuscript_draft_manifest.csv"
    wifs_out = tmp_path / "wifs_manuscript_draft.md"
    dff_out = tmp_path / "dff_manuscript_draft.md"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_manuscript_drafts.py"),
            "--assembly-map",
            str(ROOT / "reports" / "assets" / "manuscript_assembly_map.csv"),
            "--section-drafts",
            str(ROOT / "reports" / "paper_section_drafts_2026_06_14.md"),
            "--submission-text",
            str(ROOT / "reports" / "submission_text_drafts_2026_06_14.md"),
            "--manifest-out",
            str(manifest_out),
            "--wifs-out",
            str(wifs_out),
            "--dff-out",
            str(dff_out),
        ],
        cwd=ROOT,
        check=True,
    )

    wifs = wifs_out.read_text(encoding="utf-8")
    dff = dff_out.read_text(encoding="utf-8")
    manifest = pd.read_csv(manifest_out)

    assert "Source-Heldout Evaluation" in wifs
    assert "SCP-Fusion: Source-Calibrated" in dff
    assert "## Abstract" in wifs
    assert "## Failure analysis and ablations" in dff
    assert "Assets to place or cite" in wifs
    assert "single-image physical-proxy" in dff
    assert set(manifest["venue"]) == {"WIFS", "DFF"}
    assert manifest["draft_word_count"].min() > 900
