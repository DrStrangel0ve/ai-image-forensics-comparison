from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_references_bib_builder_writes_draft_bibtex(tmp_path: Path) -> None:
    literature_map = tmp_path / "literature_map.csv"
    out_bib = tmp_path / "references.bib"
    manifest_out = tmp_path / "references_manifest.csv"
    report_out = tmp_path / "references_report.md"
    pd.DataFrame(
        [
            {
                "key": "paper_one_2026",
                "title": "Paper One",
                "year": 2026,
                "primary_url": "https://example.com/paper-one",
                "theme": "foundation generalization",
            },
            {
                "key": "paper_two_2025",
                "title": "Paper Two",
                "year": 2025,
                "primary_url": "https://example.com/paper-two",
                "theme": "compression bias",
            },
        ]
    ).to_csv(literature_map, index=False)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_references_bib.py"),
            "--literature-map",
            str(literature_map),
            "--out-bib",
            str(out_bib),
            "--manifest-out",
            str(manifest_out),
            "--report-out",
            str(report_out),
        ],
        cwd=ROOT,
        check=True,
    )

    bib = out_bib.read_text(encoding="utf-8")
    manifest = pd.read_csv(manifest_out)
    report = report_out.read_text(encoding="utf-8")

    assert "@misc{paper_one_2026" in bib
    assert "Authors to verify" in bib
    assert "Verify bibliographic metadata before submission" in bib
    assert set(manifest["key"]) == {"paper_one_2026", "paper_two_2025"}
    assert set(manifest["status"]) == {"draft_metadata_verify_before_submission"}
    assert "Draft References BibTeX" in report
