from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_manuscript_drafts_lint_passes_generated_drafts(tmp_path: Path) -> None:
    out_path = tmp_path / "manuscript_drafts_lint.md"
    checks_out = tmp_path / "manuscript_drafts_lint.csv"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "lint_manuscript_drafts.py"),
            "--repo-root",
            str(ROOT),
            "--manifest",
            str(ROOT / "reports" / "assets" / "manuscript_draft_manifest.csv"),
            "--wifs-draft",
            str(ROOT / "reports" / "wifs_manuscript_draft_2026_06_14.md"),
            "--dff-draft",
            str(ROOT / "reports" / "dff_manuscript_draft_2026_06_14.md"),
            "--out-path",
            str(out_path),
            "--checks-out",
            str(checks_out),
        ],
        cwd=ROOT,
        check=True,
    )

    report = out_path.read_text(encoding="utf-8")
    checks = pd.read_csv(checks_out)

    assert "Manuscript Drafts Lint" in report
    assert "Status: **PASS**" in report
    assert checks["passed"].all()
    assert set(checks["venue"]) == {"WIFS", "DFF"}
    assert "asset callouts exist" in set(checks["check"])
