from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_submission_upload_checklist_writes_status_summary(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _write_all_referenced_files(repo_root)

    out_path = tmp_path / "submission_upload_checklist.md"
    csv_out = tmp_path / "submission_upload_checklist.csv"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_submission_upload_checklist.py"),
            "--repo-root",
            str(repo_root),
            "--out-path",
            str(out_path),
            "--csv-out",
            str(csv_out),
        ],
        cwd=ROOT,
        check=True,
    )

    text = out_path.read_text(encoding="utf-8")
    checklist = pd.read_csv(csv_out)
    summary_text = text.split("## DFRWS-USA 2026 poster/demo", maxsplit=1)[0]

    assert "Submission Upload Checklist" in text
    assert "DFRWS-USA 2026 poster/demo" in text
    assert "IEEE WIFS 2026 paper" in text
    assert "DFF-2026 ACM Multimedia workshop" in text
    assert "missing_asset" not in set(checklist["status"])
    assert "ready_asset" in set(checklist["status"])
    assert "decision_needed" not in set(checklist["status"])
    assert "writing_needed" in set(checklist["status"])
    assert "final_export_needed" in set(checklist["status"])
    assert "| DFRWS-USA 2026 poster/demo | 2026-07-07 | 5 | 0 | 0 | 1 | 0 |" in summary_text
    assert "| IEEE WIFS 2026 paper | 2026-07-15 | 4 | 0 | 1 | 1 | 0 |" in summary_text
    assert "selected qualitative grid" in text
    assert "seed-29 false-negative grid" in text
    assert "paper-critical breadth stance" in text
    assert "Freeze the WIFS experimental scope" in text
    assert "Edit the assembled WIFS markdown draft" in text
    assert "Edit the assembled DFF markdown draft" in text
    assert "wifs_manuscript_draft_2026_06_14.md" in text
    assert "dff_manuscript_draft_2026_06_14.md" in text


def _write_all_referenced_files(repo_root: Path) -> None:
    import importlib.util

    module_path = ROOT / "scripts" / "build_submission_upload_checklist.py"
    spec = importlib.util.spec_from_file_location("build_submission_upload_checklist", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    for item in module.UPLOAD_ITEMS:
        for relative in module._path_list(item["paths"]):
            path = repo_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("placeholder\n", encoding="utf-8")
