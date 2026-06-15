from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_submission_privacy_audit_classifies_paths_and_secrets(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    readme = repo_root / "README.md"
    detail = repo_root / "reports" / "assets" / "detail.csv"
    config = repo_root / "reports" / "secret_note.md"
    audit_output = repo_root / "reports" / "submission_privacy_audit_2026_06_15.md"
    audit_csv = repo_root / "reports" / "assets" / "submission_privacy_audit.csv"
    readme.write_text("Use C:\\Users\\<you>\\.kaggle\\kaggle.json as a placeholder.\n", encoding="utf-8")
    detail.parent.mkdir(parents=True, exist_ok=True)
    detail.write_text(
        "path,score\nC:\\Users\\arnav\\Documents\\project\\data\\image.png,0.9\n",
        encoding="utf-8",
    )
    config.write_text("api_key = abcdefghijklmnop\n", encoding="utf-8")
    audit_output.write_text("old warning: C:\\Users\\arnav\\Documents\\project\\old.csv\n", encoding="utf-8")
    audit_csv.write_text("path,example\nself,C:\\Users\\arnav\\Documents\\project\\old.csv\n", encoding="utf-8")
    manifest = tmp_path / "submission_packet_manifest.csv"
    pd.DataFrame(
        {
            "path": [
                "README.md",
                "reports/assets/detail.csv",
                "reports/secret_note.md",
                "reports/submission_privacy_audit_2026_06_15.md",
                "reports/assets/submission_privacy_audit.csv",
            ],
            "type": ["repo", "result-data", "planning", "quality-control", "quality-control"],
            "venues": ["DFRWS,WIFS,DFF"] * 5,
            "purpose": ["purpose"] * 5,
            "required": [True, False, False, False, False],
            "exists": [True, True, True, True, True],
            "size_bytes": [
                readme.stat().st_size,
                detail.stat().st_size,
                config.stat().st_size,
                audit_output.stat().st_size,
                audit_csv.stat().st_size,
            ],
        }
    ).to_csv(manifest, index=False)
    out_path = audit_output
    findings_out = audit_csv

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_submission_privacy_audit.py"),
            "--repo-root",
            str(repo_root),
            "--manifest",
            str(manifest),
            "--out-path",
            str(out_path),
            "--findings-out",
            str(findings_out),
            "--run-date",
            "2026-06-15",
        ],
        cwd=ROOT,
        check=True,
    )

    report = out_path.read_text(encoding="utf-8")
    findings = pd.read_csv(findings_out)

    assert "Status: **REVIEW**" in report
    assert set(findings["issue"]) == {
        "placeholder_local_path",
        "local_absolute_path",
        "secret_like_assignment",
    }
    assert findings.loc[findings["issue"] == "placeholder_local_path", "severity"].item() == "info"
    assert findings.loc[findings["issue"] == "local_absolute_path", "severity"].item() == "warning"
    assert findings.loc[findings["issue"] == "secret_like_assignment", "severity"].item() == "blocker"
    assert "reports/submission_privacy_audit_2026_06_15.md" not in set(findings["path"])
    assert "reports/assets/submission_privacy_audit.csv" not in set(findings["path"])
