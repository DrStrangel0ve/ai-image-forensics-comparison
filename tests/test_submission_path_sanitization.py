from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_submission_path_sanitization_replaces_repo_prefix(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    detail = repo_root / "reports" / "assets" / "detail.csv"
    detail.parent.mkdir(parents=True, exist_ok=True)
    local_image = repo_root / "data" / "raw" / "example.png"
    detail.write_text(f"path,score\n{local_image},0.9\n", encoding="utf-8")
    manifest = tmp_path / "submission_packet_manifest.csv"
    pd.DataFrame(
        {
            "path": ["reports/assets/detail.csv"],
            "type": ["table"],
            "venues": ["DFRWS,WIFS,DFF"],
            "purpose": ["purpose"],
            "required": [False],
            "exists": [True],
            "size_bytes": [detail.stat().st_size],
        }
    ).to_csv(manifest, index=False)
    out_path = repo_root / "reports" / "submission_path_sanitization_2026_06_15.md"
    csv_out = repo_root / "reports" / "assets" / "submission_path_sanitization.csv"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "sanitize_submission_local_paths.py"),
            "--repo-root",
            str(repo_root),
            "--manifest",
            str(manifest),
            "--out-path",
            str(out_path.relative_to(repo_root)),
            "--csv-out",
            str(csv_out.relative_to(repo_root)),
            "--run-date",
            "2026-06-15",
            "--apply",
        ],
        cwd=ROOT,
        check=True,
    )

    sanitized = detail.read_text(encoding="utf-8")
    report = out_path.read_text(encoding="utf-8")
    changes = pd.read_csv(csv_out)

    assert str(repo_root) not in sanitized
    assert "<repo>" in sanitized
    assert "data" in sanitized
    assert "Status: **APPLIED**" in report
    assert changes.loc[0, "path"] == "reports/assets/detail.csv"
    assert int(changes.loc[0, "replacements"]) == 1
    assert int(changes.loc[0, "sanitized_placeholders"]) == 1
    assert bool(changes.loc[0, "applied"]) is True
