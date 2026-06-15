from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_submission_artifact_hash_builder_writes_hash_manifest(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    readme = repo_root / "README.md"
    citation = repo_root / "CITATION.cff"
    readme.write_text("hello packet\n", encoding="utf-8")
    citation.write_text("cite me\n", encoding="utf-8")
    out_path = repo_root / "reports" / "submission_artifact_hashes_2026_06_15.md"
    hashes_out = repo_root / "reports" / "assets" / "submission_artifact_hashes.csv"
    manifest = tmp_path / "submission_packet_manifest.csv"
    pd.DataFrame(
        {
            "path": [
                "README.md",
                "CITATION.cff",
                "reports/missing.md",
                "reports/submission_artifact_hashes_2026_06_15.md",
                "reports/assets/submission_artifact_hashes.csv",
            ],
            "type": ["repo", "repo", "planning", "reproducibility", "reproducibility"],
            "venues": ["DFRWS,WIFS,DFF"] * 5,
            "purpose": ["purpose"] * 5,
            "required": [True, True, False, False, False],
            "exists": [True, True, False, False, False],
            "size_bytes": [13, 8, pd.NA, pd.NA, pd.NA],
        }
    ).to_csv(manifest, index=False)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_submission_artifact_hashes.py"),
            "--repo-root",
            str(repo_root),
            "--manifest",
            str(manifest),
            "--out-path",
            str(out_path.relative_to(repo_root)),
            "--hashes-out",
            str(hashes_out.relative_to(repo_root)),
            "--run-date",
            "2026-06-15",
        ],
        cwd=repo_root,
        check=True,
    )

    report = out_path.read_text(encoding="utf-8")
    hashes = pd.read_csv(hashes_out)
    readme_row = hashes[hashes["path"] == "README.md"].iloc[0]
    missing_row = hashes[hashes["path"] == "reports/missing.md"].iloc[0]
    self_rows = hashes[hashes["hash_status"] == "index_or_self_skipped"]

    assert "Submission Artifact Hashes" in report
    assert "Run date: 2026-06-15" in report
    assert readme_row["sha256"] == hashlib.sha256(readme.read_bytes()).hexdigest()
    assert readme_row["hash_status"] == "hashed"
    assert missing_row["hash_status"] == "missing"
    assert pd.isna(missing_row["sha256"])
    assert set(self_rows["path"]) == {
        "reports/submission_artifact_hashes_2026_06_15.md",
        "reports/assets/submission_artifact_hashes.csv",
    }
