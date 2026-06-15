from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _write_hash_fixture(repo_root: Path, hashes: Path, bad_hash: bool = False) -> None:
    artifact = repo_root / "artifact.txt"
    skipped = repo_root / "reports" / "submission_artifact_hashes_2026_06_15.md"
    artifact.write_text("stable bytes\n", encoding="utf-8")
    skipped.parent.mkdir(parents=True, exist_ok=True)
    skipped.write_text("self output\n", encoding="utf-8")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    pd.DataFrame(
        {
            "path": [
                "artifact.txt",
                "reports/submission_artifact_hashes_2026_06_15.md",
                "missing.txt",
            ],
            "type": ["repo", "reproducibility", "planning"],
            "venues": ["DFRWS,WIFS,DFF"] * 3,
            "required": [True, False, False],
            "exists": [True, True, False],
            "size_bytes": [artifact.stat().st_size, skipped.stat().st_size, pd.NA],
            "sha256": ["0" * 64 if bad_hash else digest, pd.NA, pd.NA],
            "hash_status": ["hashed", "index_or_self_skipped", "missing"],
        }
    ).to_csv(hashes, index=False)


def test_submission_artifact_hash_lint_passes_clean_hashes(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    hashes = repo_root / "reports" / "assets" / "submission_artifact_hashes.csv"
    hashes.parent.mkdir(parents=True, exist_ok=True)
    _write_hash_fixture(repo_root, hashes)
    out_path = repo_root / "reports" / "submission_artifact_hashes_lint_2026_06_15.md"
    checks_out = repo_root / "reports" / "assets" / "submission_artifact_hashes_lint.csv"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "lint_submission_artifact_hashes.py"),
            "--repo-root",
            str(repo_root),
            "--out-path",
            str(out_path.relative_to(repo_root)),
            "--checks-out",
            str(checks_out.relative_to(repo_root)),
            "--run-date",
            "2026-06-15",
        ],
        cwd=repo_root,
        check=True,
    )

    report = out_path.read_text(encoding="utf-8")
    checks = pd.read_csv(checks_out)
    assert "Status: **PASS**" in report
    assert checks["passed"].all()
    assert "sha256 values match" in set(checks["check"])


def test_submission_artifact_hash_lint_fails_tampered_hash(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    hashes = repo_root / "reports" / "assets" / "submission_artifact_hashes.csv"
    hashes.parent.mkdir(parents=True, exist_ok=True)
    _write_hash_fixture(repo_root, hashes, bad_hash=True)
    out_path = repo_root / "reports" / "submission_artifact_hashes_lint_2026_06_15.md"
    checks_out = repo_root / "reports" / "assets" / "submission_artifact_hashes_lint.csv"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "lint_submission_artifact_hashes.py"),
            "--repo-root",
            str(repo_root),
            "--out-path",
            str(out_path.relative_to(repo_root)),
            "--checks-out",
            str(checks_out.relative_to(repo_root)),
            "--run-date",
            "2026-06-15",
        ],
        cwd=repo_root,
        check=False,
    )

    checks = pd.read_csv(checks_out)
    sha_check = checks[checks["check"] == "sha256 values match"].iloc[0]
    assert result.returncode == 1
    assert bool(sha_check["passed"]) is False
