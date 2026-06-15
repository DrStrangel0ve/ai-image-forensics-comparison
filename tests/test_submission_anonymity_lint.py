from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_submission_anonymity_lint_passes_anonymous_with_review_items(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    paper = repo_root / "paper.tex"
    paper.write_text(
        "\\author{Anonymous Author(s)}\n"
        "\\begin{document}\n"
        "The artifact package is mirrored in the public repo after review.\n"
        "\\end{document}\n",
        encoding="utf-8",
    )
    out_path = tmp_path / "anonymity.md"
    findings_out = tmp_path / "anonymity.csv"
    checks_out = tmp_path / "anonymity_checks.csv"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "lint_submission_anonymity.py"),
            "--repo-root",
            str(repo_root),
            "--artifacts",
            str(paper.relative_to(repo_root)),
            "--out-path",
            str(out_path),
            "--findings-out",
            str(findings_out),
            "--checks-out",
            str(checks_out),
            "--run-date",
            "2026-06-15",
        ],
        cwd=ROOT,
        check=True,
    )

    report = out_path.read_text(encoding="utf-8")
    findings = pd.read_csv(findings_out)
    checks = pd.read_csv(checks_out)

    assert "Status: **PASS** (4/4 checks passed)." in report
    assert findings.loc[0, "issue"] == "public_repo_wording"
    assert findings.loc[0, "severity"] == "review"
    assert checks["passed"].all()


def test_submission_anonymity_lint_fails_direct_identifiers(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    paper = repo_root / "paper.tex"
    paper.write_text(
        "\\author{Jane Doe}\n"
        "Contact jane@example.com and see https://github.com/DrStrangel0ve/ai-image-forensics-comparison.\n",
        encoding="utf-8",
    )
    out_path = tmp_path / "anonymity.md"
    findings_out = tmp_path / "anonymity.csv"
    checks_out = tmp_path / "anonymity_checks.csv"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "lint_submission_anonymity.py"),
            "--repo-root",
            str(repo_root),
            "--artifacts",
            str(paper.relative_to(repo_root)),
            "--out-path",
            str(out_path),
            "--findings-out",
            str(findings_out),
            "--checks-out",
            str(checks_out),
            "--run-date",
            "2026-06-15",
        ],
        cwd=ROOT,
        check=False,
    )

    findings = pd.read_csv(findings_out)
    assert result.returncode == 1
    assert {"non_anonymous_author_field", "email_address", "github_url", "personal_identifier"} <= set(
        findings["issue"]
    )
    assert "blocker" in set(findings["severity"])
