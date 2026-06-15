from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_anonymous_review_bundle_indexes_safe_artifacts(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    draft = repo_root / "reports" / "paper.md"
    figure = repo_root / "reports" / "figure.png"
    draft.parent.mkdir(parents=True)
    draft.write_text("Anonymous supplementary artifact with metrics and caveats.\n", encoding="utf-8")
    figure.write_bytes(b"\x89PNG\r\n\x1a\nsafe-figure")
    out_path = repo_root / "reports" / "anonymous_review_bundle_2026_06_15.md"
    manifest_out = repo_root / "reports" / "assets" / "anonymous_review_bundle_manifest.csv"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_anonymous_review_bundle.py"),
            "--repo-root",
            str(repo_root),
            "--artifacts",
            str(draft.relative_to(repo_root)),
            str(figure.relative_to(repo_root)),
            "--out-path",
            str(out_path.relative_to(repo_root)),
            "--manifest-out",
            str(manifest_out.relative_to(repo_root)),
            "--run-date",
            "2026-06-15",
        ],
        cwd=repo_root,
        check=True,
    )

    report = out_path.read_text(encoding="utf-8")
    manifest = pd.read_csv(manifest_out)
    draft_row = manifest[manifest["path"] == "reports/paper.md"].iloc[0]
    figure_row = manifest[manifest["path"] == "reports/figure.png"].iloc[0]

    assert "Status: **PASS**" in report
    assert set(manifest["anonymous_status"]) == {"safe", "not_scanned_binary"}
    assert draft_row["sha256"] == hashlib.sha256(draft.read_bytes()).hexdigest()
    assert bool(draft_row["text_scanned"]) is True
    assert bool(figure_row["text_scanned"]) is False


def test_anonymous_review_bundle_fails_identifier_findings(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    draft = repo_root / "reports" / "paper.md"
    draft.parent.mkdir(parents=True)
    draft.write_text(
        "Contact jane@example.com and see https://github.com/example/project. The public repo mirrors this.\n",
        encoding="utf-8",
    )
    out_path = repo_root / "reports" / "anonymous_review_bundle_2026_06_15.md"
    manifest_out = repo_root / "reports" / "assets" / "anonymous_review_bundle_manifest.csv"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_anonymous_review_bundle.py"),
            "--repo-root",
            str(repo_root),
            "--artifacts",
            str(draft.relative_to(repo_root)),
            "--out-path",
            str(out_path.relative_to(repo_root)),
            "--manifest-out",
            str(manifest_out.relative_to(repo_root)),
            "--run-date",
            "2026-06-15",
        ],
        cwd=repo_root,
        check=False,
    )

    report = out_path.read_text(encoding="utf-8")
    manifest = pd.read_csv(manifest_out)
    row = manifest.iloc[0]

    assert result.returncode == 1
    assert "Status: **FAIL**" in report
    assert row["anonymous_status"] == "blocker"
    assert {"email_address", "github_url"} <= set(str(row["blocker_issues"]).split(";"))
    assert "public_release_wording" in str(row["review_issues"])
