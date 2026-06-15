from __future__ import annotations

import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _write_manifest(repo_root: Path, manifest: Path, status: str = "safe") -> tuple[Path, Path]:
    paper = repo_root / "reports" / "paper.md"
    figure = repo_root / "reports" / "figure.png"
    paper.parent.mkdir(parents=True, exist_ok=True)
    paper.write_text("Anonymous paper text.\n", encoding="utf-8")
    figure.write_bytes(b"\x89PNG\r\n\x1a\nfigure")
    pd.DataFrame(
        [
            {
                "path": "reports/paper.md",
                "section": "paper-drafts",
                "venues": "WIFS,DFF",
                "required": True,
                "purpose": "paper",
                "exists": True,
                "size_bytes": paper.stat().st_size,
                "sha256": hashlib.sha256(paper.read_bytes()).hexdigest(),
                "text_scanned": True,
                "anonymous_status": status,
                "blocker_issues": "" if status == "safe" else "email_address",
                "review_issues": "",
            },
            {
                "path": "reports/figure.png",
                "section": "figures",
                "venues": "WIFS,DFF",
                "required": False,
                "purpose": "figure",
                "exists": True,
                "size_bytes": figure.stat().st_size,
                "sha256": hashlib.sha256(figure.read_bytes()).hexdigest(),
                "text_scanned": False,
                "anonymous_status": "not_scanned_binary",
                "blocker_issues": "",
                "review_issues": "",
            },
        ]
    ).to_csv(manifest, index=False)
    return paper, figure


def test_anonymous_review_bundle_packager_writes_deterministic_zip(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    manifest = repo_root / "reports" / "assets" / "anonymous_review_bundle_manifest.csv"
    manifest.parent.mkdir(parents=True)
    paper, figure = _write_manifest(repo_root, manifest)
    zip_out = repo_root / "outputs" / "anonymous_review_bundle_2026_06_15.zip"
    report_out = repo_root / "reports" / "anonymous_review_bundle_package_2026_06_15.md"
    entries_out = repo_root / "reports" / "assets" / "anonymous_review_bundle_zip_manifest.csv"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "package_anonymous_review_bundle.py"),
            "--repo-root",
            str(repo_root),
            "--bundle-manifest",
            str(manifest.relative_to(repo_root)),
            "--zip-out",
            str(zip_out.relative_to(repo_root)),
            "--out-path",
            str(report_out.relative_to(repo_root)),
            "--entries-out",
            str(entries_out.relative_to(repo_root)),
            "--run-date",
            "2026-06-15",
        ],
        cwd=repo_root,
        check=True,
    )

    report = report_out.read_text(encoding="utf-8")
    entries = pd.read_csv(entries_out)
    with zipfile.ZipFile(zip_out, "r") as archive:
        names = archive.namelist()
        infos = {info.filename: info for info in archive.infolist()}

    assert "Status: **PASS** (5/5 checks passed)." in report
    assert set(names) == {
        "anonymous_review_bundle/reports/figure.png",
        "anonymous_review_bundle/reports/paper.md",
    }
    assert all(not name.startswith("/") and ".." not in Path(name).parts for name in names)
    assert all(infos[name].date_time == (2026, 6, 15, 0, 0, 0) for name in names)
    assert set(entries["include_status"]) == {"included"}
    assert set(entries["source_sha256"]) == {
        hashlib.sha256(paper.read_bytes()).hexdigest(),
        hashlib.sha256(figure.read_bytes()).hexdigest(),
    }


def test_anonymous_review_bundle_packager_blocks_identifier_findings(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    manifest = repo_root / "reports" / "assets" / "anonymous_review_bundle_manifest.csv"
    manifest.parent.mkdir(parents=True)
    _write_manifest(repo_root, manifest, status="blocker")
    zip_out = repo_root / "outputs" / "anonymous_review_bundle_2026_06_15.zip"
    report_out = repo_root / "reports" / "anonymous_review_bundle_package_2026_06_15.md"
    entries_out = repo_root / "reports" / "assets" / "anonymous_review_bundle_zip_manifest.csv"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "package_anonymous_review_bundle.py"),
            "--repo-root",
            str(repo_root),
            "--bundle-manifest",
            str(manifest.relative_to(repo_root)),
            "--zip-out",
            str(zip_out.relative_to(repo_root)),
            "--out-path",
            str(report_out.relative_to(repo_root)),
            "--entries-out",
            str(entries_out.relative_to(repo_root)),
            "--run-date",
            "2026-06-15",
        ],
        cwd=repo_root,
        check=False,
    )

    report = report_out.read_text(encoding="utf-8")
    entries = pd.read_csv(entries_out)

    assert result.returncode == 1
    assert "Status: **FAIL**" in report
    assert "blocked entrie(s)" in report
    assert not zip_out.exists()
    assert "blocked" in set(entries["include_status"])
