from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_external_benchmark_claim_lint_passes_guarded_proxy_language(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    safe_file = repo_root / "safe.md"
    readiness_report = repo_root / "reports" / "external_benchmark_readiness_2026_06_14.md"
    readiness_status = repo_root / "reports" / "assets" / "external_benchmark_readiness.csv"
    out_path = tmp_path / "external_benchmark_claim_lint.md"
    checks_out = tmp_path / "external_benchmark_claim_lint.csv"

    safe_file.write_text(
        "NTIRE/ImageCLEF-style proxy robustness evidence is not an official scored submission.\n",
        encoding="utf-8",
    )
    readiness_report.parent.mkdir(parents=True)
    readiness_report.write_text(
        "\n".join(
            [
                "This report tracks benchmarks not as official scored submissions.",
                "Use proxy robustness evidence only.",
            ]
        ),
        encoding="utf-8",
    )
    readiness_status.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "benchmark_id": ["ntire_2026_robust_aigc", "imageclef_2026_deepfake"],
            "official_status": ["closed_not_submitted", "closed_not_submitted"],
            "official_score": ["none", "none"],
        }
    ).to_csv(readiness_status, index=False)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "lint_external_benchmark_claims.py"),
            "--repo-root",
            str(repo_root),
            "--scan-files",
            "safe.md",
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
    assert "Status: **PASS**" in report
    assert checks["passed"].all()
    assert "official scores remain none" in checks["check"].str.cat(sep=" ")


def test_external_benchmark_claim_lint_rejects_unguarded_official_claim(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    unsafe_file = repo_root / "unsafe.md"
    readiness_report = repo_root / "reports" / "external_benchmark_readiness_2026_06_14.md"
    readiness_status = repo_root / "reports" / "assets" / "external_benchmark_readiness.csv"

    unsafe_file.write_text("Our NTIRE leaderboard rank is excellent.\n", encoding="utf-8")
    readiness_report.parent.mkdir(parents=True)
    readiness_report.write_text(
        "not as official scored submissions\nproxy robustness evidence\n",
        encoding="utf-8",
    )
    readiness_status.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "benchmark_id": ["ntire_2026_robust_aigc", "imageclef_2026_deepfake"],
            "official_status": ["closed_not_submitted", "closed_not_submitted"],
            "official_score": ["none", "none"],
        }
    ).to_csv(readiness_status, index=False)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "lint_external_benchmark_claims.py"),
            "--repo-root",
            str(repo_root),
            "--scan-files",
            "unsafe.md",
            "--out-path",
            str(tmp_path / "lint.md"),
            "--checks-out",
            str(tmp_path / "lint.csv"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "no unsafe official NTIRE/ImageCLEF claims" in result.stderr
