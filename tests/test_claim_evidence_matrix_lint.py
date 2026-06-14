from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_claim_evidence_matrix_lint_passes_resolved_and_artifact_backed_claims(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    assets_dir = repo_root / "reports" / "assets"
    assets_dir.mkdir(parents=True)
    primary_report = repo_root / "reports" / "calibration_operating_modes_2026_06_14.md"
    primary_report.parent.mkdir(parents=True, exist_ok=True)
    primary_report.write_text("placeholder\n", encoding="utf-8")
    core_report = assets_dir / "publication_core_results.md"
    core_report.write_text("placeholder\n", encoding="utf-8")

    core_results = assets_dir / "publication_core_results.csv"
    pd.DataFrame(
        {
            "finding_id": ["finding_a"],
            "method": ["method"],
            "accuracy": [0.8],
        }
    ).to_csv(core_results, index=False)

    claim_matrix = assets_dir / "claim_evidence_matrix.csv"
    pd.DataFrame(
        [
            {
                "claim_id": "resolved_claim",
                "claim": "Source shift separates ranking and decisions.",
                "submission_use": "WIFS",
                "status": "ready",
                "evidence_finding_ids": "finding_a",
                "evidence_summary": "finding_a (method: acc=0.8000)",
                "primary_artifact": "reports/assets/publication_core_results.md",
                "risk_or_caveat": "Keep this diagnostic.",
                "next_action": "Use as context.",
            },
            {
                "claim_id": "artifact_claim",
                "claim": "Operating modes are objective-specific across calibration and ranking.",
                "submission_use": "DFF",
                "status": "ready_with_caveat",
                "evidence_finding_ids": "",
                "evidence_summary": "calibration_operating_modes (ranking_auc: method / mode (auc=0.9000))",
                "primary_artifact": "reports/calibration_operating_modes_2026_06_14.md",
                "risk_or_caveat": "Do not turn this into a universal-detector claim.",
                "next_action": "Keep the caveat visible.",
            },
        ]
    ).to_csv(claim_matrix, index=False)

    out_path = tmp_path / "claim_evidence_matrix_lint.md"
    checks_out = tmp_path / "claim_evidence_matrix_lint.csv"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "lint_claim_evidence_matrix.py"),
            "--repo-root",
            str(repo_root),
            "--claim-matrix",
            str(claim_matrix),
            "--core-results",
            str(core_results),
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

    assert "Claim Evidence Matrix Lint" in report
    assert "Status: **PASS**" in report
    assert checks["passed"].all()
    assert "primary artifacts exist" in set(checks["check"])
    assert "evidence IDs resolve or artifact-backed summaries are named" in set(checks["check"])
