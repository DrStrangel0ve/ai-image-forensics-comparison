from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_competition_submission_dry_run_builds_assets(tmp_path: Path) -> None:
    source_detail = tmp_path / "detail.csv"
    out_dir = tmp_path / "competition_dry_run"
    report_path = tmp_path / "competition_submission_dry_run.md"
    pd.DataFrame(
        {
            "seed": [7, 7, 7, 17],
            "path": [
                str(tmp_path / "AI-images" / "000000.png"),
                str(tmp_path / "AI-images" / "000001.png"),
                str(tmp_path / "real" / "000000.png"),
                str(tmp_path / "real" / "000002.png"),
            ],
            "class_name": ["AI-images", "AI-images", "real", "real"],
            "y_true": [1, 1, 0, 0],
            "tile_max_score": [0.91, 0.72, 0.10, 0.22],
        }
    ).to_csv(source_detail, index=False)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_competition_submission_dry_run.py"),
            "--source-detail",
            str(source_detail),
            "--out-dir",
            str(out_dir),
            "--report-path",
            str(report_path),
            "--run-date",
            "2026-06-15",
        ],
        cwd=ROOT,
        check=True,
    )

    predictions = pd.read_csv(out_dir / "predictions.csv")
    submission = pd.read_csv(out_dir / "submission.csv")
    package_manifest = json.loads((out_dir / "submission_manifest.json").read_text(encoding="utf-8"))
    lint_manifest = json.loads((out_dir / "submission_lint.json").read_text(encoding="utf-8"))
    report = report_path.read_text(encoding="utf-8")

    assert list(predictions["image_id"]) == ["ai_images_000000", "ai_images_000001", "real_000000"]
    assert "y_true" in predictions.columns
    assert "y_true" not in submission.columns
    assert set(submission["image_id"]) == set(predictions["image_id"])
    assert package_manifest["y_true_present_excluded"] is True
    assert lint_manifest["status"] == "pass"
    assert "Competition Submission Dry Run" in report
    assert "lint manifest" in report
