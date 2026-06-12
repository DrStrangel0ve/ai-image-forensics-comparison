from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from forensic_compare.metrics import (
    brier_score,
    calibration_bins,
    expected_calibration_error,
    maximum_calibration_error,
)


ROOT = Path(__file__).resolve().parents[1]


def _predictions(path: Path, scores: list[float], labels: list[int]) -> None:
    pd.DataFrame(
        {
            "path": [str(path.parent / f"image-{index}.jpg") for index in range(len(scores))],
            "y_true": labels,
            "fake_score": scores,
        }
    ).to_csv(path, index=False)


def test_calibration_metrics_match_manual_two_bin_example() -> None:
    y_true = [0, 1, 1, 0]
    scores = [0.1, 0.9, 0.7, 0.2]

    assert brier_score(y_true, scores) == pytest.approx(0.0375)
    assert expected_calibration_error(y_true, scores, n_bins=2) == pytest.approx(0.175)
    assert maximum_calibration_error(y_true, scores, n_bins=2) == pytest.approx(0.2)

    bins = calibration_bins(y_true, scores, n_bins=2)
    assert bins[0]["count"] == 2
    assert bins[0]["confidence"] == pytest.approx(0.15)
    assert bins[0]["accuracy"] == 0.0
    assert bins[1]["count"] == 2
    assert bins[1]["confidence"] == pytest.approx(0.8)
    assert bins[1]["accuracy"] == 1.0


def test_calibration_summary_script_writes_metric_and_bin_tables(tmp_path: Path) -> None:
    method_a = tmp_path / "method_a.csv"
    method_b = tmp_path / "method_b.csv"
    _predictions(method_a, [0.1, 0.9, 0.7, 0.2], [0, 1, 1, 0])
    _predictions(method_b, [0.4, 0.6, 0.55, 0.45], [0, 1, 1, 0])
    out_dir = tmp_path / "calibration"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "summarize_calibration_metrics.py"),
            "--out-dir",
            str(out_dir),
            "--n-bins",
            "2",
            "--predictions",
            f"seed7:a={method_a}",
            "--predictions",
            f"seed7:b={method_b}",
        ],
        cwd=ROOT,
        check=True,
    )

    metrics = pd.read_csv(out_dir / "calibration_metrics.csv")
    summary = pd.read_csv(out_dir / "calibration_summary_by_method.csv")
    bins = pd.read_csv(out_dir / "calibration_bins.csv")
    reliability = pd.read_csv(out_dir / "calibration_reliability_by_method.csv")

    assert set(metrics["method"]) == {"a", "b"}
    method_a = metrics.loc[metrics["method"] == "a"].iloc[0]
    assert method_a["expected_calibration_error"] == pytest.approx(0.175)
    assert set(summary["method"]) == {"a", "b"}
    assert len(bins) == 4
    assert set(reliability["method"]) == {"a", "b"}
    assert (out_dir / "report.md").exists()
    assert (out_dir / "reliability_by_method.png").exists()
