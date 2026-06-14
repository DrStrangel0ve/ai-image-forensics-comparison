from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _write_predictions(path: Path, rows: list[tuple[str, int, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=["path", "y_true", "fake_score"])
    frame.to_csv(path, index=False)


def _seed_rows(prefix: str) -> list[tuple[str, int, float]]:
    return [
        (f"{prefix}/real_0.jpg", 0, 0.20),
        (f"{prefix}/real_1.jpg", 0, 0.30),
        (f"{prefix}/fake_0.jpg", 1, 0.70),
        (f"{prefix}/fake_1.jpg", 1, 0.80),
    ]


def test_reconstruction_lite_fusion_probe_summarizes_paired_predictions(
    tmp_path: Path,
) -> None:
    source_reconstruction = tmp_path / "source_reconstruction"
    source_combined = tmp_path / "source_combined"
    target_reconstruction = tmp_path / "target_reconstruction"
    target_combined = tmp_path / "target_combined"
    seed_out = tmp_path / "seed.csv"
    mean_out = tmp_path / "mean.csv"
    delta_out = tmp_path / "delta.csv"
    coefficient_out = tmp_path / "coefficients.csv"
    report_out = tmp_path / "report.md"

    for seed in [7, 17]:
        _write_predictions(
            source_reconstruction / f"seed{seed}" / "predictions.csv",
            _seed_rows(f"source/{seed}"),
        )
        _write_predictions(
            source_combined / f"seed{seed}" / "predictions.csv",
            _seed_rows(f"source/{seed}"),
        )
        _write_predictions(
            target_reconstruction / f"seed{seed}" / "predictions.csv",
            [
                (f"target/{seed}/real_0.jpg", 0, 0.20),
                (f"target/{seed}/real_1.jpg", 0, 0.35),
                (f"target/{seed}/fake_0.jpg", 1, 0.65),
                (f"target/{seed}/fake_1.jpg", 1, 0.75),
            ],
        )
        _write_predictions(
            target_combined / f"seed{seed}" / "predictions.csv",
            [
                (f"target/{seed}/real_0.jpg", 0, 0.40),
                (f"target/{seed}/real_1.jpg", 0, 0.45),
                (f"target/{seed}/fake_0.jpg", 1, 0.55),
                (f"target/{seed}/fake_1.jpg", 1, 0.60),
            ],
        )

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "summarize_reconstruction_lite_fusion_probe.py"),
            "--source-reconstruction-root",
            str(source_reconstruction),
            "--source-combined-root",
            str(source_combined),
            "--target-reconstruction-root",
            str(target_reconstruction),
            "--target-combined-root",
            str(target_combined),
            "--seed-out",
            str(seed_out),
            "--mean-out",
            str(mean_out),
            "--delta-out",
            str(delta_out),
            "--coefficient-out",
            str(coefficient_out),
            "--report-out",
            str(report_out),
        ],
        cwd=ROOT,
        check=True,
    )

    seed_summary = pd.read_csv(seed_out)
    mean_summary = pd.read_csv(mean_out)
    coefficients = pd.read_csv(coefficient_out)
    report = report_out.read_text(encoding="utf-8")

    assert set(seed_summary["candidate"]) == {
        "combined_v3_logreg",
        "reconstruction_lite_logreg",
        "mean_fusion",
        "source_logreg_fusion",
    }
    assert len(seed_summary) == 8
    assert set(mean_summary["candidate"]) == set(seed_summary["candidate"])
    assert len(coefficients) == 2
    assert "reconstruction_lite Fusion Probe" in report
    assert "not an official benchmark result" in report
