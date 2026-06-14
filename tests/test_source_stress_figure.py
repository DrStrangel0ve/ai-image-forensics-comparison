from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def test_source_stress_figure_builder_writes_chart_and_report(tmp_path: Path) -> None:
    source_summary = tmp_path / "source_summary.csv"
    out_dir = tmp_path / "assets"
    report_out = tmp_path / "source_stress.md"
    pd.DataFrame(
        [
            {
                "selection_policy": "source_holdout_mean_utility_cap_0p48",
                "heldout_source_name": "sdxl",
                "source_holdout_utility_mean": 1.89,
                "source_holdout_recall_mean": 0.98,
                "source_holdout_fake_miss_rate_mean": 0.02,
                "source_holdout_predicted_positive_rate_mean": 0.21,
            },
            {
                "selection_policy": "source_holdout_mean_utility_cap_0p48",
                "heldout_source_name": "sd3",
                "source_holdout_utility_mean": 1.42,
                "source_holdout_recall_mean": 0.7961,
                "source_holdout_fake_miss_rate_mean": 0.2039,
                "source_holdout_predicted_positive_rate_mean": 0.13,
            },
        ]
    ).to_csv(source_summary, index=False)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_source_stress_figure.py"),
            "--source-summary",
            str(source_summary),
            "--out-dir",
            str(out_dir),
            "--report-out",
            str(report_out),
            "--dpi",
            "90",
        ],
        cwd=ROOT,
        check=True,
    )

    output = pd.read_csv(out_dir / "source_holdout_generator_stress.csv")
    report = report_out.read_text(encoding="utf-8")

    assert output.iloc[0]["heldout_source_name"] == "sd3"
    assert "weakest held-out generator is `sd3`" in report
    assert "0.7961" in report
    assert (out_dir / "source_holdout_generator_stress.svg").exists()
    with Image.open(out_dir / "source_holdout_generator_stress.png") as image:
        width, height = image.size
    assert width >= 800
    assert height >= 350
