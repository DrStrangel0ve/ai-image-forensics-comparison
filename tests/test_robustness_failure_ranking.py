from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_robustness_failure_ranking_builds_transform_stress_table(tmp_path: Path) -> None:
    out_path = tmp_path / "robustness_failure_ranking.md"
    csv_out = tmp_path / "robustness_failure_ranking.csv"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_robustness_failure_ranking.py"),
            "--out-path",
            str(out_path),
            "--csv-out",
            str(csv_out),
        ],
        cwd=ROOT,
        check=True,
    )

    text = out_path.read_text(encoding="utf-8")
    ranking = pd.read_csv(csv_out)

    assert "Robustness Failure Ranking" in text
    assert "proxy robustness evidence" in text
    assert len(ranking) == 10
    assert set(["blur1", "resize_half", "jpeg30", "screenshot"]) <= set(ranking["transform"])
    assert ranking.iloc[0]["transform"] == "resize_half"
    assert ranking.sort_values("delta_accuracy_vs_clean").iloc[0]["transform"] == "jpeg30"
    assert "major_drop" in set(ranking["stress_tier"])
    assert "apparent_gain" in set(ranking["stress_tier"])
    assert ranking["delta_auc_vs_clean"].is_monotonic_increasing
