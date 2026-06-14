from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from source_stress_utils import load_source_stress_summary, source_stress_sentence  # noqa: E402


def test_source_stress_summary_picks_weakest_policy_source(tmp_path: Path) -> None:
    path = tmp_path / "source_summary.csv"
    pd.DataFrame(
        [
            {
                "selection_policy": "source_holdout_mean_utility_cap_0p48",
                "heldout_source_name": "sdxl",
                "source_holdout_utility_mean": 1.9,
                "source_holdout_recall_mean": 0.98,
                "source_holdout_fake_miss_rate_mean": 0.02,
            },
            {
                "selection_policy": "source_holdout_mean_utility_cap_0p48",
                "heldout_source_name": "sd3",
                "source_holdout_utility_mean": 1.42,
                "source_holdout_recall_mean": 0.7961,
                "source_holdout_fake_miss_rate_mean": 0.2039,
            },
        ]
    ).to_csv(path, index=False)

    summary = load_source_stress_summary(path)

    assert summary is not None
    assert summary["heldout_source_name"] == "sd3"
    assert summary["recall"] == "0.7961"
    assert "fake-miss rate 0.2039" in source_stress_sentence(summary)


def test_source_stress_summary_validates_columns(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    pd.DataFrame({"selection_policy": ["source_holdout_mean_utility_cap_0p48"]}).to_csv(
        path, index=False
    )

    with pytest.raises(ValueError, match="missing source-stress columns"):
        load_source_stress_summary(path)
