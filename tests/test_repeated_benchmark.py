from pathlib import Path

import pandas as pd
import pytest

from scripts.run_repeated_benchmark import _clean_forwarded_args
from scripts.summarize_repeated_benchmarks import summarize


def test_clean_forwarded_args_strips_separator() -> None:
    assert _clean_forwarded_args(["--", "--methods", "combined_v3", "neural"]) == [
        "--methods",
        "combined_v3",
        "neural",
    ]


def test_clean_forwarded_args_rejects_reserved_values() -> None:
    with pytest.raises(SystemExit, match="--seed"):
        _clean_forwarded_args(["--", "--methods", "neural", "--seed", "17"])

    with pytest.raises(SystemExit, match="--out-dir"):
        _clean_forwarded_args(["--methods", "neural", "--out-dir=runs/example"])


def _comparison(path: Path, accuracy_a: float, accuracy_b: float) -> None:
    pd.DataFrame(
        [
            {
                "method": "a",
                "accuracy": accuracy_a,
                "precision": 0.8,
                "recall": 0.7,
                "f1": 0.74,
                "roc_auc": 0.9,
            },
            {
                "method": "b",
                "accuracy": accuracy_b,
                "precision": 0.7,
                "recall": 0.6,
                "f1": 0.64,
                "roc_auc": 0.8,
            },
        ]
    ).to_csv(path, index=False)


def test_repeated_benchmark_summary_writes_bootstrap_columns(tmp_path: Path) -> None:
    run_7 = tmp_path / "seed7.csv"
    run_17 = tmp_path / "seed17.csv"
    out_dir = tmp_path / "summary"
    _comparison(run_7, accuracy_a=0.75, accuracy_b=0.65)
    _comparison(run_17, accuracy_a=0.85, accuracy_b=0.70)

    summarize([("seed7", run_7), ("seed17", run_17)], out_dir)

    summary = pd.read_csv(out_dir / "repeated_summary.csv")
    assert set(summary["method"]) == {"a", "b"}
    method_a = summary[summary["method"] == "a"].iloc[0]
    assert method_a["accuracy_mean"] == pytest.approx(0.8)
    assert "accuracy_ci_low" in summary.columns
    assert "accuracy_ci_high" in summary.columns
    assert "roc_auc_ci_low" in summary.columns
    assert "roc_auc_ci_high" in summary.columns
    assert (out_dir / "report.md").exists()
