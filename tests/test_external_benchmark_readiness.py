from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_external_benchmark_readiness_marks_official_scores_absent(tmp_path: Path) -> None:
    out_path = tmp_path / "external_benchmark_readiness.md"
    status_out = tmp_path / "external_benchmark_readiness.csv"
    metrics_out = tmp_path / "external_benchmark_proxy_metrics.csv"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_external_benchmark_readiness.py"),
            "--out-path",
            str(out_path),
            "--status-out",
            str(status_out),
            "--metrics-out",
            str(metrics_out),
        ],
        cwd=ROOT,
        check=True,
    )

    text = out_path.read_text(encoding="utf-8")
    status = pd.read_csv(status_out)
    metrics = pd.read_csv(metrics_out)
    rows = status.set_index("benchmark_id")

    assert "External Benchmark Readiness" in text
    assert "not as official scored submissions" in text
    assert set(status["benchmark_id"]) == {"ntire_2026_robust_aigc", "imageclef_2026_deepfake"}
    assert set(status["official_status"]) == {"closed_not_submitted"}
    assert set(status["official_score"]) == {"none"}
    assert rows.loc["ntire_2026_robust_aigc", "proxy_status"] == "proxy_tested"
    assert rows.loc["imageclef_2026_deepfake", "proxy_status"] == "protocol_inspired_only"
    assert "ishu_to_ms_clip_standalone" in set(metrics["finding_id"])
    assert "ms_to_ishu_tuned_fusion_blur1" in set(metrics["finding_id"])
    assert "ishu_to_ms_triage5_clip_standalone" in set(metrics["finding_id"])
    assert metrics["metric_summary"].str.len().gt(0).all()
