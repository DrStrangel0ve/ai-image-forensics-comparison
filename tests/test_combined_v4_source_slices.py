from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _write_predictions(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_combined_v4_source_slice_analyzer_outputs_group_deltas(tmp_path: Path) -> None:
    source = tmp_path / "source"
    transfer = tmp_path / "transfer"
    metadata = tmp_path / "metadata.csv"
    out_dir = tmp_path / "assets"
    report = tmp_path / "report.md"

    source_rows_v3 = [
        {"path": str(tmp_path / "AI-images" / "AI-images" / "ai_items" / "a.png"), "y_true": 1, "fake_score": 0.60},
        {"path": str(tmp_path / "Real-images" / "Real-images" / "real_items" / "b.png"), "y_true": 0, "fake_score": 0.40},
        {"path": str(tmp_path / "AI-images" / "AI-images" / "ai_human" / "c.png"), "y_true": 1, "fake_score": 0.40},
        {"path": str(tmp_path / "Real-images" / "Real-images" / "real_humans" / "d.png"), "y_true": 0, "fake_score": 0.60},
    ]
    source_rows_v4 = [
        {**source_rows_v3[0], "fake_score": 0.70},
        {**source_rows_v3[1], "fake_score": 0.30},
        {**source_rows_v3[2], "fake_score": 0.55},
        {**source_rows_v3[3], "fake_score": 0.45},
    ]
    transfer_rows_v3 = [
        {"path": str(tmp_path / "validation" / "ai_generated" / "fake1.jpg"), "y_true": 1, "fake_score": 0.40},
        {"path": str(tmp_path / "validation" / "ai_generated" / "fake2.jpg"), "y_true": 1, "fake_score": 0.30},
        {"path": str(tmp_path / "validation" / "real" / "real1.jpg"), "y_true": 0, "fake_score": 0.30},
    ]
    transfer_rows_v4 = [
        {**transfer_rows_v3[0], "fake_score": 0.60},
        {**transfer_rows_v3[1], "fake_score": 0.55},
        {**transfer_rows_v3[2], "fake_score": 0.45},
    ]

    for run, source_rows, transfer_rows in [
        ("combined_v3_logreg", source_rows_v3, transfer_rows_v3),
        ("combined_v4_logreg", source_rows_v4, transfer_rows_v4),
        ("combined_v4_logreg_selectk60", source_rows_v4, transfer_rows_v4),
    ]:
        _write_predictions(source / "seed7" / run / "predictions.csv", source_rows)
        _write_predictions(transfer / "seed7" / run / "predictions.csv", transfer_rows)

    pd.DataFrame(
        [
            {"split": "validation", "label": 1, "source_label": 1, "path": transfer_rows_v3[0]["path"]},
            {"split": "validation", "label": 1, "source_label": 2, "path": transfer_rows_v3[1]["path"]},
            {"split": "validation", "label": 0, "source_label": 0, "path": transfer_rows_v3[2]["path"]},
        ]
    ).to_csv(metadata, index=False)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "analyze_combined_v4_source_slices.py"),
            "--source-root",
            str(source),
            "--transfer-root",
            str(transfer),
            "--metadata",
            str(metadata),
            "--out-dir",
            str(out_dir),
            "--report-out",
            str(report),
        ],
        cwd=ROOT,
        check=True,
    )

    seed_metrics = pd.read_csv(out_dir / "combined_v4_source_slice_seed_metrics.csv")
    delta_metrics = pd.read_csv(out_dir / "combined_v4_source_slice_delta_metrics.csv")

    assert {"items", "human"}.issubset(set(seed_metrics["group_label"]))
    sd21_delta = delta_metrics[
        (delta_metrics["group_label"] == "sd21")
        & (delta_metrics["candidate"] == "combined_v4_logreg_selectk60")
    ].iloc[0]
    assert sd21_delta["fake_detection_rate_delta_mean"] == 1.0
    assert "combined_v4 Source-Slice Diagnostics" in report.read_text(encoding="utf-8")
