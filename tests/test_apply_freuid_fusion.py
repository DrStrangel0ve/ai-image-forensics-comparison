from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _write_branch(path: Path, ids: list[str], scores: list[float]) -> None:
    pd.DataFrame({"id": ids, "fraud_score": scores}).to_csv(path, index=False)


def test_apply_freuid_fusion_writes_thresholded_unlabeled_predictions(tmp_path: Path) -> None:
    branch_a = tmp_path / "a.csv"
    branch_b = tmp_path / "b.csv"
    fusion_summary = tmp_path / "fusion_summary.json"
    threshold_manifest = tmp_path / "threshold.json"
    predictions = tmp_path / "fused.csv"
    manifest_path = tmp_path / "fused_manifest.json"
    _write_branch(branch_a, ["x", "y", "z"], [0.10, 0.60, 0.90])
    _write_branch(branch_b, ["z", "x", "y"], [0.80, 0.20, 0.40])
    fusion_summary.write_text(
        json.dumps(
            {
                "source_names": ["branch_a", "branch_b"],
                "best": {"normalization": "rank", "weights": [0.25, 0.75], "threshold_at_1pct_bpcer": 0.7},
            }
        ),
        encoding="utf-8",
    )
    threshold_manifest.write_text(json.dumps({"threshold": 0.7}), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "apply_freuid_fusion.py"),
            "--predictions",
            str(branch_a),
            str(branch_b),
            "--names",
            "branch_a",
            "branch_b",
            "--fusion-summary",
            str(fusion_summary),
            "--threshold-json",
            str(threshold_manifest),
            "--out-predictions",
            str(predictions),
            "--manifest-out",
            str(manifest_path),
        ],
        cwd=ROOT,
        check=True,
    )

    frame = pd.read_csv(predictions)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert list(frame.columns) == ["id", "fraud_score", "label"]
    assert list(frame["id"]) == ["x", "y", "z"]
    assert frame["fraud_score"].round(4).tolist() == [0.3333, 0.6667, 1.0]
    assert frame["label"].tolist() == [0, 0, 1]
    assert manifest["normalization"] == "rank"
    assert manifest["weights"] == [0.25, 0.75]
    assert manifest["threshold"] == 0.7
    assert manifest["label_counts"] == {"0": 2, "1": 1}


def test_apply_freuid_fusion_rejects_source_name_mismatch(tmp_path: Path) -> None:
    branch = tmp_path / "branch.csv"
    fusion_summary = tmp_path / "fusion_summary.json"
    predictions = tmp_path / "fused.csv"
    _write_branch(branch, ["x"], [0.1])
    fusion_summary.write_text(
        json.dumps({"source_names": ["expected"], "best": {"normalization": "raw", "weights": [1.0], "threshold_at_1pct_bpcer": 0.5}}),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "apply_freuid_fusion.py"),
            "--predictions",
            str(branch),
            "--names",
            "wrong",
            "--fusion-summary",
            str(fusion_summary),
            "--out-predictions",
            str(predictions),
        ],
        cwd=ROOT,
        check=False,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 1
    assert "Prediction names must match" in result.stderr
