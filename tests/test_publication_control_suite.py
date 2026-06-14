from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_publication_control_suite_dry_run_lists_ordered_commands(tmp_path: Path) -> None:
    out_path = tmp_path / "publication_control_suite.md"
    csv_out = tmp_path / "publication_control_suite.csv"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_publication_control_suite.py"),
            "--repo-root",
            str(ROOT),
            "--dry-run",
            "--out-path",
            str(out_path),
            "--csv-out",
            str(csv_out),
        ],
        cwd=ROOT,
        check=True,
    )

    text = out_path.read_text(encoding="utf-8")
    frame = pd.read_csv(csv_out)
    order = dict(zip(frame["asset"], frame["order"], strict=True))

    assert "Publication Control Suite" in text
    assert "Status: **DRY-RUN**" in text
    assert set(frame["status"]) == {"planned"}
    assert order["publication tables"] < order["robustness failure ranking"]
    assert order["robustness failure ranking"] < order["tiled DINO calibration tradeoff"]
    assert order["tiled DINO transform stress comparison"] < order["tiled DINO calibration tradeoff"]
    assert order["tiled DINO calibration tradeoff"] < order["claim matrix"]
    assert order["claim matrix"] < order["submission result tables"]
    assert order["claim matrix"] < order["method family comparison"]
    assert order["method family comparison"] < order["submission packet"]
    assert order["claim matrix"] < order["submission text drafts"]
    assert order["claim matrix"] < order["DFRWS qualitative grid selection"]
    assert order["DFRWS qualitative grid selection"] < order["DFRWS poster brief"]
    assert order["DFRWS poster brief"] < order["DFRWS poster package lint"]
    assert order["DFRWS poster package lint"] < order["WIFS breadth decision"]
    assert order["WIFS breadth decision"] < order["publication control suite dry run"]
    assert order["robustness failure ranking"] < order["submission packet"]
    assert order["external benchmark claim lint"] < order["submission packet"]
    assert order["SOTA gap report"] < order["submission packet"]
    assert order["SOTA gap closure plan"] < order["submission packet"]
    assert order["reconstruction-lite feature report"] < order["reconstruction-v2 feature report"]
    assert order["reconstruction-v2 feature report"] < order["reconstruction-v2 bounded probe"]
    assert order["reconstruction-v2 bounded probe"] < order["reconstruction-lite bounded probe"]
    assert order["reconstruction-lite bounded probe"] < order["reconstruction-lite transfer probe"]
    assert order["reconstruction-lite transfer probe"] < order["reconstruction-lite fusion probe"]
    assert order["reconstruction-lite fusion probe"] < order["submission result tables"]
    assert order["DFRWS qualitative grid selection"] < order["submission packet"]
    assert order["WIFS breadth decision"] < order["submission packet"]
    assert order["submission packet"] < order["submission package lint"]
    assert order["submission package lint"] < order["submission scorecard"]
