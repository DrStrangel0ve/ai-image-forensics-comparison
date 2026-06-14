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
    assert order["publication tables"] < order["claim matrix"]
    assert order["robustness failure ranking"] < order["submission packet"]
    assert order["external benchmark claim lint"] < order["submission packet"]
    assert order["SOTA gap report"] < order["submission packet"]
    assert order["submission packet"] < order["submission package lint"]
    assert order["submission package lint"] < order["submission scorecard"]
