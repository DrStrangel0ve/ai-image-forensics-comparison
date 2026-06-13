from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_literature_map_builder_writes_paper_facing_rows(tmp_path: Path) -> None:
    out_csv = tmp_path / "literature_map.csv"
    out_md = tmp_path / "literature_map.md"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_literature_map.py"),
            "--out-csv",
            str(out_csv),
            "--out-md",
            str(out_md),
        ],
        cwd=ROOT,
        check=True,
    )

    frame = pd.read_csv(out_csv)
    report = out_md.read_text(encoding="utf-8")

    assert len(frame) >= 10
    assert frame["key"].is_unique
    assert {"key", "title", "primary_url", "theme", "use_in_paper", "caveat"}.issubset(frame.columns)
    assert "clip" in " ".join(frame["use_in_paper"].str.lower())
    assert "single-image proxy" in " ".join(frame["caveat"].str.lower())
    assert "Literature Map" in report
    assert "Paper-Facing Map" in report
