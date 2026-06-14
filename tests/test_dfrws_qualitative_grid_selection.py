from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_dfrws_qualitative_grid_selection_recommends_seed_29(tmp_path: Path) -> None:
    out_path = tmp_path / "dfrws_qualitative_grid_selection.md"
    csv_out = tmp_path / "dfrws_qualitative_grid_selection.csv"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_dfrws_qualitative_grid_selection.py"),
            "--repo-root",
            str(ROOT),
            "--out-path",
            str(out_path),
            "--csv-out",
            str(csv_out),
        ],
        cwd=ROOT,
        check=True,
    )

    text = out_path.read_text(encoding="utf-8")
    audit = pd.read_csv(csv_out)
    selected = audit.sort_values("poster_selection_score", ascending=False).iloc[0]

    assert "DFRWS Qualitative Grid Selection" in text
    assert "seed-29" in text
    assert selected["seed"] == 29
    assert selected["recommendation"] == "selected_for_dfrws_poster"
    assert set(audit["seed"]) == {17, 29}
    assert audit["sources"].min() >= 4
