from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_manuscript_assembly_map_writes_wifs_and_dff_plans(tmp_path: Path) -> None:
    out_path = tmp_path / "manuscript_assembly_map.md"
    csv_out = tmp_path / "manuscript_assembly_map.csv"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_manuscript_assembly_map.py"),
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
    frame = pd.read_csv(csv_out)

    assert "Manuscript Assembly Map" in text
    assert "WIFS Assembly" in text
    assert "DFF Assembly" in text
    assert "source-aware decisions" in text
    assert set(frame["venue"]) == {"WIFS", "DFF"}
    assert frame["assets_exist"].all()
    assert frame.groupby("venue")["target_pages"].sum().to_dict()["WIFS"] <= 6.0
    assert "Failure analysis and ablations" in set(frame["section"])
