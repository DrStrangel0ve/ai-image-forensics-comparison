from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_wifs_breadth_decision_freezes_current_scope(tmp_path: Path) -> None:
    out_path = tmp_path / "wifs_breadth_decision.md"
    csv_out = tmp_path / "wifs_breadth_decision.csv"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_wifs_breadth_decision.py"),
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
    options = pd.read_csv(csv_out)
    selected = options.sort_values("decision_score", ascending=False).iloc[0]

    assert "WIFS Breadth Decision" in text
    assert "freeze the WIFS experimental scope" in text
    assert "source-aware v4 selection" in text
    assert selected["option_id"] == "freeze_current_scope"
    assert selected["decision"] == "selected"
    assert set(options["option_id"]) == {
        "freeze_current_scope",
        "source_aware_v4_selection",
        "larger_source_split",
        "true_tiled_neural_foundation",
    }
