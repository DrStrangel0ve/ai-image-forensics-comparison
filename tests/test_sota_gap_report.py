from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_sota_gap_report_separates_proxy_from_official_sota(tmp_path: Path) -> None:
    out_path = tmp_path / "sota_gap_report.md"
    csv_out = tmp_path / "sota_gap_report.csv"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_sota_gap_report.py"),
            "--out-path",
            str(out_path),
            "--csv-out",
            str(csv_out),
        ],
        cwd=ROOT,
        check=True,
    )

    text = out_path.read_text(encoding="utf-8")
    gaps = pd.read_csv(csv_out)

    assert "SOTA Gap Report" in text
    assert "not SOTA on official public benchmarks" in text
    assert "proxy result" in text
    assert "MICV" in text
    assert "0.9723" in text
    assert "ImageCLEF" in text
    assert set(gaps["comparison_validity"]) == {"proxy_not_official"}
    assert "ms_to_ishu_tuned_fusion_noise3" in set(gaps["finding_id"])
    assert gaps["auc_gap_to_official_sota"].lt(0).all()
    assert gaps["official_sota_auc"].eq(0.9723).all()
