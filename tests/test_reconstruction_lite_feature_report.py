from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_reconstruction_lite_feature_report_documents_standalone_ablation(tmp_path: Path) -> None:
    out_path = tmp_path / "reconstruction_lite.md"
    csv_out = tmp_path / "reconstruction_lite.csv"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_reconstruction_lite_feature_report.py"),
            "--out-path",
            str(out_path),
            "--csv-out",
            str(csv_out),
        ],
        cwd=ROOT,
        check=True,
    )

    text = out_path.read_text(encoding="utf-8")
    manifest = pd.read_csv(csv_out)

    assert "reconstruction_lite Feature Set" in text
    assert "standalone AEROBLADE-lite style ablation" in text
    assert "--feature-set reconstruction_lite" in text
    assert "recon_half_quarter_laplacian_delta" in set(manifest["feature_name"])
    assert set(manifest["family"]) >= {"half_scale_resize_residual", "cross_scale_delta"}
    assert manifest["order"].is_monotonic_increasing
