from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_reconstruction_v2_feature_report_documents_extended_ablation(tmp_path: Path) -> None:
    out_path = tmp_path / "reconstruction_v2.md"
    csv_out = tmp_path / "reconstruction_v2.csv"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_reconstruction_v2_feature_report.py"),
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

    assert "reconstruction_v2 Feature Set" in text
    assert "--feature-set reconstruction_v2" in text
    assert "not a claim of AEROBLADE/FIRE parity" in text
    assert {"fft_lowpass_residual", "svd_lowrank_residual"} <= set(manifest["family"])
    assert "recon_fft20_abs_mean" in set(manifest["feature_name"])
    assert "recon_svd16_laplacian_abs_mean" in set(manifest["feature_name"])
