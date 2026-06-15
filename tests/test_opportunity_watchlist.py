from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_opportunity_watchlist_writes_current_targets(tmp_path: Path) -> None:
    out_path = tmp_path / "opportunity_watchlist.md"
    csv_out = tmp_path / "opportunity_watchlist.csv"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_opportunity_watchlist.py"),
            "--out-path",
            str(out_path),
            "--csv-out",
            str(csv_out),
            "--run-date",
            "2026-06-14",
        ],
        cwd=ROOT,
        check=True,
    )

    text = out_path.read_text(encoding="utf-8")
    watchlist = pd.read_csv(csv_out)
    rows = watchlist.set_index("opportunity_id")

    assert "Opportunity Watchlist" in text
    assert "DFRWS-USA 2026 poster/demo" in text
    assert "IEEE WIFS 2026 paper" in text
    assert "DFF-2026 at ACM Multimedia" in text
    assert "NTIRE 2026 Robust AI-Generated Image Detection in the Wild" in text
    assert "ImageCLEF 2026 Deepfake Detection and Generation" in text
    assert len(watchlist) == 6
    assert rows.loc["dfrws_usa_2026_poster", "days_until_deadline"] == 23
    assert rows.loc["wifs_2026_paper", "urgency"] == "active_window"
    assert rows.loc["dff_2026_workshop", "status"] == "open"
    assert rows.loc["ntire_2026_robust_aigc", "status"] == "closed_benchmark"
    assert rows.loc["imageclef_2026_deepfake", "urgency"] == "benchmark_only"
    assert watchlist["source_url"].notna().all()
    assert watchlist["source_url"].str.startswith("https://").all()
