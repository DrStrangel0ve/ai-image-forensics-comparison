from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from forensic_compare.freuid import freuid_competition_path


ROOT = Path(__file__).resolve().parents[1]


def test_freuid_competition_path_maps_nested_kaggle_paths() -> None:
    assert freuid_competition_path("train/abc.jpeg") == "train/train/abc.jpeg"
    assert freuid_competition_path("public_test/def.jpeg") == "public_test/public_test/def.jpeg"
    assert freuid_competition_path("ghi", split="public_test") == "public_test/public_test/ghi.jpeg"


def test_download_freuid_images_dry_run_builds_plan(tmp_path: Path) -> None:
    metadata = tmp_path / "sample.csv"
    manifest = tmp_path / "manifest.json"
    pd.DataFrame({"id": ["b", "a"]}).to_csv(metadata, index=False)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "download_freuid_images.py"),
            "--metadata-csv",
            str(metadata),
            "--manifest-out",
            str(manifest),
            "--limit",
            "2",
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
    )

    report = json.loads(manifest.read_text(encoding="utf-8"))
    planned = {row["competition_path"] for row in report["rows"]}
    assert report["n_planned"] == 2
    assert report["n_completed"] == 2
    assert report["stopped_early"] is False
    assert report["status_counts"] == {"dry_run": 2}
    assert planned == {"public_test/public_test/a.jpeg", "public_test/public_test/b.jpeg"}


def test_download_freuid_images_dry_run_balances_limit_by_columns(tmp_path: Path) -> None:
    metadata = tmp_path / "sample.csv"
    manifest = tmp_path / "manifest.json"
    pd.DataFrame(
        {
            "id": [f"{label}_{idx}" for label in [0, 1] for idx in range(4)],
            "label": [label for label in [0, 1] for _idx in range(4)],
        }
    ).to_csv(metadata, index=False)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "download_freuid_images.py"),
            "--metadata-csv",
            str(metadata),
            "--manifest-out",
            str(manifest),
            "--limit",
            "4",
            "--balance-columns",
            "label",
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
    )

    report = json.loads(manifest.read_text(encoding="utf-8"))
    planned_ids = [Path(row["competition_path"]).stem for row in report["rows"]]
    assert report["n_planned"] == 4
    assert report["n_completed"] == 4
    assert report["stopped_early"] is False
    assert report["balance_columns"] == ["label"]
    assert sum(value.startswith("0_") for value in planned_ids) == 2
    assert sum(value.startswith("1_") for value in planned_ids) == 2
