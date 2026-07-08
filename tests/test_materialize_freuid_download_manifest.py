from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_materialize_freuid_download_manifest_writes_balanced_metadata(tmp_path: Path) -> None:
    manifest = tmp_path / "download_manifest.json"
    train_labels = tmp_path / "train_labels.csv"
    out_csv = tmp_path / "slice.csv"
    report_path = tmp_path / "slice_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "rows": [
                    {"competition_path": "train/train/a.jpeg", "status": "downloaded"},
                    {"competition_path": "train/train/b.jpeg", "status": "skipped"},
                    {"competition_path": "train/train/c.jpeg", "status": "failed"},
                ]
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "image_path": ["train/a.jpeg", "train/b.jpeg", "train/c.jpeg"],
            "label": [0, 1, 1],
            "is_digital": [True, True, True],
            "type": ["A/DL", "B/ID", "B/ID"],
        }
    ).to_csv(train_labels, index=False)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "materialize_freuid_download_manifest.py"),
            "--manifest",
            str(manifest),
            "--train-labels",
            str(train_labels),
            "--out-csv",
            str(out_csv),
            "--manifest-out",
            str(report_path),
        ],
        cwd=ROOT,
        check=True,
    )

    frame = pd.read_csv(out_csv)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert list(frame["id"]) == ["a", "b"]
    assert report["n_rows"] == 2
    assert report["label_counts"] == {"0": 1, "1": 1}
    assert report["type_counts"] == {"A/DL": 1, "B/ID": 1}
