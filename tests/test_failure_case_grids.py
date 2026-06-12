from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def test_failure_case_grid_export_writes_csvs_and_images(tmp_path: Path) -> None:
    image_paths = []
    colors = ["#dddddd", "#bbbbbb", "#88ccff", "#66aaff", "#ffcc88", "#ffaa66"]
    for index, color in enumerate(colors):
        path = tmp_path / f"image_{index}.jpg"
        Image.new("RGB", (48, 40), color).save(path)
        image_paths.append(path)

    labels = [0, 0, 1, 1, 1, 0]
    source_labels = [0, 0, 1, 2, 1, 0]
    metadata = pd.DataFrame(
        {
            "path": [str(path) for path in image_paths],
            "split": ["validation"] * len(image_paths),
            "label": labels,
            "source_label": source_labels,
            "caption": [f"caption {index}" for index in range(len(image_paths))],
        }
    )
    method_a = pd.DataFrame(
        {
            "path": [str(path) for path in image_paths],
            "y_true": labels,
            "fake_score": [0.9, 0.2, 0.1, 0.8, 0.3, 0.7],
        }
    )
    method_b = pd.DataFrame(
        {
            "path": [str(path) for path in image_paths],
            "y_true": labels,
            "fake_score": [0.1, 0.8, 0.9, 0.2, 0.7, 0.1],
        }
    )
    metadata_path = tmp_path / "metadata.csv"
    method_a_path = tmp_path / "method_a.csv"
    method_b_path = tmp_path / "method_b.csv"
    out_dir = tmp_path / "failure_cases"
    metadata.to_csv(metadata_path, index=False)
    method_a.to_csv(method_a_path, index=False)
    method_b.to_csv(method_b_path, index=False)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "export_failure_case_grids.py"),
            "--metadata",
            str(metadata_path),
            "--predictions",
            f"a={method_a_path}",
            "--predictions",
            f"b={method_b_path}",
            "--primary-method",
            "a",
            "--out-dir",
            str(out_dir),
            "--top-k",
            "3",
            "--tile-size",
            "64",
            "--columns",
            "2",
            "--min-disagreement",
            "0.4",
        ],
        cwd=ROOT,
        check=True,
    )

    assert (out_dir / "joined_predictions_wide.csv").exists()
    assert len(pd.read_csv(out_dir / "false_positives.csv")) == 2
    assert len(pd.read_csv(out_dir / "false_negatives.csv")) == 2
    assert len(pd.read_csv(out_dir / "disagreements.csv")) == 3
    assert (out_dir / "false_positives_grid.png").exists()
    assert (out_dir / "false_negatives_grid.png").exists()
    assert (out_dir / "disagreements_grid.png").exists()
    assert (out_dir / "report.md").exists()
