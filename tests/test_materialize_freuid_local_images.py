from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_materialize_freuid_local_images_filters_to_present_train_files(tmp_path: Path) -> None:
    labels = tmp_path / "train_labels.csv"
    pd.DataFrame(
        [
            {"id": "aaa", "image_path": "train/aaa.jpeg", "label": 0, "is_digital": 0, "type": "A"},
            {"id": "bbb", "image_path": "train/bbb.jpeg", "label": 1, "is_digital": 1, "type": "B"},
        ]
    ).to_csv(labels, index=False)
    sample = tmp_path / "sample_submission.csv"
    pd.DataFrame([{"id": "ccc", "label": 0}]).to_csv(sample, index=False)
    image_root = tmp_path / "images"
    (image_root / "train" / "train").mkdir(parents=True)
    (image_root / "train" / "train" / "bbb.jpeg").write_bytes(b"fake image bytes")

    out_csv = tmp_path / "available.csv"
    manifest = tmp_path / "available.manifest.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "materialize_freuid_local_images.py"),
            "--train-labels",
            str(labels),
            "--sample-submission",
            str(sample),
            "--image-root",
            str(image_root),
            "--split",
            "train",
            "--out-csv",
            str(out_csv),
            "--manifest-out",
            str(manifest),
        ],
        check=True,
    )

    materialized = pd.read_csv(out_csv)
    assert materialized["id"].tolist() == ["bbb"]
    assert materialized["competition_path"].tolist() == ["train/train/bbb.jpeg"]

    report = json.loads(manifest.read_text(encoding="utf-8"))
    assert report["n_reference_rows"] == 2
    assert report["n_local_images"] == 1
    assert report["n_matched_rows"] == 1
    assert report["label_counts"] == {"1": 1}


def test_materialize_freuid_local_images_can_index_public_test_files(tmp_path: Path) -> None:
    labels = tmp_path / "train_labels.csv"
    pd.DataFrame([{"id": "aaa", "image_path": "train/aaa.jpeg", "label": 0, "type": "A"}]).to_csv(labels, index=False)
    sample = tmp_path / "sample_submission.csv"
    pd.DataFrame([{"id": "ccc", "label": 0}, {"id": "ddd", "label": 0}]).to_csv(sample, index=False)
    image_root = tmp_path / "images"
    (image_root / "public_test" / "public_test").mkdir(parents=True)
    (image_root / "public_test" / "public_test" / "ccc.jpeg").write_bytes(b"fake image bytes")

    out_csv = tmp_path / "available_public.csv"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "materialize_freuid_local_images.py"),
            "--train-labels",
            str(labels),
            "--sample-submission",
            str(sample),
            "--image-root",
            str(image_root),
            "--split",
            "public_test",
            "--out-csv",
            str(out_csv),
        ],
        check=True,
    )

    materialized = pd.read_csv(out_csv)
    assert materialized["id"].tolist() == ["ccc"]
    assert materialized["image_path"].tolist() == ["public_test/ccc.jpeg"]
    assert materialized["competition_path"].tolist() == ["public_test/public_test/ccc.jpeg"]
