from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_prepare_freuid_split_is_type_label_stratified(tmp_path: Path) -> None:
    labels = tmp_path / "train_labels.csv"
    train_out = tmp_path / "train.csv"
    val_out = tmp_path / "val.csv"
    manifest_out = tmp_path / "manifest.json"
    rows = []
    for doc_type in ["A/DL", "B/ID"]:
        for label in [0, 1]:
            for idx in range(10):
                image_id = f"{doc_type.replace('/', '_')}_{label}_{idx:02d}"
                rows.append(
                    {
                        "id": image_id,
                        "image_path": f"train/{image_id}.jpeg",
                        "label": label,
                        "is_digital": True,
                        "type": doc_type,
                    }
                )
    pd.DataFrame(rows).to_csv(labels, index=False)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "prepare_freuid_split.py"),
            "--train-labels",
            str(labels),
            "--train-out",
            str(train_out),
            "--val-out",
            str(val_out),
            "--manifest-out",
            str(manifest_out),
            "--val-fraction",
            "0.2",
            "--seed",
            "11",
        ],
        cwd=ROOT,
        check=True,
    )

    train = pd.read_csv(train_out)
    val = pd.read_csv(val_out)
    manifest = json.loads(manifest_out.read_text(encoding="utf-8"))

    assert len(train) == 32
    assert len(val) == 8
    assert set(train["id"]).isdisjoint(set(val["id"]))
    assert train.groupby(["type", "label"]).size().to_dict() == {
        ("A/DL", 0): 8,
        ("A/DL", 1): 8,
        ("B/ID", 0): 8,
        ("B/ID", 1): 8,
    }
    assert val.groupby(["type", "label"]).size().to_dict() == {
        ("A/DL", 0): 2,
        ("A/DL", 1): 2,
        ("B/ID", 0): 2,
        ("B/ID", 1): 2,
    }
    assert manifest["n_total"] == 40
    assert manifest["n_train"] == 32
    assert manifest["n_val"] == 8
    assert manifest["strata_columns"] == ["type", "label"]
