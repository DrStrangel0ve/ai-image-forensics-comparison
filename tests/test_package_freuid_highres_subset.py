from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "package_freuid_highres_subset.py"
SPEC = importlib.util.spec_from_file_location("package_freuid_highres_subset", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
PACKAGER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PACKAGER)


def test_select_and_package_private_subset(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    train_root = image_root / "train"
    train_root.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    for index, (doc_type, label) in enumerate(
        [
            ("A", 0),
            ("A", 1),
            ("A", 0),
            ("A", 1),
            ("EGYPT/DL", 0),
            ("EGYPT/DL", 1),
            ("EGYPT/DL", 0),
            ("EGYPT/DL", 1),
        ]
    ):
        image_id = f"id-{index}"
        relative = f"train/{image_id}.jpeg"
        (image_root / relative).write_bytes(f"image-{index}".encode("ascii"))
        rows.append(
            {
                "id": image_id,
                "image_path": relative,
                "label": label,
                "is_digital": True,
                "type": doc_type,
            }
        )
    labels_path = tmp_path / "train_labels.csv"
    frame = pd.DataFrame(rows)
    frame.to_csv(labels_path, index=False)

    subset = PACKAGER.select_subset(labels_path, "EGYPT/DL", 2, 2, 43)
    manifest = PACKAGER.write_package(
        subset,
        frame,
        image_root,
        tmp_path / "package",
        "owner/private-subset",
        "Private Subset",
        "EGYPT/DL",
        43,
    )

    assert len(subset) == 4
    assert set(subset.groupby("type")["label"].nunique()) == {2}
    assert manifest["samples"] == 4
    archive = tmp_path / "package" / PACKAGER.ARCHIVE_NAME
    with ZipFile(archive) as handle:
        names = set(handle.namelist())
        assert "train_labels.csv" in names
        assert "subset_manifest.json" in names
        assert len([name for name in names if name.startswith("train/")]) == 4
    metadata = json.loads((tmp_path / "package" / "dataset-metadata.json").read_text(encoding="utf-8"))
    assert metadata["id"] == "owner/private-subset"
    assert metadata["isPrivate"] is True
