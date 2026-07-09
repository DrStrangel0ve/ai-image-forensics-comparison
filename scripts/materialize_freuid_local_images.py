from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forensic_compare.freuid import freuid_competition_path
from forensic_compare.utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a FREUID metadata CSV from images already present on disk.")
    parser.add_argument("--train-labels", default="data/raw/freuid_2026/small_files/train_labels.csv")
    parser.add_argument("--sample-submission", default="data/raw/freuid_2026/small_files/sample_submission.csv")
    parser.add_argument("--image-root", default="data/raw/freuid_2026/images")
    parser.add_argument("--split", choices=["train", "public_test"], default="train")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--manifest-out", default=None, help="Optional JSON report; defaults to <out-csv>.manifest.json.")
    return parser.parse_args()


def _local_images(image_root: Path, split: str) -> pd.DataFrame:
    split_root = image_root / split
    if not split_root.exists():
        return pd.DataFrame(columns=["id", "local_path", "size_bytes"])
    rows = []
    for path in sorted(candidate for candidate in split_root.rglob("*") if candidate.is_file()):
        rows.append({"id": path.stem, "local_path": str(path), "size_bytes": int(path.stat().st_size)})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["id", "local_path", "size_bytes"])
    duplicates = frame[frame["id"].duplicated()]["id"].unique().tolist()
    if duplicates:
        raise ValueError(f"Local FREUID image ids are duplicated under {split_root}: {duplicates[:5]}")
    return frame


def _load_reference(split: str, train_labels_path: Path, sample_submission_path: Path) -> pd.DataFrame:
    if split == "train":
        frame = pd.read_csv(train_labels_path)
        required = {"id", "image_path", "label", "type"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"train labels missing required columns: {missing}")
        frame = frame.copy()
        frame["id"] = frame["id"].astype(str)
        return frame

    frame = pd.read_csv(sample_submission_path)
    if "id" not in frame.columns:
        raise ValueError("sample submission must contain an id column")
    frame = frame[["id"]].copy()
    frame["id"] = frame["id"].astype(str)
    frame["image_path"] = frame["id"].map(lambda value: f"public_test/{value}.jpeg")
    return frame


def materialize_local_images(
    train_labels_path: Path,
    sample_submission_path: Path,
    image_root: Path,
    split: str,
    out_csv: Path,
    manifest_out: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    reference = _load_reference(split, train_labels_path, sample_submission_path)
    local = _local_images(image_root, split)
    frame = reference.merge(local, on="id", how="inner", validate="one_to_one")
    if frame.empty:
        raise ValueError(f"No local FREUID {split} images matched the reference metadata under {image_root}")
    frame["competition_path"] = frame["image_path"].map(lambda value: freuid_competition_path(value, split=split))
    frame = frame.sort_values("id", kind="mergesort").reset_index(drop=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_csv, index=False)

    report: dict[str, object] = {
        "split": split,
        "image_root": str(image_root),
        "out_csv": str(out_csv),
        "n_reference_rows": int(len(reference)),
        "n_local_images": int(len(local)),
        "n_matched_rows": int(len(frame)),
        "n_unmatched_local_images": int(len(set(local["id"]) - set(reference["id"]))),
        "n_missing_reference_images": int(len(set(reference["id"]) - set(local["id"]))),
        "size_bytes": {
            "min": int(frame["size_bytes"].min()),
            "median": float(frame["size_bytes"].median()),
            "max": int(frame["size_bytes"].max()),
        },
    }
    if "label" in frame.columns:
        report["label_counts"] = {str(key): int(value) for key, value in frame["label"].value_counts().sort_index().items()}
    if "type" in frame.columns:
        report["type_counts"] = {str(key): int(value) for key, value in frame["type"].value_counts().sort_index().items()}
    write_json(report, manifest_out)
    return frame, report


def main() -> None:
    args = parse_args()
    out_csv = Path(args.out_csv)
    manifest_out = Path(args.manifest_out) if args.manifest_out else out_csv.with_suffix(".manifest.json")
    _frame, report = materialize_local_images(
        train_labels_path=Path(args.train_labels),
        sample_submission_path=Path(args.sample_submission),
        image_root=Path(args.image_root),
        split=args.split,
        out_csv=out_csv,
        manifest_out=manifest_out,
    )
    print(out_csv.resolve())
    print(manifest_out.resolve())
    print(f"n_matched_rows={report['n_matched_rows']}")


if __name__ == "__main__":
    main()
