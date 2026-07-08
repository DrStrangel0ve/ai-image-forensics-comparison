from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forensic_compare.utils import write_json


DEFAULT_KEEP_STATUSES = ["downloaded", "skipped"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize a FREUID download manifest into a labeled metadata CSV.")
    parser.add_argument("--manifest", required=True, help="Manifest JSON from download_freuid_images.py.")
    parser.add_argument("--train-labels", required=True, help="Kaggle train_labels.csv.")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--manifest-out", default=None, help="Optional JSON report; defaults to <out-csv>.manifest.json.")
    parser.add_argument(
        "--keep-statuses",
        nargs="+",
        default=DEFAULT_KEEP_STATUSES,
        help="Download statuses to include in the output CSV.",
    )
    return parser.parse_args()


def _ids_from_manifest(manifest_path: Path, keep_statuses: set[str]) -> list[str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ids: list[str] = []
    for row in manifest.get("rows", []):
        if row.get("status") not in keep_statuses:
            continue
        competition_path = str(row.get("competition_path", ""))
        if not competition_path:
            continue
        ids.append(Path(competition_path).stem)
    if not ids:
        raise ValueError(f"No rows with statuses {sorted(keep_statuses)} found in {manifest_path}")
    if len(ids) != len(set(ids)):
        duplicates = sorted(value for value, count in Counter(ids).items() if count > 1)
        raise ValueError(f"Manifest contains duplicate image ids: {duplicates[:5]}")
    return ids


def materialize_manifest(
    manifest_path: Path,
    train_labels_path: Path,
    out_csv: Path,
    report_path: Path,
    keep_statuses: list[str] | tuple[str, ...] = tuple(DEFAULT_KEEP_STATUSES),
) -> tuple[pd.DataFrame, dict[str, object]]:
    keep_status_set = set(keep_statuses)
    ids = _ids_from_manifest(manifest_path, keep_status_set)
    labels = pd.read_csv(train_labels_path)
    required = {"id", "image_path", "label", "type"}
    missing_columns = sorted(required - set(labels.columns))
    if missing_columns:
        raise ValueError(f"train labels missing required columns: {missing_columns}")
    labels = labels.copy()
    labels["id"] = labels["id"].astype(str)
    if labels["id"].duplicated().any():
        raise ValueError("train labels contain duplicate ids")
    labels_by_id = labels.set_index("id", drop=False)
    missing_ids = [image_id for image_id in ids if image_id not in labels_by_id.index]
    if missing_ids:
        raise ValueError(f"Manifest ids missing from train labels: {missing_ids[:5]}")

    frame = labels_by_id.loc[ids].reset_index(drop=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_csv, index=False)
    report = {
        "manifest": str(manifest_path),
        "train_labels": str(train_labels_path),
        "out_csv": str(out_csv),
        "keep_statuses": sorted(keep_status_set),
        "n_rows": int(len(frame)),
        "label_counts": {str(key): int(value) for key, value in frame["label"].value_counts().sort_index().items()},
        "type_counts": {str(key): int(value) for key, value in frame["type"].value_counts().sort_index().items()},
    }
    write_json(report, report_path)
    return frame, report


def main() -> None:
    args = parse_args()
    out_csv = Path(args.out_csv)
    report_path = Path(args.manifest_out) if args.manifest_out else out_csv.with_suffix(".manifest.json")
    materialize_manifest(
        manifest_path=Path(args.manifest),
        train_labels_path=Path(args.train_labels),
        out_csv=out_csv,
        report_path=report_path,
        keep_statuses=list(args.keep_statuses),
    )
    print(out_csv.resolve())
    print(report_path.resolve())


if __name__ == "__main__":
    main()
