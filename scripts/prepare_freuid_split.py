from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forensic_compare.datasets import stable_path_score
from forensic_compare.utils import write_json


REQUIRED_COLUMNS = ["id", "image_path", "label", "type"]


def _json_scalar(value: object) -> object:
    if hasattr(value, "item"):
        return value.item()
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a type-and-label-stratified FREUID train/validation split.")
    parser.add_argument("--train-labels", required=True, help="Kaggle train_labels.csv.")
    parser.add_argument("--train-out", required=True, help="Training split CSV to write.")
    parser.add_argument("--val-out", required=True, help="Validation split CSV to write.")
    parser.add_argument("--manifest-out", default=None, help="Optional JSON manifest; defaults beside --val-out.")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--strata-columns",
        nargs="+",
        default=["type", "label"],
        help="Columns used for stratification. Defaults to type label.",
    )
    return parser.parse_args()


def _validate_labels(frame: pd.DataFrame, strata_columns: list[str], val_fraction: float) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"train_labels is missing required columns: {missing}")
    missing_strata = [column for column in strata_columns if column not in frame.columns]
    if missing_strata:
        raise ValueError(f"strata columns are missing from train_labels: {missing_strata}")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("--val-fraction must be in (0, 1)")
    if not frame["label"].isin([0, 1]).all():
        raise ValueError("FREUID labels must be binary 0/1")
    if frame["id"].astype(str).duplicated().any():
        raise ValueError("train_labels contains duplicate ids")


def build_freuid_split(
    train_labels_path: Path,
    train_out: Path,
    val_out: Path,
    manifest_out: Path,
    val_fraction: float = 0.2,
    seed: int = 7,
    strata_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    strata_columns = strata_columns or ["type", "label"]
    labels = pd.read_csv(train_labels_path)
    _validate_labels(labels, strata_columns, val_fraction)

    labels = labels.copy()
    labels["_split_score"] = labels["id"].astype(str).map(lambda value: stable_path_score(value, seed))
    train_indices: list[int] = []
    val_indices: list[int] = []
    for _stratum, group in labels.groupby(strata_columns, sort=True):
        ordered = group.sort_values(["_split_score", "id"], kind="mergesort")
        if len(ordered) == 1:
            train_indices.extend(ordered.index.tolist())
            continue
        n_val = min(len(ordered) - 1, max(1, int(round(len(ordered) * val_fraction))))
        val_indices.extend(ordered.index[:n_val].tolist())
        train_indices.extend(ordered.index[n_val:].tolist())

    train_frame = labels.loc[train_indices].drop(columns=["_split_score"]).sort_values("id").reset_index(drop=True)
    val_frame = labels.loc[val_indices].drop(columns=["_split_score"]).sort_values("id").reset_index(drop=True)

    train_out.parent.mkdir(parents=True, exist_ok=True)
    val_out.parent.mkdir(parents=True, exist_ok=True)
    train_frame.to_csv(train_out, index=False)
    val_frame.to_csv(val_out, index=False)

    stratum_summary = []
    for key, group in labels.drop(columns=["_split_score"]).groupby(strata_columns, sort=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        key_dict = {column: _json_scalar(value) for column, value in zip(strata_columns, key_tuple)}
        train_count = int(train_frame.merge(group[["id"]], on="id", how="inner").shape[0])
        val_count = int(val_frame.merge(group[["id"]], on="id", how="inner").shape[0])
        stratum_summary.append({**key_dict, "train": train_count, "val": val_count, "total": int(len(group))})

    manifest = {
        "train_labels_path": str(train_labels_path),
        "train_out": str(train_out),
        "val_out": str(val_out),
        "n_total": int(len(labels)),
        "n_train": int(len(train_frame)),
        "n_val": int(len(val_frame)),
        "val_fraction_requested": val_fraction,
        "val_fraction_actual": float(len(val_frame) / len(labels)) if len(labels) else 0.0,
        "seed": int(seed),
        "strata_columns": strata_columns,
        "train_label_counts": {str(k): int(v) for k, v in train_frame["label"].value_counts().sort_index().items()},
        "val_label_counts": {str(k): int(v) for k, v in val_frame["label"].value_counts().sort_index().items()},
        "strata": stratum_summary,
    }
    write_json(manifest, manifest_out)
    return train_frame, val_frame, manifest


def main() -> None:
    args = parse_args()
    val_out = Path(args.val_out)
    manifest_out = Path(args.manifest_out) if args.manifest_out else val_out.with_suffix(".manifest.json")
    build_freuid_split(
        train_labels_path=Path(args.train_labels),
        train_out=Path(args.train_out),
        val_out=val_out,
        manifest_out=manifest_out,
        val_fraction=args.val_fraction,
        seed=args.seed,
        strata_columns=list(args.strata_columns),
    )
    print(Path(args.train_out).resolve())
    print(val_out.resolve())
    print(manifest_out.resolve())


if __name__ == "__main__":
    main()
