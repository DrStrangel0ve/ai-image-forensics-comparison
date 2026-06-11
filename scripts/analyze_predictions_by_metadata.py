from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from forensic_compare.datasets import collect_labeled_images, discover_layout
from forensic_compare.utils import ensure_dir


SOURCE_LABEL_NAMES = {
    0: "real",
    1: "sd21",
    2: "sdxl",
    3: "sd3",
    4: "dalle3",
    5: "midjourney6",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze prediction scores by metadata labels such as generator source."
    )
    parser.add_argument("--metadata", required=True, help="metadata.csv from export_hf_image_dataset.py.")
    parser.add_argument(
        "--predictions",
        action="append",
        required=True,
        help="Prediction in NAME=PATH form. Repeat for multiple methods.",
    )
    parser.add_argument("--data-dir", default=None, help="Dataset root for reconstructing missing paths.")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def _parse_prediction_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.parent.name or path.stem, path
    name, path = value.split("=", 1)
    if not name or not path:
        raise ValueError(f"Prediction arguments must be NAME=PATH, got {value!r}")
    return name, Path(path)


def _norm(path: str | Path) -> str:
    return str(Path(path).resolve()).replace("\\", "/").lower()


def _metadata_frame(path: Path, split: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[frame["split"] == split].copy()
    if frame.empty:
        raise ValueError(f"No metadata rows for split {split!r}")
    frame["path_key"] = frame["path"].map(_norm)
    frame["source_name"] = frame["source_label"].map(
        lambda value: SOURCE_LABEL_NAMES.get(int(value), f"source_{value}")
    )
    return frame


def _reconstructed_paths(data_dir: str | Path, split: str) -> list[str]:
    layout = discover_layout(data_dir)
    if split in {"validation", "val"} and layout.test:
        folder = layout.test
    elif split == "test" and layout.test:
        folder = layout.test
    elif split == "train" and layout.train:
        folder = layout.train
    elif layout.single:
        folder = layout.single
    else:
        raise ValueError(f"Could not resolve split {split!r} under {data_dir}")
    return [str(path) for path, _label, _class_name in collect_labeled_images(folder)]


def _prediction_frame(
    path: Path,
    method: str,
    data_dir: str | None,
    split: str,
    threshold: float,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "path" not in frame.columns:
        if not data_dir:
            raise ValueError(f"{path} has no path column; provide --data-dir to reconstruct paths")
        paths = _reconstructed_paths(data_dir, split)
        if len(paths) != len(frame):
            raise ValueError(f"{path} has {len(frame)} rows, but reconstructed {len(paths)} paths")
        frame.insert(0, "path", paths)
    frame["method"] = method
    frame["path_key"] = frame["path"].map(_norm)
    frame["predicted_fake"] = frame["fake_score"] >= threshold
    return frame


def _group_rows(joined: pd.DataFrame, threshold: float) -> list[dict]:
    rows = []
    for (method, source_label, source_name), group in joined.groupby(
        ["method", "source_label", "source_name"], sort=True
    ):
        y_true = group["y_true"].astype(int)
        predicted_fake = group["fake_score"] >= threshold
        correct = (predicted_fake.astype(int) == y_true).mean()
        is_fake_group = int(source_label) != 0
        rows.append(
            {
                "method": method,
                "source_label": int(source_label),
                "source_name": source_name,
                "n": int(len(group)),
                "accuracy": float(correct),
                "mean_fake_score": float(group["fake_score"].mean()),
                "fake_score_p10": float(group["fake_score"].quantile(0.10)),
                "fake_score_p50": float(group["fake_score"].quantile(0.50)),
                "fake_score_p90": float(group["fake_score"].quantile(0.90)),
                "false_positive_rate": float(predicted_fake.mean()) if not is_fake_group else None,
                "detection_rate": float(predicted_fake.mean()) if is_fake_group else None,
                "miss_rate": float((~predicted_fake).mean()) if is_fake_group else None,
            }
        )
    return rows


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append("" if pd.isna(value) else f"{value:.4f}")
            else:
                values.append("" if pd.isna(value) else str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    metadata = _metadata_frame(Path(args.metadata), args.split)
    prediction_frames = [
        _prediction_frame(path, method, args.data_dir, args.split, args.threshold)
        for method, path in map(_parse_prediction_arg, args.predictions)
    ]
    predictions = pd.concat(prediction_frames, ignore_index=True)
    joined = predictions.merge(metadata, on="path_key", suffixes=("_prediction", "_metadata"))
    expected = len(predictions)
    if len(joined) != expected:
        raise ValueError(f"Matched {len(joined)} prediction rows but expected {expected}")
    label_mismatches = joined[joined["y_true"].astype(int) != joined["label"].astype(int)]
    if not label_mismatches.empty:
        raise ValueError(
            f"Prediction labels disagree with metadata labels for {len(label_mismatches)} rows"
        )

    summary = pd.DataFrame(_group_rows(joined, args.threshold))
    summary.to_csv(out_dir / "source_summary.csv", index=False)
    joined.to_csv(out_dir / "joined_predictions.csv", index=False)

    report = [
        "# Prediction Source Analysis",
        "",
        f"Split: `{args.split}`",
        f"Rows matched: {len(joined)}",
        f"Threshold: {args.threshold:.2f}",
        "",
        _markdown_table(summary),
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Wrote source analysis to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
