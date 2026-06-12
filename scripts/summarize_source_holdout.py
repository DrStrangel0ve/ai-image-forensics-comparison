from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from forensic_compare.datasets import collect_labeled_images, discover_layout, stable_path_score
from forensic_compare.metrics import best_threshold, binary_metrics, bootstrap_mean_ci
from forensic_compare.utils import ensure_dir


SOURCE_LABEL_NAMES = {
    "0": "real",
    "1": "sd21",
    "2": "sdxl",
    "3": "sd3",
    "4": "dalle3",
    "5": "midjourney6",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize leave-one-source-out threshold behavior for prediction CSVs joined "
            "with metadata from export_hf_image_dataset.py."
        )
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
    parser.add_argument("--real-source-label", default="0")
    parser.add_argument("--default-threshold", type=float, default=0.5)
    parser.add_argument("--real-test-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--objective",
        choices=["accuracy", "f1"],
        default="accuracy",
        help="Metric to maximize when selecting source-calibrated and oracle thresholds.",
    )
    parser.add_argument("--ci-confidence", type=float, default=0.95)
    parser.add_argument("--ci-resamples", type=int, default=2000)
    parser.add_argument("--ci-seed", type=int, default=0)
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


def _source_key(value) -> str:
    if pd.isna(value):
        return "missing"
    text = str(value)
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _source_name(value) -> str:
    key = _source_key(value)
    return SOURCE_LABEL_NAMES.get(key, f"source_{key}")


def _metadata_frame(path: Path, split: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"path", "split", "label", "source_label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing metadata columns: {sorted(missing)}")
    frame = frame[frame["split"] == split].copy()
    if frame.empty:
        raise ValueError(f"No metadata rows for split {split!r}")
    frame["path_key"] = frame["path"].map(_norm)
    frame["source_key"] = frame["source_label"].map(_source_key)
    frame["source_name"] = frame["source_label"].map(_source_name)
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


def _prediction_frame(path: Path, method: str, data_dir: str | None, split: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"y_true", "fake_score"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing prediction columns: {sorted(missing)}")
    if "path" not in frame.columns:
        if not data_dir:
            raise ValueError(f"{path} has no path column; provide --data-dir to reconstruct paths")
        paths = _reconstructed_paths(data_dir, split)
        if len(paths) != len(frame):
            raise ValueError(f"{path} has {len(frame)} rows, but reconstructed {len(paths)} paths")
        frame.insert(0, "path", paths)
    frame["method"] = method
    frame["path_key"] = frame["path"].map(_norm)
    return frame


def _real_split_keys(real_paths: pd.Series, fraction: float, seed: int) -> set[str]:
    if not 0.0 < fraction < 1.0:
        raise ValueError("--real-test-fraction must be between 0 and 1")
    unique_paths = sorted(set(real_paths))
    if len(unique_paths) <= 1:
        return set(unique_paths)
    n_test = int(round(len(unique_paths) * fraction))
    n_test = min(max(n_test, 1), len(unique_paths) - 1)
    ordered = sorted(unique_paths, key=lambda path: stable_path_score(path, seed))
    return set(ordered[:n_test])


def _metrics_at(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    metrics = binary_metrics(y_true, scores, threshold=threshold)
    return {
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "roc_auc": None if metrics["roc_auc"] is None else float(metrics["roc_auc"]),
    }


def _row_for_holdout(
    method: str,
    source_key: str,
    source_name: str,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    objective: str,
    default_threshold: float,
) -> dict:
    calib_y = calibration["y_true"].to_numpy(dtype=int)
    calib_scores = calibration["fake_score"].to_numpy(dtype=float)
    test_y = test["y_true"].to_numpy(dtype=int)
    test_scores = test["fake_score"].to_numpy(dtype=float)
    source_threshold, source_metrics = best_threshold(calib_y, calib_scores, objective=objective)
    oracle_threshold, _oracle_threshold_metrics = best_threshold(
        test_y, test_scores, objective=objective
    )
    default_metrics = _metrics_at(test_y, test_scores, default_threshold)
    calibrated_metrics = _metrics_at(test_y, test_scores, source_threshold)
    oracle_metrics = _metrics_at(test_y, test_scores, oracle_threshold)
    fake_test = test[test["source_key"] == source_key]
    real_test = test[test["source_key"] != source_key]
    real_fpr = (
        float((real_test["fake_score"].to_numpy(dtype=float) >= source_threshold).mean())
        if len(real_test)
        else None
    )
    fake_detection = (
        float((fake_test["fake_score"].to_numpy(dtype=float) >= source_threshold).mean())
        if len(fake_test)
        else None
    )
    return {
        "method": method,
        "heldout_source": source_name,
        "heldout_source_key": source_key,
        "calibration_samples": int(len(calibration)),
        "test_samples": int(len(test)),
        "heldout_fake_samples": int(len(fake_test)),
        "real_test_samples": int(len(real_test)),
        "default_accuracy": default_metrics["accuracy"],
        "default_f1": default_metrics["f1"],
        "source_threshold": float(source_threshold),
        "source_threshold_calibration_accuracy": float(source_metrics["accuracy"]),
        "source_threshold_accuracy": calibrated_metrics["accuracy"],
        "source_threshold_f1": calibrated_metrics["f1"],
        "source_threshold_real_fpr": real_fpr,
        "source_threshold_fake_detection": fake_detection,
        "oracle_threshold": float(oracle_threshold),
        "oracle_accuracy": oracle_metrics["accuracy"],
        "oracle_f1": oracle_metrics["f1"],
        "roc_auc": default_metrics["roc_auc"],
    }


def _format_cell(value) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _markdown_table(rows: list[dict], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_cell(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def _method_summary(
    detail: pd.DataFrame,
    confidence: float,
    n_resamples: int,
    seed: int,
) -> pd.DataFrame:
    metric_map = {
        "default_accuracy": "mean_default_accuracy",
        "source_threshold_accuracy": "mean_source_threshold_accuracy",
        "oracle_accuracy": "mean_oracle_accuracy",
        "roc_auc": "mean_roc_auc",
    }
    rows = []
    for method, group in detail.groupby("method", sort=True):
        row = {"method": method, "n_holdouts": int(len(group))}
        for index, (metric, output_name) in enumerate(metric_map.items()):
            interval = bootstrap_mean_ci(
                group[metric].to_numpy(dtype=float),
                confidence=confidence,
                n_resamples=n_resamples,
                seed=seed + index,
            )
            row[output_name] = interval["mean"]
            row[f"{output_name}_ci_low"] = interval["ci_low"]
            row[f"{output_name}_ci_high"] = interval["ci_high"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mean_source_threshold_accuracy", ascending=False)


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    metadata = _metadata_frame(Path(args.metadata), args.split)
    prediction_frames = [
        _prediction_frame(path, method, args.data_dir, args.split)
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

    real_mask = joined["source_key"] == _source_key(args.real_source_label)
    fake_sources = sorted(joined.loc[~real_mask, ["source_key", "source_name"]].drop_duplicates().itertuples(index=False))
    if not fake_sources:
        raise ValueError("No fake source labels were found in the joined predictions")
    real_test_keys = _real_split_keys(joined.loc[real_mask, "path_key"], args.real_test_fraction, args.seed)
    rows = []
    for method, method_frame in joined.groupby("method", sort=True):
        is_real = method_frame["source_key"] == _source_key(args.real_source_label)
        real_test = method_frame["path_key"].isin(real_test_keys)
        for source_key, source_name in fake_sources:
            heldout_fake = method_frame["source_key"] == source_key
            calibration = method_frame[(is_real & ~real_test) | (~is_real & ~heldout_fake)]
            test = method_frame[(is_real & real_test) | heldout_fake]
            if calibration["y_true"].nunique() < 2 or test["y_true"].nunique() < 2:
                continue
            rows.append(
                _row_for_holdout(
                    method,
                    source_key,
                    source_name,
                    calibration,
                    test,
                    args.objective,
                    args.default_threshold,
                )
            )

    rows = sorted(rows, key=lambda row: (row["method"], row["heldout_source"]))
    if not rows:
        raise ValueError("No source-holdout rows could be computed")

    with (out_dir / "source_holdout.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    detail = pd.DataFrame(rows)
    summary = _method_summary(detail, args.ci_confidence, args.ci_resamples, args.ci_seed)
    summary.to_csv(out_dir / "source_holdout_method_summary.csv", index=False)
    summary_rows = summary.to_dict(orient="records")

    detail_columns = [
        "method",
        "heldout_source",
        "default_accuracy",
        "source_threshold",
        "source_threshold_accuracy",
        "source_threshold_real_fpr",
        "source_threshold_fake_detection",
        "oracle_accuracy",
        "roc_auc",
    ]
    summary_columns = [
        "method",
        "n_holdouts",
        "mean_default_accuracy",
        "mean_default_accuracy_ci_low",
        "mean_default_accuracy_ci_high",
        "mean_source_threshold_accuracy",
        "mean_source_threshold_accuracy_ci_low",
        "mean_source_threshold_accuracy_ci_high",
        "mean_oracle_accuracy",
        "mean_roc_auc",
    ]
    report = [
        "# Source-Heldout Threshold Summary",
        "",
        f"Split: `{args.split}`",
        f"Threshold objective: `{args.objective}`",
        f"Real test fraction: `{args.real_test_fraction}`",
        f"Seed: `{args.seed}`",
        f"Mean confidence intervals: `{args.ci_confidence:.0%}` bootstrap over held-out sources/seeds, `{args.ci_resamples}` resamples",
        "",
        "For each held-out fake source, the source threshold is selected on all other fake sources plus a deterministic calibration subset of real images. It is then evaluated on the held-out fake source plus the remaining real images.",
        "",
        "## Method Summary",
        "",
        _markdown_table(summary_rows, summary_columns),
        "",
        "## Held-Out Source Detail",
        "",
        _markdown_table(rows, detail_columns),
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(_markdown_table(summary_rows, summary_columns))
    print(f"Wrote source-holdout summary to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
