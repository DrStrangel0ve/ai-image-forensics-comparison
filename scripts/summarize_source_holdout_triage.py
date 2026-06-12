from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from forensic_compare.utils import ensure_dir
from scripts.summarize_source_holdout import (
    _metadata_frame,
    _prediction_frame,
    _real_split_keys,
    _source_key,
)
from scripts.summarize_source_holdout_calibration import (
    CALIBRATORS,
    _fit_calibrator,
    _predict_calibrated,
)


SCORE_MODES = {"raw", *CALIBRATORS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate source-heldout two-threshold triage: likely real, uncertain, likely fake."
        )
    )
    parser.add_argument("--metadata", required=True, help="metadata.csv from export_hf_image_dataset.py.")
    parser.add_argument(
        "--predictions",
        action="append",
        required=True,
        help=(
            "Prediction in GROUP:METHOD=PATH form. GROUP is optional; METHOD=PATH "
            "is treated as group 'default'. Repeat for each seed/method."
        ),
    )
    parser.add_argument("--data-dir", default=None, help="Dataset root for reconstructing missing paths.")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--real-source-label", default="0")
    parser.add_argument("--real-test-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--max-real-fpr",
        type=float,
        default=0.05,
        help="Calibration constraint for likely-fake calls: max real images called fake.",
    )
    parser.add_argument(
        "--max-fake-clearance",
        type=float,
        default=0.05,
        help="Calibration constraint for likely-real calls: max fake images cleared as real.",
    )
    parser.add_argument(
        "--score-mode",
        action="append",
        choices=sorted(SCORE_MODES),
        default=None,
        help="Score transform before threshold selection. Repeat to include several modes.",
    )
    return parser.parse_args()


def _parse_prediction_arg(value: str) -> tuple[str, str, Path]:
    if "=" not in value:
        raise ValueError(f"Prediction arguments must be GROUP:METHOD=PATH, got {value!r}")
    left, path = value.split("=", 1)
    if ":" in left:
        group, method = left.split(":", 1)
    else:
        group = "default"
        method = left
    if not group or not method or not path:
        raise ValueError(f"Prediction arguments must be GROUP:METHOD=PATH, got {value!r}")
    return group, method, Path(path)


def _validate_fraction(name: str, value: float) -> None:
    if not 0.0 <= value < 1.0:
        raise ValueError(f"{name} must be in [0, 1), got {value}")


def _fake_threshold(real_scores: np.ndarray, max_real_fpr: float) -> float:
    if len(real_scores) == 0:
        return 1.0
    ordered = np.sort(np.clip(real_scores, 0.0, 1.0))
    allowed_false_calls = int(math.floor(max_real_fpr * len(ordered)))
    if allowed_false_calls <= 0:
        return float(np.nextafter(ordered[-1], 1.0))
    index = max(len(ordered) - allowed_false_calls, 0)
    return float(ordered[index])


def _real_threshold(fake_scores: np.ndarray, max_fake_clearance: float) -> float:
    if len(fake_scores) == 0:
        return 0.0
    ordered = np.sort(np.clip(fake_scores, 0.0, 1.0))
    allowed_false_clearances = int(math.floor(max_fake_clearance * len(ordered)))
    if allowed_false_clearances <= 0:
        return float(np.nextafter(ordered[0], 0.0))
    index = min(allowed_false_clearances - 1, len(ordered) - 1)
    return float(ordered[index])


def _thresholds(
    y_true: np.ndarray,
    scores: np.ndarray,
    max_real_fpr: float,
    max_fake_clearance: float,
) -> tuple[float, float]:
    real_scores = scores[y_true == 0]
    fake_scores = scores[y_true == 1]
    real_threshold = _real_threshold(fake_scores, max_fake_clearance)
    fake_threshold = _fake_threshold(real_scores, max_real_fpr)
    if real_threshold >= fake_threshold:
        real_threshold = float(np.nextafter(fake_threshold, 0.0))
    return real_threshold, fake_threshold


def _transform_scores(
    score_mode: str,
    calibration_y: np.ndarray,
    calibration_scores: np.ndarray,
    test_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float | None]:
    if score_mode == "raw":
        return calibration_scores, test_scores, None
    calibrator = _fit_calibrator(score_mode, calibration_y, calibration_scores)
    return (
        _predict_calibrated(calibrator, calibration_scores),
        _predict_calibrated(calibrator, test_scores),
        calibrator.temperature,
    )


def _triage_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    real_threshold: float,
    fake_threshold: float,
) -> dict:
    likely_real = scores <= real_threshold
    likely_fake = scores >= fake_threshold
    decided = likely_real | likely_fake
    correct_real = likely_real & (y_true == 0)
    correct_fake = likely_fake & (y_true == 1)
    real = y_true == 0
    fake = y_true == 1
    decided_count = int(decided.sum())
    fake_call_count = int(likely_fake.sum())
    real_call_count = int(likely_real.sum())
    return {
        "coverage": float(decided.mean()) if len(y_true) else math.nan,
        "uncertain_rate": float((~decided).mean()) if len(y_true) else math.nan,
        "triage_accuracy": float((correct_real.sum() + correct_fake.sum()) / decided_count)
        if decided_count
        else math.nan,
        "fake_call_rate": float(likely_fake.mean()) if len(y_true) else math.nan,
        "real_call_rate": float(likely_real.mean()) if len(y_true) else math.nan,
        "fake_precision": float(correct_fake.sum() / fake_call_count) if fake_call_count else math.nan,
        "real_precision": float(correct_real.sum() / real_call_count) if real_call_count else math.nan,
        "real_fpr": float((likely_fake & real).sum() / max(real.sum(), 1)),
        "fake_false_clearance": float((likely_real & fake).sum() / max(fake.sum(), 1)),
        "fake_detection": float((likely_fake & fake).sum() / max(fake.sum(), 1)),
        "real_clearance": float((likely_real & real).sum() / max(real.sum(), 1)),
    }


def _row(
    group: str,
    method: str,
    heldout_source: str,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    score_mode: str,
    max_real_fpr: float,
    max_fake_clearance: float,
) -> dict:
    calibration_y = calibration["y_true"].to_numpy(dtype=int)
    calibration_scores = calibration["fake_score"].to_numpy(dtype=float)
    test_y = test["y_true"].to_numpy(dtype=int)
    test_scores = test["fake_score"].to_numpy(dtype=float)
    transformed_calibration_scores, transformed_test_scores, temperature = _transform_scores(
        score_mode, calibration_y, calibration_scores, test_scores
    )
    real_threshold, fake_threshold = _thresholds(
        calibration_y,
        transformed_calibration_scores,
        max_real_fpr,
        max_fake_clearance,
    )
    calibration_metrics = _triage_metrics(
        calibration_y,
        transformed_calibration_scores,
        real_threshold,
        fake_threshold,
    )
    test_metrics = _triage_metrics(test_y, transformed_test_scores, real_threshold, fake_threshold)
    row = {
        "group": group,
        "method": method,
        "score_mode": score_mode,
        "heldout_source": heldout_source,
        "calibration_samples": int(len(calibration)),
        "test_samples": int(len(test)),
        "real_threshold": float(real_threshold),
        "fake_threshold": float(fake_threshold),
        "temperature": temperature,
    }
    row.update({f"calibration_{key}": value for key, value in calibration_metrics.items()})
    row.update({f"test_{key}": value for key, value in test_metrics.items()})
    return row


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


def main() -> None:
    args = parse_args()
    _validate_fraction("--max-real-fpr", args.max_real_fpr)
    _validate_fraction("--max-fake-clearance", args.max_fake_clearance)
    out_dir = ensure_dir(args.out_dir)
    score_modes = args.score_mode or ["raw", "temperature_balanced"]
    metadata = _metadata_frame(Path(args.metadata), args.split)

    frames = []
    for group, method, path in map(_parse_prediction_arg, args.predictions):
        frame = _prediction_frame(path, method, args.data_dir, args.split)
        frame["group"] = group
        frames.append(frame)
    predictions = pd.concat(frames, ignore_index=True)
    joined = predictions.merge(metadata, on="path_key", suffixes=("_prediction", "_metadata"))
    if len(joined) != len(predictions):
        raise ValueError(f"Matched {len(joined)} prediction rows but expected {len(predictions)}")
    label_mismatches = joined[joined["y_true"].astype(int) != joined["label"].astype(int)]
    if not label_mismatches.empty:
        raise ValueError(
            f"Prediction labels disagree with metadata labels for {len(label_mismatches)} rows"
        )

    real_key = _source_key(args.real_source_label)
    real_mask = joined["source_key"] == real_key
    fake_sources = sorted(
        joined.loc[~real_mask, ["source_key", "source_name"]]
        .drop_duplicates()
        .itertuples(index=False)
    )
    if not fake_sources:
        raise ValueError("No fake source labels were found in the joined predictions")
    real_test_keys = _real_split_keys(joined.loc[real_mask, "path_key"], args.real_test_fraction, args.seed)

    rows = []
    for (group, method), method_frame in joined.groupby(["group", "method"], sort=True):
        is_real = method_frame["source_key"] == real_key
        real_test = method_frame["path_key"].isin(real_test_keys)
        for source_key, source_name in fake_sources:
            heldout_fake = method_frame["source_key"] == source_key
            calibration = method_frame[(is_real & ~real_test) | (~is_real & ~heldout_fake)]
            test = method_frame[(is_real & real_test) | heldout_fake]
            if calibration["y_true"].nunique() < 2 or test["y_true"].nunique() < 2:
                continue
            for score_mode in score_modes:
                rows.append(
                    _row(
                        group,
                        method,
                        source_name,
                        calibration,
                        test,
                        score_mode,
                        args.max_real_fpr,
                        args.max_fake_clearance,
                    )
                )
    if not rows:
        raise ValueError("No source-holdout triage rows could be computed")

    detail = pd.DataFrame(sorted(rows, key=lambda row: (row["method"], row["score_mode"], row["group"], row["heldout_source"])))
    summary = (
        detail.groupby(["method", "score_mode"], as_index=False)
        .agg(
            n_holdouts=("heldout_source", "count"),
            mean_test_coverage=("test_coverage", "mean"),
            mean_test_triage_accuracy=("test_triage_accuracy", "mean"),
            mean_test_real_fpr=("test_real_fpr", "mean"),
            mean_test_fake_false_clearance=("test_fake_false_clearance", "mean"),
            mean_test_fake_detection=("test_fake_detection", "mean"),
            mean_test_real_clearance=("test_real_clearance", "mean"),
            mean_test_fake_precision=("test_fake_precision", "mean"),
            mean_test_real_precision=("test_real_precision", "mean"),
            mean_real_threshold=("real_threshold", "mean"),
            mean_fake_threshold=("fake_threshold", "mean"),
        )
        .sort_values(["mean_test_triage_accuracy", "mean_test_coverage"], ascending=[False, False])
    )

    detail.to_csv(out_dir / "source_holdout_triage.csv", index=False)
    summary.to_csv(out_dir / "source_holdout_triage_summary.csv", index=False)
    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["artifact", "path"])
        for artifact in [
            "source_holdout_triage.csv",
            "source_holdout_triage_summary.csv",
            "report.md",
        ]:
            writer.writerow([artifact, str((out_dir / artifact).resolve())])

    summary_columns = [
        "method",
        "score_mode",
        "mean_test_coverage",
        "mean_test_triage_accuracy",
        "mean_test_real_fpr",
        "mean_test_fake_false_clearance",
        "mean_test_fake_detection",
        "mean_test_real_clearance",
    ]
    detail_columns = [
        "group",
        "method",
        "score_mode",
        "heldout_source",
        "real_threshold",
        "fake_threshold",
        "test_coverage",
        "test_triage_accuracy",
        "test_real_fpr",
        "test_fake_false_clearance",
        "test_fake_detection",
        "test_real_clearance",
    ]
    report = [
        "# Source-Heldout Triage Summary",
        "",
        f"Split: `{args.split}`",
        f"Real test fraction: `{args.real_test_fraction}`",
        f"Seed: `{args.seed}`",
        f"Calibration max real FPR: `{args.max_real_fpr}`",
        f"Calibration max fake clearance: `{args.max_fake_clearance}`",
        "",
        "Scores above the fake threshold are called likely fake. Scores below the real threshold are called likely real. Scores between the two thresholds are marked uncertain.",
        "",
        "## Method Summary",
        "",
        _markdown_table(summary.to_dict("records"), summary_columns),
        "",
        "## Held-Out Source Detail",
        "",
        _markdown_table(detail.to_dict("records"), detail_columns),
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(_markdown_table(summary.to_dict("records"), summary_columns))
    print(f"Wrote source-holdout triage summary to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
