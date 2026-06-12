from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from forensic_compare.metrics import binary_metrics, bootstrap_mean_ci
from forensic_compare.utils import ensure_dir
from scripts.summarize_source_holdout import (
    _metadata_frame,
    _prediction_frame,
    _real_split_keys,
    _source_key,
)


CALIBRATORS = {
    "temperature",
    "temperature_balanced",
    "platt",
    "platt_balanced",
    "isotonic",
    "isotonic_balanced",
}
EPS = 1e-6


@dataclass
class FittedCalibrator:
    name: str
    model: Any
    temperature: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit post-hoc calibrators while holding out one generated source, "
            "then evaluate on that held-out source plus a real test split."
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
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument(
        "--calibrator",
        action="append",
        choices=sorted(CALIBRATORS),
        default=None,
        help="Post-hoc calibrator to evaluate. Repeat to include several.",
    )
    parser.add_argument("--ci-confidence", type=float, default=0.95)
    parser.add_argument("--ci-resamples", type=int, default=2000)
    parser.add_argument("--ci-seed", type=int, default=0)
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


def _logit(scores: np.ndarray) -> np.ndarray:
    clipped = np.clip(scores.astype(float), EPS, 1.0 - EPS)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))


def _class_balanced_weights(y_true: np.ndarray) -> np.ndarray:
    weights = np.ones(len(y_true), dtype=float)
    for label in np.unique(y_true):
        mask = y_true == label
        weights[mask] = len(y_true) / max(2.0 * float(mask.sum()), 1.0)
    return weights


def _fit_temperature(
    y_true: np.ndarray,
    scores: np.ndarray,
    balanced: bool,
) -> FittedCalibrator:
    logits = _logit(scores)
    weights = _class_balanced_weights(y_true) if balanced else np.ones(len(y_true), dtype=float)

    def loss(temperature: float) -> float:
        probabilities = np.clip(_sigmoid(logits / temperature), EPS, 1.0 - EPS)
        losses = -(y_true * np.log(probabilities) + (1 - y_true) * np.log(1.0 - probabilities))
        return float(np.average(losses, weights=weights))

    grid = np.geomspace(0.05, 20.0, 240)
    best_temperature = float(grid[int(np.argmin([loss(float(value)) for value in grid]))])
    lower = max(best_temperature / 2.0, 0.01)
    upper = min(best_temperature * 2.0, 100.0)
    fine_grid = np.geomspace(lower, upper, 240)
    best_temperature = float(fine_grid[int(np.argmin([loss(float(value)) for value in fine_grid]))])
    return FittedCalibrator(
        name="temperature_balanced" if balanced else "temperature",
        model=None,
        temperature=best_temperature,
    )


def _fit_calibrator(name: str, y_true: np.ndarray, scores: np.ndarray) -> FittedCalibrator:
    if name not in CALIBRATORS:
        raise ValueError(f"Unsupported calibrator: {name}")
    balanced = name.endswith("_balanced")
    base_name = name.removesuffix("_balanced")
    if len(np.unique(y_true)) != 2:
        raise ValueError("Calibration split must contain both classes")

    if base_name == "temperature":
        return _fit_temperature(y_true, scores, balanced=balanced)
    if base_name == "platt":
        model = LogisticRegression(
            class_weight="balanced" if balanced else None,
            max_iter=1000,
            random_state=0,
        )
        model.fit(_logit(scores).reshape(-1, 1), y_true)
        return FittedCalibrator(name=name, model=model)
    if base_name == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        weights = _class_balanced_weights(y_true) if balanced else None
        model.fit(np.clip(scores, 0.0, 1.0), y_true, sample_weight=weights)
        return FittedCalibrator(name=name, model=model)
    raise ValueError(f"Unsupported calibrator: {name}")


def _predict_calibrated(calibrator: FittedCalibrator, scores: np.ndarray) -> np.ndarray:
    if calibrator.name.startswith("temperature"):
        if calibrator.temperature is None:
            raise ValueError("Temperature calibrator is missing fitted temperature")
        return _sigmoid(_logit(scores) / calibrator.temperature)
    if calibrator.name.startswith("platt"):
        return calibrator.model.predict_proba(_logit(scores).reshape(-1, 1))[:, 1]
    if calibrator.name.startswith("isotonic"):
        return calibrator.model.predict(np.clip(scores, 0.0, 1.0))
    raise ValueError(f"Unsupported calibrator: {calibrator.name}")


def _rate(scores: np.ndarray) -> float:
    return float((scores >= 0.5).mean()) if len(scores) else math.nan


def _metrics_row(
    group: str,
    method: str,
    heldout_source: str,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    calibrator_name: str,
    n_bins: int,
) -> dict:
    calibration_y = calibration["y_true"].to_numpy(dtype=int)
    calibration_scores = calibration["fake_score"].to_numpy(dtype=float)
    test_y = test["y_true"].to_numpy(dtype=int)
    raw_scores = test["fake_score"].to_numpy(dtype=float)
    calibrator = _fit_calibrator(calibrator_name, calibration_y, calibration_scores)
    calibrated_scores = _predict_calibrated(calibrator, raw_scores)

    raw_metrics = binary_metrics(test_y, raw_scores, n_bins=n_bins)
    calibrated_metrics = binary_metrics(test_y, calibrated_scores, n_bins=n_bins)
    fake_test = test[test["source_name"] == heldout_source]
    real_test = test[test["source_name"] != heldout_source]
    raw_fake_scores = fake_test["fake_score"].to_numpy(dtype=float)
    raw_real_scores = real_test["fake_score"].to_numpy(dtype=float)
    calibrated_fake_scores = calibrated_scores[test["source_name"].to_numpy() == heldout_source]
    calibrated_real_scores = calibrated_scores[test["source_name"].to_numpy() != heldout_source]

    return {
        "group": group,
        "method": method,
        "calibrator": calibrator.name,
        "heldout_source": heldout_source,
        "calibration_samples": int(len(calibration)),
        "test_samples": int(len(test)),
        "heldout_fake_samples": int(len(fake_test)),
        "real_test_samples": int(len(real_test)),
        "temperature": calibrator.temperature,
        "raw_accuracy": float(raw_metrics["accuracy"]),
        "calibrated_accuracy": float(calibrated_metrics["accuracy"]),
        "raw_f1": float(raw_metrics["f1"]),
        "calibrated_f1": float(calibrated_metrics["f1"]),
        "raw_roc_auc": None if raw_metrics["roc_auc"] is None else float(raw_metrics["roc_auc"]),
        "calibrated_roc_auc": None
        if calibrated_metrics["roc_auc"] is None
        else float(calibrated_metrics["roc_auc"]),
        "raw_brier_score": float(raw_metrics["brier_score"]),
        "calibrated_brier_score": float(calibrated_metrics["brier_score"]),
        "raw_expected_calibration_error": float(raw_metrics["expected_calibration_error"]),
        "calibrated_expected_calibration_error": float(
            calibrated_metrics["expected_calibration_error"]
        ),
        "raw_maximum_calibration_error": float(raw_metrics["maximum_calibration_error"]),
        "calibrated_maximum_calibration_error": float(
            calibrated_metrics["maximum_calibration_error"]
        ),
        "raw_real_fpr": _rate(raw_real_scores),
        "calibrated_real_fpr": _rate(calibrated_real_scores),
        "raw_fake_detection": _rate(raw_fake_scores),
        "calibrated_fake_detection": _rate(calibrated_fake_scores),
        "delta_accuracy": float(calibrated_metrics["accuracy"] - raw_metrics["accuracy"]),
        "delta_brier_score": float(calibrated_metrics["brier_score"] - raw_metrics["brier_score"]),
        "delta_expected_calibration_error": float(
            calibrated_metrics["expected_calibration_error"]
            - raw_metrics["expected_calibration_error"]
        ),
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


def _summary_frame(
    detail: pd.DataFrame,
    confidence: float,
    n_resamples: int,
    seed: int,
) -> pd.DataFrame:
    metric_map = {
        "raw_accuracy": "mean_raw_accuracy",
        "calibrated_accuracy": "mean_calibrated_accuracy",
        "raw_brier_score": "mean_raw_brier_score",
        "calibrated_brier_score": "mean_calibrated_brier_score",
        "raw_expected_calibration_error": "mean_raw_ece",
        "calibrated_expected_calibration_error": "mean_calibrated_ece",
        "raw_real_fpr": "mean_raw_real_fpr",
        "calibrated_real_fpr": "mean_calibrated_real_fpr",
        "raw_fake_detection": "mean_raw_fake_detection",
        "calibrated_fake_detection": "mean_calibrated_fake_detection",
        "calibrated_roc_auc": "mean_calibrated_roc_auc",
    }
    rows = []
    for (method, calibrator), group in detail.groupby(["method", "calibrator"], sort=True):
        row = {
            "method": method,
            "calibrator": calibrator,
            "n_holdouts": int(len(group)),
        }
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
    return pd.DataFrame(rows).sort_values(
        ["mean_calibrated_brier_score", "mean_calibrated_ece"],
        ascending=[True, True],
    )


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    calibrators = args.calibrator or ["temperature", "platt", "isotonic"]
    metadata = _metadata_frame(Path(args.metadata), args.split)

    frames = []
    for group, method, path in map(_parse_prediction_arg, args.predictions):
        frame = _prediction_frame(path, method, args.data_dir, args.split)
        frame["group"] = group
        frames.append(frame)
    predictions = pd.concat(frames, ignore_index=True)
    joined = predictions.merge(metadata, on="path_key", suffixes=("_prediction", "_metadata"))
    expected = len(predictions)
    if len(joined) != expected:
        raise ValueError(f"Matched {len(joined)} prediction rows but expected {expected}")
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
            for calibrator in calibrators:
                rows.append(
                    _metrics_row(
                        group,
                        method,
                        source_name,
                        calibration,
                        test,
                        calibrator,
                        args.n_bins,
                    )
                )
    if not rows:
        raise ValueError("No source-holdout calibration rows could be computed")

    rows = sorted(rows, key=lambda row: (row["method"], row["calibrator"], row["group"], row["heldout_source"]))
    detail = pd.DataFrame(rows)
    summary = _summary_frame(detail, args.ci_confidence, args.ci_resamples, args.ci_seed)

    detail.to_csv(out_dir / "source_holdout_calibration.csv", index=False)
    summary.to_csv(out_dir / "source_holdout_calibration_summary.csv", index=False)
    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["artifact", "path"])
        for artifact in [
            "source_holdout_calibration.csv",
            "source_holdout_calibration_summary.csv",
            "report.md",
        ]:
            writer.writerow([artifact, str((out_dir / artifact).resolve())])

    summary_columns = [
        "method",
        "calibrator",
        "n_holdouts",
        "mean_raw_accuracy",
        "mean_calibrated_accuracy",
        "mean_raw_brier_score",
        "mean_calibrated_brier_score",
        "mean_calibrated_brier_score_ci_low",
        "mean_calibrated_brier_score_ci_high",
        "mean_raw_ece",
        "mean_calibrated_ece",
        "mean_calibrated_ece_ci_low",
        "mean_calibrated_ece_ci_high",
        "mean_calibrated_real_fpr",
        "mean_calibrated_fake_detection",
    ]
    detail_columns = [
        "group",
        "method",
        "calibrator",
        "heldout_source",
        "raw_accuracy",
        "calibrated_accuracy",
        "raw_brier_score",
        "calibrated_brier_score",
        "raw_expected_calibration_error",
        "calibrated_expected_calibration_error",
        "calibrated_real_fpr",
        "calibrated_fake_detection",
    ]
    report = [
        "# Source-Heldout Calibration Summary",
        "",
        f"Split: `{args.split}`",
        f"Real test fraction: `{args.real_test_fraction}`",
        f"Seed: `{args.seed}`",
        f"Reliability bins: `{args.n_bins}`",
        f"Mean confidence intervals: `{args.ci_confidence:.0%}` bootstrap over held-out sources/seeds, `{args.ci_resamples}` resamples",
        "",
        "Each row holds out one generated source. Calibrators are fitted on all other generated sources plus the calibration half of real images, then evaluated on the held-out generated source plus the real test half.",
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
    print(f"Wrote source-holdout calibration summary to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
