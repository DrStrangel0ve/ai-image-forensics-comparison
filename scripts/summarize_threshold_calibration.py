from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from forensic_compare.metrics import binary_metrics
from forensic_compare.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare default, clean-calibrated, and oracle thresholds for predictions."
    )
    parser.add_argument(
        "--clean",
        action="append",
        required=True,
        help="Clean validation predictions in METHOD=PATH form. Repeat for each method.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        help="Variant predictions in VARIANT:METHOD=PATH form. Repeat for each variant/method.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--objective",
        choices=["accuracy", "f1"],
        default="accuracy",
        help="Metric to maximize when selecting thresholds.",
    )
    return parser.parse_args()


def _parse_clean(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Clean arguments must be METHOD=PATH, got {value!r}")
    method, path = value.split("=", 1)
    if not method or not path:
        raise ValueError(f"Clean arguments must be METHOD=PATH, got {value!r}")
    return method, Path(path)


def _parse_variant(value: str) -> tuple[str, str, Path]:
    if "=" not in value:
        raise ValueError(f"Variant arguments must be VARIANT:METHOD=PATH, got {value!r}")
    left, path = value.split("=", 1)
    if ":" not in left:
        raise ValueError(f"Variant arguments must be VARIANT:METHOD=PATH, got {value!r}")
    variant, method = left.split(":", 1)
    if not variant or not method or not path:
        raise ValueError(f"Variant arguments must be VARIANT:METHOD=PATH, got {value!r}")
    return variant, method, Path(path)


def _load_predictions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    return frame["y_true"].to_numpy(dtype=int), frame["fake_score"].to_numpy(dtype=float)


def _threshold_candidates(scores: np.ndarray) -> np.ndarray:
    finite_scores = np.unique(np.clip(scores[np.isfinite(scores)], 0.0, 1.0))
    if len(finite_scores) == 0:
        return np.asarray([0.5], dtype=float)
    midpoint_candidates = (finite_scores[:-1] + finite_scores[1:]) / 2.0
    return np.unique(np.concatenate(([0.0, 0.5, 1.0], finite_scores, midpoint_candidates)))


def _fast_threshold_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = scores >= threshold
    y_true_bool = y_true.astype(bool)
    tp = float(np.logical_and(y_pred, y_true_bool).sum())
    fp = float(np.logical_and(y_pred, ~y_true_bool).sum())
    tn = float(np.logical_and(~y_pred, ~y_true_bool).sum())
    fn = float(np.logical_and(~y_pred, y_true_bool).sum())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    return {
        "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1.0),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": float(threshold),
        "n_samples": int(len(y_true)),
    }


def _best_threshold(y_true: np.ndarray, scores: np.ndarray, objective: str) -> tuple[float, dict]:
    best_threshold = 0.5
    best_metrics = _fast_threshold_metrics(y_true, scores, threshold=best_threshold)
    best_value = float(best_metrics[objective])
    for threshold in _threshold_candidates(scores):
        metrics = _fast_threshold_metrics(y_true, scores, threshold=float(threshold))
        value = float(metrics[objective])
        if value > best_value or (value == best_value and abs(float(threshold) - 0.5) < abs(best_threshold - 0.5)):
            best_threshold = float(threshold)
            best_metrics = metrics
            best_value = value
    return best_threshold, best_metrics


def _row(
    variant: str,
    method: str,
    path: Path,
    y_true: np.ndarray,
    scores: np.ndarray,
    clean_threshold: float,
    objective: str,
) -> dict:
    default_metrics = binary_metrics(y_true, scores, threshold=0.5)
    clean_threshold_metrics = binary_metrics(y_true, scores, threshold=clean_threshold)
    oracle_threshold, oracle_metrics = _best_threshold(y_true, scores, objective=objective)
    return {
        "variant": variant,
        "method": method,
        "predictions_path": str(path),
        "n_samples": int(default_metrics["n_samples"]),
        "default_threshold": 0.5,
        "default_accuracy": float(default_metrics["accuracy"]),
        "default_f1": float(default_metrics["f1"]),
        "clean_threshold": float(clean_threshold),
        "clean_threshold_accuracy": float(clean_threshold_metrics["accuracy"]),
        "clean_threshold_f1": float(clean_threshold_metrics["f1"]),
        "oracle_threshold": float(oracle_threshold),
        "oracle_accuracy": float(oracle_metrics["accuracy"]),
        "oracle_f1": float(oracle_metrics["f1"]),
        "roc_auc": float(default_metrics["roc_auc"]) if default_metrics["roc_auc"] is not None else None,
    }


def _format_cell(value) -> str:
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
    out_dir = ensure_dir(args.out_dir)

    clean_predictions = {}
    clean_thresholds = {}
    clean_rows = []
    for method, path in map(_parse_clean, args.clean):
        y_true, scores = _load_predictions(path)
        threshold, metrics = _best_threshold(y_true, scores, args.objective)
        default_metrics = binary_metrics(y_true, scores, threshold=0.5)
        clean_predictions[method] = (y_true, scores, path)
        clean_thresholds[method] = threshold
        clean_rows.append(
            {
                "variant": "clean",
                "method": method,
                "predictions_path": str(path),
                "n_samples": int(metrics["n_samples"]),
                "default_threshold": 0.5,
                "default_accuracy": float(default_metrics["accuracy"]),
                "default_f1": float(default_metrics["f1"]),
                "clean_threshold": float(threshold),
                "clean_threshold_accuracy": float(metrics["accuracy"]),
                "clean_threshold_f1": float(metrics["f1"]),
                "oracle_threshold": float(threshold),
                "oracle_accuracy": float(metrics["accuracy"]),
                "oracle_f1": float(metrics["f1"]),
                "roc_auc": float(default_metrics["roc_auc"]) if default_metrics["roc_auc"] is not None else None,
            }
        )

    rows = list(clean_rows)
    for variant, method, path in map(_parse_variant, args.variant):
        if method not in clean_thresholds:
            raise ValueError(f"No clean threshold provided for method {method!r}")
        y_true, scores = _load_predictions(path)
        rows.append(
            _row(
                variant,
                method,
                path,
                y_true,
                scores,
                clean_thresholds[method],
                args.objective,
            )
        )
    rows = sorted(rows, key=lambda row: (row["method"], row["variant"]))

    with (out_dir / "threshold_calibration.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    columns = [
        "method",
        "variant",
        "default_accuracy",
        "clean_threshold",
        "clean_threshold_accuracy",
        "oracle_threshold",
        "oracle_accuracy",
        "roc_auc",
    ]
    report = [
        "# Threshold Calibration Summary",
        "",
        f"Threshold objective: `{args.objective}`",
        "",
        "The clean-threshold column applies a threshold selected on clean validation predictions. The oracle threshold is diagnostic only because it is selected on the target variant.",
        "",
        _markdown_table(rows, columns),
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(_markdown_table(rows, columns))
    print(f"Wrote threshold calibration summary to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
