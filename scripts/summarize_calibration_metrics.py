from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from forensic_compare.metrics import binary_metrics, calibration_bins
from forensic_compare.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize probability-calibration metrics from saved prediction CSVs."
    )
    parser.add_argument(
        "--predictions",
        action="append",
        required=True,
        help=(
            "Prediction CSV in GROUP:METHOD=PATH form. GROUP is optional; "
            "METHOD=PATH is treated as group 'default'. Repeat for each run."
        ),
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-bins", type=int, default=10)
    return parser.parse_args()


def _parse_prediction(value: str) -> tuple[str, str, Path]:
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


def _load_predictions(path: Path) -> tuple[pd.Series, pd.Series]:
    frame = pd.read_csv(path)
    required = {"y_true", "fake_score"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return frame["y_true"].astype(int), frame["fake_score"].astype(float)


def _row(group: str, method: str, path: Path, n_bins: int) -> tuple[dict, list[dict]]:
    y_true, scores = _load_predictions(path)
    metrics = binary_metrics(y_true.to_numpy(), scores.to_numpy(), n_bins=n_bins)
    bins = calibration_bins(y_true.to_numpy(), scores.to_numpy(), n_bins=n_bins)
    predicted = scores >= 0.5
    row = {
        "group": group,
        "method": method,
        "predictions_path": str(path),
        "n_samples": int(metrics["n_samples"]),
        "positive_rate": float(y_true.mean()),
        "predicted_positive_rate": float(predicted.mean()),
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std(ddof=0)),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "roc_auc": metrics["roc_auc"],
        "brier_score": float(metrics["brier_score"]),
        "expected_calibration_error": float(metrics["expected_calibration_error"]),
        "maximum_calibration_error": float(metrics["maximum_calibration_error"]),
    }
    bin_rows = [
        {
            "group": group,
            "method": method,
            "predictions_path": str(path),
            **bin_row,
        }
        for bin_row in bins
    ]
    return row, bin_rows


def _summarize_by_method(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby("method", as_index=False)
        .agg(
            n_runs=("group", "count"),
            mean_accuracy=("accuracy", "mean"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_brier_score=("brier_score", "mean"),
            mean_expected_calibration_error=("expected_calibration_error", "mean"),
            mean_maximum_calibration_error=("maximum_calibration_error", "mean"),
            mean_score=("score_mean", "mean"),
            mean_positive_rate=("positive_rate", "mean"),
            mean_predicted_positive_rate=("predicted_positive_rate", "mean"),
        )
        .sort_values(["mean_expected_calibration_error", "mean_brier_score"], ascending=[True, True])
    )


def _reliability_by_method(bins_frame: pd.DataFrame) -> pd.DataFrame:
    nonempty = bins_frame[bins_frame["count"] > 0].copy()
    nonempty["confidence_weighted"] = nonempty["confidence"] * nonempty["count"]
    nonempty["accuracy_weighted"] = nonempty["accuracy"] * nonempty["count"]
    grouped = (
        nonempty.groupby(["method", "bin", "bin_lower", "bin_upper"], as_index=False)
        .agg(
            count=("count", "sum"),
            confidence_weighted=("confidence_weighted", "sum"),
            accuracy_weighted=("accuracy_weighted", "sum"),
        )
        .sort_values(["method", "bin"])
    )
    grouped["confidence"] = grouped["confidence_weighted"] / grouped["count"]
    grouped["accuracy"] = grouped["accuracy_weighted"] / grouped["count"]
    grouped["abs_gap"] = (grouped["confidence"] - grouped["accuracy"]).abs()
    return grouped[
        [
            "method",
            "bin",
            "bin_lower",
            "bin_upper",
            "count",
            "confidence",
            "accuracy",
            "abs_gap",
        ]
    ]


def _write_reliability_plot(reliability_frame: pd.DataFrame, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5), dpi=160)
    ax.plot([0.0, 1.0], [0.0, 1.0], color="#666666", linestyle="--", linewidth=1.0)
    for method, group in reliability_frame.groupby("method"):
        ax.plot(
            group["confidence"],
            group["accuracy"],
            marker="o",
            linewidth=1.5,
            markersize=4,
            label=method,
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Mean predicted fake score")
    ax.set_ylabel("Observed fake rate")
    ax.set_title("Reliability by method")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _format_cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
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

    rows = []
    bin_rows = []
    for group, method, path in map(_parse_prediction, args.predictions):
        row, bins = _row(group, method, path, args.n_bins)
        rows.append(row)
        bin_rows.extend(bins)

    rows = sorted(rows, key=lambda row: (row["method"], row["group"]))
    metrics_frame = pd.DataFrame(rows)
    summary_frame = _summarize_by_method(metrics_frame)
    bins_frame = pd.DataFrame(bin_rows)
    reliability_frame = _reliability_by_method(bins_frame)

    metrics_frame.to_csv(out_dir / "calibration_metrics.csv", index=False)
    summary_frame.to_csv(out_dir / "calibration_summary_by_method.csv", index=False)
    bins_frame.to_csv(out_dir / "calibration_bins.csv", index=False)
    reliability_frame.to_csv(out_dir / "calibration_reliability_by_method.csv", index=False)
    _write_reliability_plot(reliability_frame, out_dir / "reliability_by_method.png")

    columns = [
        "method",
        "n_runs",
        "mean_accuracy",
        "mean_roc_auc",
        "mean_brier_score",
        "mean_expected_calibration_error",
        "mean_maximum_calibration_error",
        "mean_predicted_positive_rate",
    ]
    detail_columns = [
        "group",
        "method",
        "accuracy",
        "roc_auc",
        "brier_score",
        "expected_calibration_error",
        "maximum_calibration_error",
        "predicted_positive_rate",
    ]
    report = [
        "# Calibration Metrics Summary",
        "",
        f"Reliability bins: `{args.n_bins}` equal-width bins over fake-class score.",
        "",
        "Lower Brier score and expected calibration error are better. AUC still measures ranking; calibration metrics measure whether scores behave like probabilities.",
        "",
        "## Mean by Method",
        "",
        _markdown_table(summary_frame.to_dict("records"), columns),
        "",
        "## Reliability Curve",
        "",
        "![Reliability by method](reliability_by_method.png)",
        "",
        "## Runs",
        "",
        _markdown_table(metrics_frame.to_dict("records"), detail_columns),
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")

    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["artifact", "path"])
        for artifact in [
            "calibration_metrics.csv",
            "calibration_summary_by_method.csv",
        "calibration_bins.csv",
        "calibration_reliability_by_method.csv",
        "reliability_by_method.png",
        "report.md",
        ]:
            writer.writerow([artifact, str((out_dir / artifact).resolve())])

    print(_markdown_table(summary_frame.to_dict("records"), columns))
    print(f"Wrote calibration summary to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
