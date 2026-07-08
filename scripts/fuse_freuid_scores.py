from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forensic_compare.freuid import apcer_at_bpcer, freuid_metrics
from forensic_compare.metrics import binary_metrics
from forensic_compare.utils import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid-search score fusion for FREUID validation predictions.")
    parser.add_argument("--predictions", nargs="+", required=True, help="Prediction CSVs with id,y_true,fraud_score.")
    parser.add_argument("--names", nargs="+", default=None, help="Optional names for the prediction CSVs.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--weight-step", type=float, default=0.05)
    parser.add_argument("--normalization", choices=["raw", "minmax", "rank"], nargs="+", default=["raw", "rank"])
    return parser.parse_args()


def _source_name(path: Path) -> str:
    parent = path.parent.name
    return parent if parent else path.stem


def _load_prediction_frames(paths: list[Path], names: list[str] | None) -> tuple[pd.DataFrame, list[str]]:
    if names is not None and len(names) != len(paths):
        raise ValueError("--names must have the same length as --predictions")
    source_names = names or [_source_name(path) for path in paths]
    if len(set(source_names)) != len(source_names):
        raise ValueError(f"Prediction source names must be unique: {source_names}")

    base: pd.DataFrame | None = None
    for source_name, path in zip(source_names, paths):
        frame = pd.read_csv(path)
        required = {"id", "y_true", "fraud_score"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        frame = frame.copy()
        frame["id"] = frame["id"].astype(str)
        if frame["id"].duplicated().any():
            raise ValueError(f"{path} contains duplicate ids")
        keep = frame[["id", "y_true", "fraud_score"]].rename(columns={"fraud_score": source_name})
        keep["y_true"] = keep["y_true"].astype(int)
        keep[source_name] = pd.to_numeric(keep[source_name], errors="raise").astype(float)
        if base is None:
            base = keep
        else:
            base = base.merge(keep, on=["id", "y_true"], how="inner", validate="one_to_one")
    if base is None or base.empty:
        raise ValueError("No prediction rows to fuse")
    return base, source_names


def _normalize(values: np.ndarray, mode: str) -> np.ndarray:
    if mode == "raw":
        return values.astype(float)
    if mode == "minmax":
        result = np.zeros_like(values, dtype=float)
        for col in range(values.shape[1]):
            column = values[:, col]
            span = float(column.max() - column.min())
            result[:, col] = 0.5 if span < 1e-12 else (column - column.min()) / span
        return result
    if mode == "rank":
        return np.column_stack(
            [pd.Series(values[:, col]).rank(method="average", pct=True).to_numpy(dtype=float) for col in range(values.shape[1])]
        )
    raise ValueError(f"Unsupported normalization: {mode}")


def _weight_grid(n_sources: int, step: float) -> list[tuple[float, ...]]:
    if n_sources < 2:
        raise ValueError("Score fusion requires at least two sources")
    if not 0.0 < step <= 1.0:
        raise ValueError("--weight-step must be in (0, 1]")
    units = int(round(1.0 / step))
    if not np.isclose(units * step, 1.0):
        raise ValueError("--weight-step must evenly divide 1.0")
    weights: list[tuple[float, ...]] = []
    for prefix in itertools.product(range(units + 1), repeat=n_sources - 1):
        remaining = units - sum(prefix)
        if remaining < 0:
            continue
        weights.append(tuple([value * step for value in prefix] + [remaining * step]))
    return weights


def _metric_row(y_true: np.ndarray, scores: np.ndarray, normalization: str, weights: tuple[float, ...]) -> dict[str, object]:
    binary = binary_metrics(y_true, scores)
    freuid = freuid_metrics(y_true, scores)
    return {
        "normalization": normalization,
        "weights": list(weights),
        "accuracy": float(binary["accuracy"]),
        "roc_auc": float(binary["roc_auc"]),
        "apcer_at_1pct_bpcer": float(freuid["apcer_at_1pct_bpcer"]),
        "bpcer_at_operating_point": float(freuid["bpcer_at_operating_point"]),
        "threshold_at_1pct_bpcer": float(freuid["threshold_at_1pct_bpcer"]),
        "audet_proxy": float(freuid["audet_proxy"]),
        "brier_score": float(binary["brier_score"]),
        "expected_calibration_error": float(binary["expected_calibration_error"]),
    }


def fuse_scores(
    prediction_paths: list[Path],
    output_dir: Path,
    names: list[str] | None = None,
    weight_step: float = 0.05,
    normalizations: list[str] | None = None,
) -> dict[str, object]:
    normalizations = normalizations or ["raw", "rank"]
    output_dir = ensure_dir(output_dir)
    frame, source_names = _load_prediction_frames(prediction_paths, names)
    y_true = frame["y_true"].to_numpy(dtype=int)
    score_matrix = frame[source_names].to_numpy(dtype=float)
    rows: list[dict[str, object]] = []
    for normalization in normalizations:
        normalized = _normalize(score_matrix, normalization)
        for weights in _weight_grid(len(source_names), weight_step):
            scores = normalized @ np.asarray(weights, dtype=float)
            rows.append(_metric_row(y_true, scores, normalization, weights))
    rows = sorted(rows, key=lambda row: (row["apcer_at_1pct_bpcer"], row["audet_proxy"], -row["roc_auc"]))
    best = rows[0]

    with (output_dir / "fusion_grid.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "normalization",
            "weights",
            "accuracy",
            "roc_auc",
            "apcer_at_1pct_bpcer",
            "bpcer_at_operating_point",
            "threshold_at_1pct_bpcer",
            "audet_proxy",
            "brier_score",
            "expected_calibration_error",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "weights": ",".join(f"{value:.4f}" for value in row["weights"])})

    normalized = _normalize(score_matrix, str(best["normalization"]))
    best_scores = normalized @ np.asarray(best["weights"], dtype=float)
    point = apcer_at_bpcer(y_true, best_scores)
    fused = pd.DataFrame(
        {
            "id": frame["id"],
            "y_true": y_true,
            "fraud_score": best_scores,
            "label": (best_scores >= point.threshold).astype(int),
        }
    )
    fused.to_csv(output_dir / "fused_predictions.csv", index=False)
    summary = {
        "prediction_paths": [str(path) for path in prediction_paths],
        "source_names": source_names,
        "weight_step": weight_step,
        "normalizations": normalizations,
        "n_rows": int(len(frame)),
        "best": best,
        "top_rows": rows[:10],
    }
    write_json(summary, output_dir / "fusion_summary.json")
    return summary


def main() -> None:
    args = parse_args()
    summary = fuse_scores(
        prediction_paths=[Path(path) for path in args.predictions],
        names=args.names,
        output_dir=Path(args.output_dir),
        weight_step=args.weight_step,
        normalizations=list(args.normalization),
    )
    best = summary["best"]
    print(Path(args.output_dir).resolve())
    print(
        "best="
        f"normalization={best['normalization']} "
        f"weights={best['weights']} "
        f"apcer_at_1pct_bpcer={best['apcer_at_1pct_bpcer']:.6f} "
        f"audet_proxy={best['audet_proxy']:.6f} "
        f"roc_auc={best['roc_auc']:.6f}"
    )


if __name__ == "__main__":
    main()
