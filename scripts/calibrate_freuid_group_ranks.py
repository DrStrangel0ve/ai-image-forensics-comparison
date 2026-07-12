from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forensic_compare.freuid import freuid_metrics
from forensic_compare.metrics import binary_metrics
from forensic_compare.utils import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate FREUID scores by document-type percentile rank.")
    parser.add_argument("--val-predictions", required=True)
    parser.add_argument("--val-metadata", required=True)
    parser.add_argument("--test-diagnostics", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--val-group-column", default="type")
    parser.add_argument("--test-group-column", default="predicted_type_index")
    parser.add_argument("--val-score-column", default="fraud_score")
    parser.add_argument("--test-score-column", default="fraud_score")
    parser.add_argument("--global-weights", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--selection-apcer-weight", type=float, default=0.25)
    return parser.parse_args()


def _ranks(frame: pd.DataFrame, score_column: str, group_column: str) -> tuple[pd.Series, pd.Series]:
    global_rank = frame[score_column].rank(method="average", pct=True)
    group_rank = frame.groupby(group_column, sort=False)[score_column].rank(method="average", pct=True)
    return global_rank.astype(float), group_rank.astype(float)


def _weight_name(weight: float) -> str:
    return f"{weight:.2f}".replace(".", "p")


def _write_predictions(path: Path, ids: pd.Series, scores: pd.Series) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "fraud_score"])
        writer.writeheader()
        for image_id, score in zip(ids.astype(str), scores.astype(float)):
            writer.writerow({"id": image_id, "fraud_score": float(score)})


def run(args: argparse.Namespace) -> dict[str, object]:
    output_dir = ensure_dir(args.output_dir)
    val_predictions = pd.read_csv(args.val_predictions)
    val_metadata = pd.read_csv(args.val_metadata)
    val_predictions["id"] = val_predictions["id"].astype(str)
    val_metadata["id"] = val_metadata["id"].astype(str)
    if args.val_group_column not in val_metadata.columns:
        raise ValueError(f"Validation metadata is missing {args.val_group_column!r}")
    val = val_predictions.merge(
        val_metadata[["id", args.val_group_column]],
        on="id",
        how="inner",
        validate="one_to_one",
    )
    if len(val) != len(val_predictions):
        raise ValueError("Validation predictions and metadata do not have identical id coverage")

    test = pd.read_csv(args.test_diagnostics)
    test["id"] = test["id"].astype(str)
    if args.test_group_column not in test.columns:
        raise ValueError(f"Test diagnostics are missing {args.test_group_column!r}")
    if args.test_score_column not in test.columns:
        raise ValueError(f"Test diagnostics are missing {args.test_score_column!r}")

    val_global, val_group = _ranks(val, args.val_score_column, args.val_group_column)
    test_global, test_group = _ranks(test, args.test_score_column, args.test_group_column)
    rows = []
    output_paths = {}
    for weight in args.global_weights:
        if not 0.0 <= weight <= 1.0:
            raise ValueError("--global-weights values must be in [0, 1]")
        val_scores = weight * val_global + (1.0 - weight) * val_group
        metrics = binary_metrics(val["y_true"].astype(int), val_scores)
        metrics.update(freuid_metrics(val["y_true"].astype(int), val_scores))
        metrics["global_weight"] = float(weight)
        metrics["selection_objective"] = float(
            metrics["audet_proxy"] + args.selection_apcer_weight * metrics["apcer_at_1pct_bpcer"]
        )
        rows.append(metrics)

        test_scores = weight * test_global + (1.0 - weight) * test_group
        path = output_dir / f"test_predictions_global_{_weight_name(weight)}.csv"
        _write_predictions(path, test["id"], test_scores)
        output_paths[str(weight)] = str(path)

    rows.sort(key=lambda row: (row["selection_objective"], row["apcer_at_1pct_bpcer"], row["audet_proxy"]))
    pd.DataFrame(rows).to_csv(output_dir / "validation_grid.csv", index=False)
    group_summary = (
        test.assign(
            global_rank=test_global,
            group_rank=test_group,
        )
        .groupby(args.test_group_column, sort=True)
        .agg(
            n=("id", "size"),
            raw_score_mean=(args.test_score_column, "mean"),
            global_rank_mean=("global_rank", "mean"),
            group_rank_mean=("group_rank", "mean"),
        )
        .reset_index()
    )
    group_summary.to_csv(output_dir / "test_group_summary.csv", index=False)
    summary = {
        "val_predictions": str(args.val_predictions),
        "val_metadata": str(args.val_metadata),
        "test_diagnostics": str(args.test_diagnostics),
        "n_val": int(len(val)),
        "n_test": int(len(test)),
        "val_group_column": args.val_group_column,
        "test_group_column": args.test_group_column,
        "best": rows[0],
        "rows": rows,
        "test_prediction_paths": output_paths,
    }
    write_json(summary, output_dir / "summary.json")
    return summary


def main() -> None:
    summary = run(parse_args())
    best = summary["best"]
    print(Path(str(summary["test_prediction_paths"][str(best["global_weight"])] )).resolve())
    print(
        f"best_global_weight={best['global_weight']} "
        f"objective={best['selection_objective']:.6f} "
        f"audet={best['audet_proxy']:.6f} apcer={best['apcer_at_1pct_bpcer']:.6f}"
    )


if __name__ == "__main__":
    main()
