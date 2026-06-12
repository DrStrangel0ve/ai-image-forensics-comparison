from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from forensic_compare.metrics import binary_metrics
from forensic_compare.utils import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train score-level fusion from aligned prediction CSVs."
    )
    parser.add_argument(
        "--train",
        action="append",
        required=True,
        help="Training prediction in METHOD=PATH form. Repeat for each base model.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        help="Evaluation prediction in VARIANT:METHOD=PATH form. Repeat for each method/variant.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def _parse_train(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Training predictions must be METHOD=PATH, got {value!r}")
    method, path = value.split("=", 1)
    if not method or not path:
        raise ValueError(f"Training predictions must be METHOD=PATH, got {value!r}")
    return method, Path(path)


def _parse_variant(value: str) -> tuple[str, str, Path]:
    if "=" not in value:
        raise ValueError(f"Variant predictions must be VARIANT:METHOD=PATH, got {value!r}")
    left, path = value.split("=", 1)
    if ":" not in left:
        raise ValueError(f"Variant predictions must be VARIANT:METHOD=PATH, got {value!r}")
    variant, method = left.split(":", 1)
    if not variant or not method or not path:
        raise ValueError(f"Variant predictions must be VARIANT:METHOD=PATH, got {value!r}")
    return variant, method, Path(path)


def _norm(path: str | Path) -> str:
    return str(Path(path).resolve()).replace("\\", "/").lower()


def _prediction_frame(method: str, path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"path", "y_true", "fake_score"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing prediction columns: {sorted(missing)}")
    return pd.DataFrame(
        {
            "path": frame["path"],
            "path_key": frame["path"].map(_norm),
            "y_true": frame["y_true"].astype(int),
            method: frame["fake_score"].astype(float),
        }
    )


def _aligned_matrix(named_paths: list[tuple[str, Path]]) -> tuple[pd.DataFrame, list[str]]:
    methods = [method for method, _path in named_paths]
    if len(set(methods)) != len(methods):
        raise ValueError(f"Duplicate method names are not allowed: {methods}")
    merged: pd.DataFrame | None = None
    for method, path in named_paths:
        frame = _prediction_frame(method, path)
        if merged is None:
            merged = frame
            continue
        merged = merged.merge(
            frame[["path_key", "y_true", method]],
            on="path_key",
            suffixes=("", "_next"),
        )
        mismatches = merged[merged["y_true"] != merged["y_true_next"]]
        if not mismatches.empty:
            raise ValueError(f"Label mismatch while joining {method}: {len(mismatches)} rows")
        merged = merged.drop(columns=["y_true_next"])
    if merged is None or merged.empty:
        raise ValueError("No prediction rows were aligned")
    return merged, methods


def _variant_groups(values: list[str]) -> dict[str, list[tuple[str, Path]]]:
    grouped: dict[str, list[tuple[str, Path]]] = {}
    for value in values:
        variant, method, path = _parse_variant(value)
        grouped.setdefault(variant, []).append((method, path))
    return grouped


def _classifier(seed: int):
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "logreg",
                LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed),
            ),
        ]
    )


def _metrics_row(
    variant: str,
    methods: list[str],
    frame: pd.DataFrame,
    classifier,
    threshold: float,
) -> tuple[dict, list[dict]]:
    x = frame[methods].to_numpy(dtype=float)
    y_true = frame["y_true"].to_numpy(dtype=int)
    scores = classifier.predict_proba(x)[:, 1]
    metrics = binary_metrics(y_true, scores, threshold=threshold)
    metrics.update(
        {
            "variant": variant,
            "method": "score_fusion",
            "base_methods": methods,
            "n_samples": int(len(y_true)),
            "threshold": float(threshold),
        }
    )
    rows = [
        {"path": path, "y_true": int(truth), "fake_score": float(score)}
        for path, truth, score in zip(frame["path"], y_true, scores)
    ]
    return metrics, rows


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    train_frame, methods = _aligned_matrix(list(map(_parse_train, args.train)))
    x_train = train_frame[methods].to_numpy(dtype=float)
    y_train = train_frame["y_true"].to_numpy(dtype=int)
    classifier = _classifier(args.seed)
    classifier.fit(x_train, y_train)

    all_metrics = []
    train_metrics, train_rows = _metrics_row(
        "train", methods, train_frame, classifier, args.threshold
    )
    all_metrics.append(train_metrics)
    train_dir = ensure_dir(out_dir / "train")
    with (train_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "y_true", "fake_score"])
        writer.writeheader()
        writer.writerows(train_rows)

    variant_groups = _variant_groups(args.variant)
    for variant, named_paths in sorted(variant_groups.items()):
        variant_frame, variant_methods = _aligned_matrix(named_paths)
        if variant_methods != methods:
            raise ValueError(
                f"Variant {variant!r} has methods {variant_methods}, expected {methods}"
            )
        metrics, rows = _metrics_row(variant, methods, variant_frame, classifier, args.threshold)
        all_metrics.append(metrics)
        variant_dir = ensure_dir(out_dir / variant)
        with (variant_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["path", "y_true", "fake_score"])
            writer.writeheader()
            writer.writerows(rows)

    write_json(
        {
            "method": "score_fusion",
            "base_methods": methods,
            "seed": int(args.seed),
            "threshold": float(args.threshold),
            "n_train": int(len(y_train)),
            "metrics": all_metrics,
        },
        out_dir / "metrics.json",
    )
    joblib.dump(classifier, out_dir / "score_fusion_model.joblib")
    summary_rows = [
        {
            "variant": metrics["variant"],
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
            "n_samples": metrics["n_samples"],
        }
        for metrics in all_metrics
    ]
    pd.DataFrame(summary_rows).to_csv(out_dir / "summary.csv", index=False)
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"Wrote score-fusion results to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
