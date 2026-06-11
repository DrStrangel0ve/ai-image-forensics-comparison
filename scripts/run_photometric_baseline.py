from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from forensic_compare.datasets import (
    collect_labeled_images,
    discover_layout,
    limit_records,
    stratified_split,
)
from forensic_compare.metrics import binary_metrics
from forensic_compare.photometric import FEATURE_NAMES, extract_features
from forensic_compare.utils import ensure_dir, seed_everything, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a photometric normal-consistency baseline for real-vs-generated detection."
    )
    parser.add_argument("--data-dir", default="data/raw/cifake")
    parser.add_argument("--output-dir", default="runs/photometric")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    return parser.parse_args()


def _records(args: argparse.Namespace):
    layout = discover_layout(args.data_dir)
    if layout.train and layout.test:
        train_records = collect_labeled_images(layout.train)
        test_records = collect_labeled_images(layout.test)
    elif layout.single:
        train_records, test_records = stratified_split(
            collect_labeled_images(layout.single), args.val_fraction, args.seed
        )
    else:
        raise ValueError(f"Unsupported dataset layout: {layout}")
    train_records = limit_records(train_records, args.max_train_samples, args.seed)
    test_records = limit_records(test_records, args.max_test_samples, args.seed + 1)
    return train_records, test_records, layout


def _extract_matrix(records: list[tuple[Path, int, str]], image_size: int, desc: str):
    features = []
    labels = []
    paths = []
    for path, label, _class_name in tqdm(records, desc=desc):
        features.append(extract_features(path, image_size=image_size))
        labels.append(label)
        paths.append(path)
    return np.vstack(features), np.asarray(labels, dtype=int), paths


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    output_dir = ensure_dir(args.output_dir)
    train_records, test_records, layout = _records(args)
    x_train, y_train, _ = _extract_matrix(train_records, args.image_size, "features/train")
    x_test, y_test, test_paths = _extract_matrix(test_records, args.image_size, "features/test")

    classifier = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=args.seed,
                ),
            ),
        ]
    )
    classifier.fit(x_train, y_train)
    scores = classifier.predict_proba(x_test)[:, 1]
    metrics = binary_metrics(y_test, scores)
    metrics.update(
        {
            "method": "photometric_normal_consistency",
            "note": "Single-image shape-from-shading proxy; true calibrated photometric stereo needs multi-light image stacks.",
            "feature_names": FEATURE_NAMES,
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "layout": {
                "train": str(layout.train) if layout.train else None,
                "test": str(layout.test) if layout.test else None,
                "single": str(layout.single) if layout.single else None,
            },
        }
    )
    logreg = classifier.named_steps["logreg"]
    metrics["feature_coefficients"] = {
        name: float(value) for name, value in zip(FEATURE_NAMES, logreg.coef_[0])
    }

    write_json(metrics, output_dir / "metrics.json")
    joblib.dump(classifier, output_dir / "photometric_model.joblib")
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "y_true", "fake_score"])
        writer.writeheader()
        for path, truth, score in zip(test_paths, y_test, scores):
            writer.writerow({"path": str(path), "y_true": int(truth), "fake_score": float(score)})
    print(f"Wrote photometric results to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
