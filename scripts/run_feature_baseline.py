from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from forensic_compare.conventional import extract_feature_set, feature_names
from forensic_compare.datasets import (
    collect_labeled_images,
    discover_layout,
    limit_records,
    stratified_split,
)
from forensic_compare.metrics import binary_metrics
from forensic_compare.utils import ensure_dir, seed_everything, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a conventional feature baseline for real-vs-generated detection."
    )
    parser.add_argument("--data-dir", default="data/raw/cifake")
    parser.add_argument("--output-dir", default="runs/features")
    parser.add_argument(
        "--feature-set",
        choices=[
            "photometric",
            "noise",
            "noise_v2",
            "noise_v3",
            "combined",
            "combined_v2",
            "combined_v3",
        ],
        default="combined",
    )
    parser.add_argument(
        "--classifier",
        choices=["logistic_regression", "random_forest", "hist_gradient_boosting"],
        default="logistic_regression",
    )
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--skip-errors", action="store_true", help="Skip unreadable/corrupt images.")
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


def _extract_matrix(
    records: list[tuple[Path, int, str]],
    image_size: int,
    feature_set: str,
    desc: str,
    skip_errors: bool,
):
    features = []
    labels = []
    paths = []
    skipped = []
    for path, label, _class_name in tqdm(records, desc=desc):
        try:
            features.append(extract_feature_set(path, image_size=image_size, feature_set=feature_set))
            labels.append(label)
            paths.append(path)
        except Exception as exc:
            if not skip_errors:
                raise
            skipped.append({"path": str(path), "error": repr(exc)})
    if not features:
        raise ValueError(f"No usable feature rows for {desc}")
    return np.vstack(features), np.asarray(labels, dtype=int), paths, skipped


def _classifier(name: str, seed: int):
    if name == "logistic_regression":
        return Pipeline(
            steps=[
                ("scale", StandardScaler()),
                (
                    "logreg",
                    LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed),
                ),
            ]
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=seed)
    raise ValueError(f"Unsupported classifier: {name}")


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    output_dir = ensure_dir(args.output_dir)
    train_records, test_records, layout = _records(args)
    x_train, y_train, _, skipped_train = _extract_matrix(
        train_records, args.image_size, args.feature_set, "features/train", args.skip_errors
    )
    x_test, y_test, test_paths, skipped_test = _extract_matrix(
        test_records, args.image_size, args.feature_set, "features/test", args.skip_errors
    )

    classifier = _classifier(args.classifier, args.seed)
    classifier.fit(x_train, y_train)
    if hasattr(classifier, "predict_proba"):
        scores = classifier.predict_proba(x_test)[:, 1]
    else:
        raw_scores = classifier.decision_function(x_test)
        scores = 1.0 / (1.0 + np.exp(-raw_scores))
    metrics = binary_metrics(y_test, scores)
    names = feature_names(args.feature_set)
    metrics.update(
        {
            "method": f"feature_{args.feature_set}_{args.classifier}",
            "feature_set": args.feature_set,
            "classifier": args.classifier,
            "feature_names": names,
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "n_skipped_train": len(skipped_train),
            "n_skipped_test": len(skipped_test),
            "layout": {
                "train": str(layout.train) if layout.train else None,
                "test": str(layout.test) if layout.test else None,
                "single": str(layout.single) if layout.single else None,
            },
        }
    )
    write_json(metrics, output_dir / "metrics.json")
    if skipped_train or skipped_test:
        write_json({"train": skipped_train, "test": skipped_test}, output_dir / "skipped.json")
    joblib.dump(classifier, output_dir / "feature_model.joblib")
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "y_true", "fake_score"])
        writer.writeheader()
        for path, truth, score in zip(test_paths, y_test, scores):
            writer.writerow({"path": str(path), "y_true": int(truth), "fake_score": float(score)})
    print(f"Wrote feature baseline results to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
