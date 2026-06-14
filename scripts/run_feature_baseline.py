from __future__ import annotations

import argparse
import csv
import sys
from functools import partial
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
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
from forensic_compare.transforms import ROBUSTNESS_VARIANTS, apply_robustness_variant
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
            "noise_v4",
            "reconstruction_lite",
            "combined",
            "combined_v2",
            "combined_v3",
            "combined_v4",
        ],
        default="combined",
    )
    parser.add_argument(
        "--classifier",
        choices=["logistic_regression", "random_forest", "hist_gradient_boosting"],
        default="logistic_regression",
    )
    parser.add_argument(
        "--select-k",
        type=int,
        default=0,
        help="Optionally keep the top K handcrafted features before fitting.",
    )
    parser.add_argument(
        "--selection-score-func",
        choices=["f_classif", "mutual_info"],
        default="f_classif",
        help="Feature-selection scoring function used when --select-k is positive.",
    )
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--skip-errors", action="store_true", help="Skip unreadable/corrupt images.")
    parser.add_argument(
        "--train-augment-variants",
        nargs="+",
        choices=ROBUSTNESS_VARIANTS,
        default=[],
        help="Add deterministic transformed copies of each training image before fitting.",
    )
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
    augment_variants: list[str] | tuple[str, ...] | None = None,
):
    augment_variants = list(augment_variants or [])
    features = []
    labels = []
    paths = []
    skipped = []
    for path, label, _class_name in tqdm(records, desc=desc):
        try:
            record_features = [
                extract_feature_set(path, image_size=image_size, feature_set=feature_set)
            ]
            for variant in augment_variants:
                record_features.append(
                    _extract_augmented_feature(path, image_size, feature_set, variant)
                )
        except Exception as exc:
            if not skip_errors:
                raise
            skipped.append({"path": str(path), "error": repr(exc)})
            continue
        features.extend(record_features)
        labels.extend([label] * len(record_features))
        paths.extend([path] * len(record_features))
    if not features:
        raise ValueError(f"No usable feature rows for {desc}")
    return np.vstack(features), np.asarray(labels, dtype=int), paths, skipped


def _extract_augmented_feature(
    path: Path,
    image_size: int,
    feature_set: str,
    variant: str,
) -> np.ndarray:
    with Image.open(path) as image:
        transformed = apply_robustness_variant(image, variant)
    buffer = BytesIO()
    transformed.save(buffer, format="JPEG", quality=95)
    buffer.seek(0)
    return extract_feature_set(buffer, image_size=image_size, feature_set=feature_set)


def _selection_score_func(name: str, seed: int):
    if name == "f_classif":
        return f_classif
    if name == "mutual_info":
        return partial(mutual_info_classif, random_state=seed)
    raise ValueError(f"Unsupported selection score function: {name}")


def _base_classifier(name: str, seed: int):
    if name == "logistic_regression":
        return LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)
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


def _classifier(
    name: str,
    seed: int,
    select_k: int = 0,
    n_features: int | None = None,
    selection_score_func: str = "f_classif",
):
    if select_k < 0:
        raise ValueError("--select-k must be non-negative")
    base = _base_classifier(name, seed)
    needs_scaling = name == "logistic_regression"
    steps = []
    if needs_scaling:
        steps.append(("scale", StandardScaler()))
    if select_k:
        if n_features is None:
            raise ValueError("n_features is required when --select-k is positive")
        steps.append(
            (
                "select",
                SelectKBest(
                    score_func=_selection_score_func(selection_score_func, seed),
                    k=min(select_k, n_features),
                ),
            )
        )
    if steps:
        steps.append(("model", base))
        return Pipeline(steps=steps)
    return base


def _selected_feature_names(classifier, names: list[str]) -> list[str]:
    selector = classifier.named_steps.get("select") if isinstance(classifier, Pipeline) else None
    if selector is None:
        return []
    mask = selector.get_support()
    return [name for name, keep in zip(names, mask) if keep]


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    output_dir = ensure_dir(args.output_dir)
    train_records, test_records, layout = _records(args)
    x_train, y_train, _, skipped_train = _extract_matrix(
        train_records,
        args.image_size,
        args.feature_set,
        "features/train",
        args.skip_errors,
        args.train_augment_variants,
    )
    x_test, y_test, test_paths, skipped_test = _extract_matrix(
        test_records,
        args.image_size,
        args.feature_set,
        "features/test",
        args.skip_errors,
    )

    classifier = _classifier(
        args.classifier,
        args.seed,
        select_k=args.select_k,
        n_features=x_train.shape[1],
        selection_score_func=args.selection_score_func,
    )
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
            "method": (
                f"feature_{args.feature_set}_{args.classifier}"
                + (f"_selectk{min(args.select_k, x_train.shape[1])}" if args.select_k else "")
            ),
            "feature_set": args.feature_set,
            "classifier": args.classifier,
            "feature_names": names,
            "select_k": int(args.select_k),
            "effective_select_k": int(min(args.select_k, x_train.shape[1])) if args.select_k else 0,
            "selection_score_func": args.selection_score_func if args.select_k else None,
            "selected_feature_names": _selected_feature_names(classifier, names),
            "n_train": int(len(y_train)),
            "n_train_original": int(len(train_records) - len(skipped_train)),
            "train_augment_variants": list(args.train_augment_variants),
            "n_train_augment_variants": len(args.train_augment_variants),
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
