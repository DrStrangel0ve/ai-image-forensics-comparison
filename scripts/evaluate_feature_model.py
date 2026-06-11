from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
from tqdm import tqdm

from forensic_compare.catalog import load_catalog
from forensic_compare.conventional import extract_feature_set
from forensic_compare.datasets import collect_labeled_images, discover_layout, stratified_split
from forensic_compare.metrics import binary_metrics
from forensic_compare.utils import ensure_dir, read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved conventional feature model.")
    parser.add_argument("--model-dir", required=True, help="Directory containing feature_model.joblib and metrics.json.")
    parser.add_argument("--target-key", default=None, help="Dataset key from configs/datasets.json.")
    parser.add_argument("--target-dir", default=None, help="Target dataset folder. Overrides --target-key.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--target-split", choices=["all", "test"], default="all")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--skip-errors", action="store_true")
    return parser.parse_args()


def _target_dir(args: argparse.Namespace) -> Path:
    if args.target_dir:
        return Path(args.target_dir)
    if not args.target_key:
        raise SystemExit("Provide --target-key or --target-dir")
    return load_catalog()[args.target_key].local_dir


def _target_records(target_dir: Path, args: argparse.Namespace):
    layout = discover_layout(target_dir)
    if args.target_split == "test" and layout.test:
        return collect_labeled_images(layout.test), layout
    if layout.single:
        records = collect_labeled_images(layout.single)
        if args.target_split == "test":
            _train, test = stratified_split(records, args.val_fraction, args.seed)
            return test, layout
        return records, layout
    if layout.train and layout.test:
        if args.target_split == "all":
            return collect_labeled_images(layout.train) + collect_labeled_images(layout.test), layout
        return collect_labeled_images(layout.test), layout
    raise ValueError(f"Unsupported dataset layout: {layout}")


def _extract_matrix(records, image_size: int, feature_set: str, skip_errors: bool):
    features = []
    labels = []
    paths = []
    skipped = []
    for path, label, _class_name in tqdm(records, desc="features/target"):
        try:
            features.append(extract_feature_set(path, image_size=image_size, feature_set=feature_set))
            labels.append(label)
            paths.append(path)
        except Exception as exc:
            if not skip_errors:
                raise
            skipped.append({"path": str(path), "error": repr(exc)})
    if not features:
        raise ValueError("No usable target feature rows")
    return np.vstack(features), np.asarray(labels, dtype=int), paths, skipped


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    output_dir = ensure_dir(args.output_dir)
    source_metrics = read_json(model_dir / "metrics.json")
    feature_set = source_metrics["feature_set"]
    model = joblib.load(model_dir / "feature_model.joblib")

    target_records, layout = _target_records(_target_dir(args), args)
    x_target, y_target, paths, skipped = _extract_matrix(
        target_records, args.image_size, feature_set, args.skip_errors
    )
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(x_target)[:, 1]
    else:
        raw_scores = model.decision_function(x_target)
        scores = 1.0 / (1.0 + np.exp(-raw_scores))

    metrics = binary_metrics(y_target, scores)
    metrics.update(
        {
            "method": f"cross_feature_{feature_set}",
            "source_model_dir": str(model_dir),
            "target_dir": str(_target_dir(args)),
            "target_split": args.target_split,
            "feature_set": feature_set,
            "classifier": source_metrics.get("classifier"),
            "image_size": args.image_size,
            "n_target": int(len(y_target)),
            "n_skipped_target": len(skipped),
            "layout": {
                "train": str(layout.train) if layout.train else None,
                "test": str(layout.test) if layout.test else None,
                "single": str(layout.single) if layout.single else None,
            },
        }
    )
    write_json(metrics, output_dir / "metrics.json")
    if skipped:
        write_json(skipped, output_dir / "skipped.json")
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "y_true", "fake_score"])
        writer.writeheader()
        for path, truth, score in zip(paths, y_target, scores):
            writer.writerow({"path": str(path), "y_true": int(truth), "fake_score": float(score)})
    print(f"Wrote cross-dataset feature evaluation to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
