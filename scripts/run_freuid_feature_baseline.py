from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forensic_compare.conventional import extract_feature_set, feature_names
from forensic_compare.datasets import stable_path_score
from forensic_compare.freuid import apcer_at_bpcer, freuid_competition_path, freuid_metrics
from forensic_compare.metrics import binary_metrics
from forensic_compare.utils import ensure_dir, seed_everything, write_json


FEATURE_SETS = [
    "photometric",
    "file_metadata",
    "noise",
    "noise_v2",
    "noise_v3",
    "noise_v4",
    "reconstruction_lite",
    "reconstruction_v2",
    "combined",
    "combined_v2",
    "combined_v3",
    "combined_v4",
    "combined_v5",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a conventional FREUID baseline from CSV metadata.")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--test-csv", default=None, help="Optional unlabeled/sample-submission CSV to score after training.")
    parser.add_argument("--image-root", default="data/raw/freuid_2026/images")
    parser.add_argument("--output-dir", default="runs/freuid_feature_baseline")
    parser.add_argument("--test-predictions-out", default=None, help="Optional output path for test predictions.")
    parser.add_argument("--feature-set", choices=FEATURE_SETS, default="combined_v3")
    parser.add_argument(
        "--classifier",
        choices=["logistic_regression", "random_forest", "hist_gradient_boosting"],
        default="logistic_regression",
    )
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument(
        "--limit-balance-columns",
        nargs="+",
        default=[],
        help="Optional columns used to balance --max-*-samples, e.g. type label.",
    )
    parser.add_argument(
        "--feature-cache-dir",
        default=None,
        help="Optional directory for cached per-image feature vectors keyed by path/stat/feature-set/image-size.",
    )
    parser.add_argument("--skip-errors", action="store_true")
    return parser.parse_args()


def _resolve_image_path(image_root: Path, image_path: object) -> Path:
    raw = str(image_path)
    candidates = [
        image_root / freuid_competition_path(raw),
        image_root / raw.replace("\\", "/"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find FREUID image for {raw}; tried {candidates}")


def _limit_frame(frame: pd.DataFrame, max_samples: int, seed: int, balance_columns: list[str] | None = None) -> pd.DataFrame:
    if max_samples <= 0 or max_samples >= len(frame):
        return frame
    balance_columns = list(balance_columns or [])
    scored = frame.copy()
    scored["_score"] = scored["id"].astype(str).map(lambda value: stable_path_score(value, seed))
    if not balance_columns:
        balance_columns = ["label"]
    missing = [column for column in balance_columns if column not in scored.columns]
    if missing:
        raise ValueError(f"Balance columns are missing from FREUID metadata: {missing}")
    groups = [group.sort_values(["_score", "id"], kind="mergesort") for _key, group in scored.groupby(balance_columns, sort=True)]
    selected_indices: list[int] = []
    cursor = 0
    while len(selected_indices) < max_samples:
        progressed = False
        for group in groups:
            if cursor < len(group):
                selected_indices.append(int(group.index[cursor]))
                progressed = True
                if len(selected_indices) >= max_samples:
                    break
        if not progressed:
            break
        cursor += 1
    return scored.loc[selected_indices].drop(columns=["_score"]).reset_index(drop=True)


def _limit_unlabeled_frame(frame: pd.DataFrame, max_samples: int, seed: int) -> pd.DataFrame:
    if max_samples <= 0 or max_samples >= len(frame):
        return frame.reset_index(drop=True)
    ordered = frame.copy()
    ordered["_score"] = ordered["id"].astype(str).map(lambda value: stable_path_score(value, seed))
    return ordered.sort_values(["_score", "id"], kind="mergesort").head(max_samples).drop(columns=["_score"]).reset_index(drop=True)


def _public_frame_from_ids(frame: pd.DataFrame) -> pd.DataFrame:
    if "id" not in frame.columns:
        raise ValueError("Unlabeled FREUID test CSV must contain an id column")
    frame = frame.copy()
    frame["id"] = frame["id"].astype(str)
    if "image_path" not in frame.columns:
        frame["image_path"] = frame["id"].map(lambda value: f"public_test/{value}.jpeg")
    return frame


def _cache_path(cache_dir: Path, local_path: Path, feature_set: str, image_size: int) -> Path:
    stat = local_path.stat()
    key = "|".join(
        [
            str(local_path.resolve()),
            str(stat.st_size),
            str(stat.st_mtime_ns),
            feature_set,
            str(image_size),
        ]
    )
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return cache_dir / feature_set / str(image_size) / f"{digest}.npy"


def _extract_or_load_feature(
    local_path: Path,
    image_size: int,
    feature_set: str,
    cache_dir: Path | None,
) -> tuple[np.ndarray, str]:
    if cache_dir is not None:
        path = _cache_path(cache_dir, local_path, feature_set, image_size)
        if path.exists():
            return np.load(path), "hit"
    vector = extract_feature_set(local_path, image_size=image_size, feature_set=feature_set)
    if cache_dir is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, vector.astype(np.float32))
    return vector, "miss"


def _extract_matrix(
    frame: pd.DataFrame,
    image_root: Path,
    feature_set: str,
    image_size: int,
    desc: str,
    skip_errors: bool,
    cache_dir: Path | None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]], list[dict[str, str]], dict[str, int]]:
    features = []
    labels = []
    rows = []
    skipped = []
    cache_stats = {"hits": 0, "misses": 0}
    for row in tqdm(frame.to_dict("records"), desc=desc):
        try:
            local_path = _resolve_image_path(image_root, row["image_path"])
            vector, cache_status = _extract_or_load_feature(local_path, image_size, feature_set, cache_dir)
            cache_stats["hits" if cache_status == "hit" else "misses"] += 1
        except Exception as exc:
            if not skip_errors:
                raise
            skipped.append({"id": str(row.get("id", "")), "image_path": str(row.get("image_path", "")), "error": repr(exc)})
            continue
        features.append(vector)
        labels.append(int(row["label"]))
        rows.append(
            {
                "id": str(row["id"]),
                "image_path": str(row["image_path"]),
                "local_path": str(local_path),
                "y_true": int(row["label"]),
                "type": str(row.get("type", "")),
            }
        )
    if not features:
        raise ValueError(f"No usable rows extracted for {desc}")
    return np.vstack(features), np.asarray(labels, dtype=int), rows, skipped, cache_stats


def _extract_unlabeled_matrix(
    frame: pd.DataFrame,
    image_root: Path,
    feature_set: str,
    image_size: int,
    desc: str,
    skip_errors: bool,
    cache_dir: Path | None,
) -> tuple[np.ndarray, list[dict[str, object]], list[dict[str, str]], dict[str, int]]:
    features = []
    rows = []
    skipped = []
    cache_stats = {"hits": 0, "misses": 0}
    for row in tqdm(frame.to_dict("records"), desc=desc):
        try:
            local_path = _resolve_image_path(image_root, row["image_path"])
            vector, cache_status = _extract_or_load_feature(local_path, image_size, feature_set, cache_dir)
            cache_stats["hits" if cache_status == "hit" else "misses"] += 1
        except Exception as exc:
            if not skip_errors:
                raise
            skipped.append({"id": str(row.get("id", "")), "image_path": str(row.get("image_path", "")), "error": repr(exc)})
            continue
        features.append(vector)
        rows.append(
            {
                "id": str(row["id"]),
                "image_path": str(row["image_path"]),
                "local_path": str(local_path),
                "type": str(row.get("type", "")),
            }
        )
    if not features:
        raise ValueError(f"No usable rows extracted for {desc}")
    return np.vstack(features), rows, skipped, cache_stats


def _classifier(name: str, seed: int):
    if name == "logistic_regression":
        return Pipeline(
            steps=[
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)),
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


def _scores(model, features: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(features)[:, 1], dtype=float)
    raw = np.asarray(model.decision_function(features), dtype=float)
    return 1.0 / (1.0 + np.exp(-raw))


def run_baseline(args: argparse.Namespace) -> dict[str, object]:
    seed_everything(args.seed)
    output_dir = ensure_dir(args.output_dir)
    image_root = Path(args.image_root)
    cache_dir = Path(args.feature_cache_dir) if args.feature_cache_dir else None
    train_frame = _limit_frame(pd.read_csv(args.train_csv), args.max_train_samples, args.seed, list(args.limit_balance_columns))
    val_frame = _limit_frame(pd.read_csv(args.val_csv), args.max_val_samples, args.seed + 1, list(args.limit_balance_columns))

    x_train, y_train, _train_rows, train_skipped, train_cache = _extract_matrix(
        train_frame, image_root, args.feature_set, args.image_size, "freuid/train", args.skip_errors, cache_dir
    )
    x_val, y_val, val_rows, val_skipped, val_cache = _extract_matrix(
        val_frame, image_root, args.feature_set, args.image_size, "freuid/val", args.skip_errors, cache_dir
    )

    model = _classifier(args.classifier, args.seed)
    model.fit(x_train, y_train)
    scores = _scores(model, x_val)
    operating_point = apcer_at_bpcer(y_val, scores, bpcer_target=0.01)
    labels = (scores >= operating_point.threshold).astype(int)

    metrics = binary_metrics(y_val, scores)
    metrics.update(freuid_metrics(y_val, scores))
    metrics.update(
        {
            "method": "freuid_feature_baseline",
            "feature_set": args.feature_set,
            "classifier": args.classifier,
            "image_size": int(args.image_size),
            "n_train": int(len(y_train)),
            "n_val": int(len(y_val)),
            "n_train_skipped": int(len(train_skipped)),
            "n_val_skipped": int(len(val_skipped)),
            "feature_cache_dir": str(cache_dir) if cache_dir is not None else None,
            "feature_cache": {
                "train_hits": int(train_cache["hits"]),
                "train_misses": int(train_cache["misses"]),
                "val_hits": int(val_cache["hits"]),
                "val_misses": int(val_cache["misses"]),
            },
            "feature_names": feature_names(args.feature_set),
            "threshold_for_1pct_bpcer": float(operating_point.threshold),
        }
    )
    write_json({"train": train_skipped, "val": val_skipped}, output_dir / "skipped.json")
    joblib.dump(model, output_dir / "classifier.joblib")
    np.savez_compressed(
        output_dir / "features.npz",
        x_train=x_train.astype(np.float32),
        y_train=y_train.astype(np.int64),
        x_val=x_val.astype(np.float32),
        y_val=y_val.astype(np.int64),
    )
    with (output_dir / "val_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "image_path", "local_path", "type", "y_true", "fraud_score", "label"],
        )
        writer.writeheader()
        for row, score, label in zip(val_rows, scores, labels):
            writer.writerow({**row, "fraud_score": float(score), "label": int(label)})

    if args.test_csv:
        test_frame = _limit_unlabeled_frame(_public_frame_from_ids(pd.read_csv(args.test_csv)), args.max_test_samples, args.seed + 2)
        x_test, test_rows, test_skipped, test_cache = _extract_unlabeled_matrix(
            test_frame, image_root, args.feature_set, args.image_size, "freuid/test", args.skip_errors, cache_dir
        )
        test_scores = _scores(model, x_test)
        test_predictions_path = Path(args.test_predictions_out) if args.test_predictions_out else output_dir / "test_predictions.csv"
        test_predictions_path.parent.mkdir(parents=True, exist_ok=True)
        with test_predictions_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["id", "image_path", "local_path", "type", "fraud_score"],
            )
            writer.writeheader()
            for row, score in zip(test_rows, test_scores):
                writer.writerow({**row, "fraud_score": float(score)})
        skipped = {"train": train_skipped, "val": val_skipped, "test": test_skipped}
        write_json(skipped, output_dir / "skipped.json")
        metrics.update(
            {
                "test_csv": str(args.test_csv),
                "test_predictions_path": str(test_predictions_path),
                "n_test": int(len(test_rows)),
                "n_test_skipped": int(len(test_skipped)),
            }
        )
        metrics["feature_cache"].update(
            {
                "test_hits": int(test_cache["hits"]),
                "test_misses": int(test_cache["misses"]),
            }
        )
    write_json(metrics, output_dir / "metrics.json")
    return metrics


def main() -> None:
    args = parse_args()
    metrics = run_baseline(args)
    print(Path(args.output_dir).resolve())
    print(f"apcer_at_1pct_bpcer={metrics['apcer_at_1pct_bpcer']:.6f}")
    print(f"audet_proxy={metrics['audet_proxy']:.6f}")


if __name__ == "__main__":
    main()
