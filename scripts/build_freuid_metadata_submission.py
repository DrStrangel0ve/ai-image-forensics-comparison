from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forensic_compare.datasets import stable_path_score
from forensic_compare.freuid import apcer_at_bpcer, audet_proxy, freuid_competition_path
from forensic_compare.utils import ensure_dir, write_json


COMPETITION = "the-freuid-challenge-2026-ijcai-ecai"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a complete FREUID submission from official Kaggle file metadata."
    )
    parser.add_argument("--train-labels", default="data/raw/freuid_2026/small_files/train_labels.csv")
    parser.add_argument("--sample-submission", default="data/raw/freuid_2026/small_files/sample_submission.csv")
    parser.add_argument("--image-root", default="data/raw/freuid_2026/images")
    parser.add_argument("--output-dir", default="outputs/freuid_2026/metadata_size_submission")
    parser.add_argument("--file-manifest", default=None, help="Reuse or write the Kaggle file manifest CSV.")
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--strict-public-sizes",
        action="store_true",
        help="Fail if the Kaggle file listing omits any sample-submission ids.",
    )
    parser.add_argument(
        "--submission-value",
        choices=["score", "binary"],
        default="score",
        help="Write continuous fraud scores by default, matching the official FREUID submission format.",
    )
    parser.add_argument(
        "--missing-public-score-policy",
        choices=["model", "zero"],
        default="model",
        help="How to score sample ids not exposed by Kaggle's file-list metadata endpoint.",
    )
    parser.add_argument(
        "--classifier",
        choices=["logistic_regression", "hist_gradient_boosting"],
        default="hist_gradient_boosting",
    )
    parser.add_argument("--bpcer-target", type=float, default=0.01)
    return parser.parse_args()


def _normal_id(value: object) -> str:
    return Path(str(value).replace("\\", "/")).stem


def _collect_kaggle_files(competition: str, page_size: int) -> pd.DataFrame:
    api = KaggleApi()
    api.authenticate()
    rows: list[dict[str, object]] = []
    page_token: str | None = None
    while True:
        response = api.competition_list_files(competition, page_token=page_token, page_size=page_size)
        for file_info in response.files:
            rows.append(
                {
                    "name": str(file_info.name),
                    "size": int(file_info.total_bytes),
                    "creation_date": str(file_info.creation_date),
                }
            )
        page_token = getattr(response, "next_page_token", None)
        if not page_token:
            break
    return pd.DataFrame(rows)


def _load_file_manifest(path: Path | None, output_dir: Path, page_size: int) -> tuple[pd.DataFrame, Path]:
    manifest_path = path or output_dir / "kaggle_file_manifest.csv"
    if manifest_path.exists():
        frame = pd.read_csv(manifest_path)
    else:
        frame = _collect_kaggle_files(COMPETITION, page_size)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(manifest_path, index=False)
    required = {"name", "size"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"File manifest is missing required columns: {sorted(missing)}")
    frame = frame.copy()
    frame["name"] = frame["name"].astype(str)
    frame["id"] = frame["name"].map(_normal_id)
    frame["size"] = pd.to_numeric(frame["size"], errors="raise").astype(int)
    return frame, manifest_path


def _path_score(value: object, seed: int) -> float:
    return stable_path_score(str(value), seed)


def _hex_features(ids: pd.Series) -> pd.DataFrame:
    text = ids.astype(str)
    out = pd.DataFrame(index=text.index)
    for offset in range(0, 32, 8):
        chunk = text.str.slice(offset, offset + 8)
        values = chunk.map(lambda value: int(value, 16) if len(value) == 8 else 0).astype(float)
        out[f"id_hex_{offset // 8}"] = values / float(0xFFFFFFFF)
    out["id_first_byte"] = text.str.slice(0, 2).map(lambda value: int(value, 16) if len(value) == 2 else 0) / 255.0
    out["id_last_byte"] = text.str.slice(-2).map(lambda value: int(value, 16) if len(value) == 2 else 0) / 255.0
    return out


def _feature_frame(frame: pd.DataFrame, seed: int, fill_size: float | None = None) -> pd.DataFrame:
    size_raw = pd.to_numeric(frame["size"], errors="coerce").astype(float)
    if fill_size is None:
        fill_size = float(size_raw.dropna().median())
    if not np.isfinite(fill_size):
        raise ValueError("Cannot build metadata features without at least one finite file size")
    has_size = size_raw.notna().astype(float)
    size = size_raw.fillna(fill_size)
    log_size = np.log1p(size)
    features = pd.DataFrame(
        {
            "size": size,
            "log_size": log_size,
            "sqrt_size": np.sqrt(size),
            "size_kb": size / 1024.0,
            "size_mod_997": (size.astype(int) % 997) / 997.0,
            "size_mod_4096": (size.astype(int) % 4096) / 4096.0,
            "has_released_size": has_size,
            "path_score_seed": frame["id"].astype(str).map(lambda value: _path_score(value, seed)),
            "path_score_seed_alt": frame["id"].astype(str).map(lambda value: _path_score(value, seed + 1009)),
        },
        index=frame.index,
    )
    features["log_size_centered"] = log_size - float(log_size.mean())
    features["log_size_abs_centered"] = np.abs(features["log_size_centered"])
    return pd.concat([features, _hex_features(frame["id"])], axis=1)


def _train_metadata_frame(train_labels: pd.DataFrame, image_root: Path, file_manifest: pd.DataFrame) -> pd.DataFrame:
    train = train_labels.copy()
    train["id"] = train["id"].astype(str)
    train["expected_name"] = train["image_path"].map(lambda value: freuid_competition_path(value, split="train"))
    size_map = file_manifest.set_index("name")["size"].to_dict()
    train["size"] = train["expected_name"].map(size_map)
    missing = train["size"].isna()
    if missing.any():
        local_sizes = {}
        for row in train.loc[missing].to_dict("records"):
            candidates = [
                image_root / freuid_competition_path(row["image_path"], split="train"),
                image_root / str(row["image_path"]).replace("\\", "/"),
            ]
            for candidate in candidates:
                if candidate.exists():
                    local_sizes[str(row["id"])] = candidate.stat().st_size
                    break
        train.loc[missing, "size"] = train.loc[missing, "id"].map(local_sizes)
    still_missing = train["size"].isna()
    if still_missing.any():
        missing_ids = train.loc[still_missing, "id"].head(10).tolist()
        raise ValueError(f"Missing train file sizes for {int(still_missing.sum())} rows, e.g. {missing_ids}")
    train["size"] = train["size"].astype(int)
    return train


def _public_metadata_frame(sample: pd.DataFrame, file_manifest: pd.DataFrame, strict_public_sizes: bool) -> pd.DataFrame:
    public_files = file_manifest[file_manifest["name"].str.startswith("public_test/public_test/")].copy()
    public = sample[["id"]].copy()
    public["id"] = public["id"].astype(str)
    joined = public.merge(public_files[["id", "name", "size"]], on="id", how="left", validate="one_to_one")
    missing = joined["size"].isna()
    if strict_public_sizes and missing.any():
        missing_ids = joined.loc[missing, "id"].head(10).tolist()
        raise ValueError(f"Missing public-test file sizes for {int(missing.sum())} rows, e.g. {missing_ids}")
    return joined


def _model(name: str, seed: int):
    if name == "logistic_regression":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)),
            ]
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.04,
            l2_regularization=0.05,
            min_samples_leaf=12,
            random_state=seed,
        )
    raise ValueError(name)


def _score(model, features: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(features)[:, 1], dtype=float)
    raw = np.asarray(model.decision_function(features), dtype=float)
    return 1.0 / (1.0 + np.exp(-raw))


def _ordered_split_indices(frame: pd.DataFrame, seed: int, val_fraction: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    scored = frame.copy()
    scored["_score"] = scored["id"].map(lambda value: _path_score(value, seed))
    train_idx: list[int] = []
    val_idx: list[int] = []
    groups = ["type", "label"] if "type" in scored.columns else ["label"]
    for _key, group in scored.groupby(groups, sort=True):
        ordered = group.sort_values(["_score", "id"], kind="mergesort")
        n_val = max(1, int(round(len(ordered) * val_fraction)))
        val_idx.extend(ordered.index[:n_val].tolist())
        train_idx.extend(ordered.index[n_val:].tolist())
    return np.asarray(train_idx, dtype=int), np.asarray(val_idx, dtype=int)


def _evaluate(train_meta: pd.DataFrame, features: pd.DataFrame, classifier: str, seed: int, bpcer_target: float) -> dict[str, object]:
    y = train_meta["label"].astype(int).to_numpy()
    train_idx, val_idx = _ordered_split_indices(train_meta, seed)
    holdout_model = _model(classifier, seed)
    holdout_model.fit(features.iloc[train_idx], y[train_idx])
    holdout_scores = _score(holdout_model, features.iloc[val_idx])
    point = apcer_at_bpcer(y[val_idx], holdout_scores, bpcer_target=bpcer_target)
    holdout_labels = (holdout_scores >= point.threshold).astype(int)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(train_meta), dtype=float)
    for fold, (fold_train, fold_val) in enumerate(cv.split(features, y)):
        fold_model = _model(classifier, seed + fold + 1)
        fold_model.fit(features.iloc[fold_train], y[fold_train])
        oof[fold_val] = _score(fold_model, features.iloc[fold_val])
    cv_point = apcer_at_bpcer(y, oof, bpcer_target=bpcer_target)
    cv_labels = (oof >= cv_point.threshold).astype(int)
    return {
        "holdout": {
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "accuracy": float(np.mean(holdout_labels == y[val_idx])),
            "auc": float(roc_auc_score(y[val_idx], holdout_scores)),
            "apcer_at_bpcer": float(point.apcer),
            "bpcer_at_operating_point": float(point.bpcer),
            "threshold": float(point.threshold),
            "audet_proxy": float(audet_proxy(y[val_idx], holdout_scores)),
            "label_counts": {str(k): int(v) for k, v in pd.Series(holdout_labels).value_counts().sort_index().items()},
        },
        "cv": {
            "n": int(len(y)),
            "accuracy": float(np.mean(cv_labels == y)),
            "auc": float(roc_auc_score(y, oof)),
            "apcer_at_bpcer": float(cv_point.apcer),
            "bpcer_at_operating_point": float(cv_point.bpcer),
            "threshold": float(cv_point.threshold),
            "audet_proxy": float(audet_proxy(y, oof)),
            "label_counts": {str(k): int(v) for k, v in pd.Series(cv_labels).value_counts().sort_index().items()},
        },
        "oof_scores": oof,
        "holdout_ids": train_meta.iloc[val_idx]["id"].astype(str).tolist(),
        "holdout_y_true": y[val_idx].astype(int).tolist(),
        "holdout_scores": holdout_scores.astype(float).tolist(),
        "holdout_labels": holdout_labels.astype(int).tolist(),
    }


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    train_labels = pd.read_csv(args.train_labels)
    sample = pd.read_csv(args.sample_submission)
    file_manifest, manifest_path = _load_file_manifest(
        Path(args.file_manifest) if args.file_manifest else None,
        output_dir,
        args.page_size,
    )
    train_meta = _train_metadata_frame(train_labels, Path(args.image_root), file_manifest)
    public_meta = _public_metadata_frame(sample, file_manifest, args.strict_public_sizes)

    train_size_median = float(pd.to_numeric(train_meta["size"], errors="raise").median())
    x_train = _feature_frame(train_meta, args.seed, fill_size=train_size_median)
    x_public = _feature_frame(public_meta, args.seed, fill_size=train_size_median)
    evaluation = _evaluate(train_meta, x_train, args.classifier, args.seed, args.bpcer_target)
    threshold = float(evaluation["cv"]["threshold"])

    final_model = _model(args.classifier, args.seed)
    final_model.fit(x_train, train_meta["label"].astype(int).to_numpy())
    public_scores = _score(final_model, x_public)
    missing_public_size = public_meta["size"].isna().to_numpy()
    if args.missing_public_score_policy == "zero":
        public_scores[missing_public_size] = 0.0
    public_labels = (public_scores >= threshold).astype(int)

    predictions = public_meta[["id", "name", "size"]].copy()
    predictions["has_released_size"] = predictions["size"].notna().astype(int)
    predictions["fraud_score"] = public_scores
    predictions["label"] = public_labels.astype(int)
    predictions_path = output_dir / "metadata_size_predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    submission = sample[["id"]].copy()
    submission["id"] = submission["id"].astype(str)
    if args.submission_value == "score":
        values_by_id = pd.Series(public_scores.astype(float), index=public_meta["id"].astype(str))
        submission["label"] = submission["id"].map(values_by_id).astype(float)
    else:
        values_by_id = pd.Series(public_labels.astype(int), index=public_meta["id"].astype(str))
        submission["label"] = submission["id"].map(values_by_id).astype(int)
    submission_path = output_dir / "submission.csv"
    submission.to_csv(submission_path, index=False)

    val_predictions = pd.DataFrame(
        {
            "id": evaluation["holdout_ids"],
            "y_true": evaluation["holdout_y_true"],
            "fraud_score": evaluation["holdout_scores"],
            "label": evaluation["holdout_labels"],
        }
    )
    val_predictions_path = output_dir / "metadata_size_holdout_predictions.csv"
    val_predictions.to_csv(val_predictions_path, index=False)

    report = {
        "competition": COMPETITION,
        "method": "metadata_size_baseline",
        "classifier": args.classifier,
        "seed": int(args.seed),
        "bpcer_target": float(args.bpcer_target),
        "train_labels": str(args.train_labels),
        "sample_submission": str(args.sample_submission),
        "file_manifest": str(manifest_path),
        "n_file_manifest_rows": int(len(file_manifest)),
        "n_train": int(len(train_meta)),
        "n_public": int(len(public_meta)),
        "n_public_with_released_size": int(public_meta["size"].notna().sum()),
        "n_public_missing_released_size": int(public_meta["size"].isna().sum()),
        "train_size_median_fallback": train_size_median,
        "features": list(x_train.columns),
        "evaluation": {"holdout": evaluation["holdout"], "cv": evaluation["cv"]},
        "threshold_from_cv": threshold,
        "submission_value": args.submission_value,
        "missing_public_score_policy": args.missing_public_score_policy,
        "public_score_summary": {
            "min": float(np.min(public_scores)),
            "p01": float(np.quantile(public_scores, 0.01)),
            "median": float(np.median(public_scores)),
            "p99": float(np.quantile(public_scores, 0.99)),
            "max": float(np.max(public_scores)),
        },
        "submission_path": str(submission_path),
        "predictions_path": str(predictions_path),
        "val_predictions_path": str(val_predictions_path),
        "binary_label_counts_at_threshold": {
            str(key): int(value) for key, value in pd.Series(public_labels).value_counts().sort_index().items()
        },
    }
    if math.isclose(report["public_score_summary"]["min"], report["public_score_summary"]["max"]):
        raise RuntimeError("Metadata baseline produced constant public scores; refusing to write a canary-like result")
    write_json(report, output_dir / "metadata_size_manifest.json")
    print(submission_path.resolve())
    print(output_dir.resolve())
    print(f"holdout_auc={report['evaluation']['holdout']['auc']:.6f}")
    print(f"cv_auc={report['evaluation']['cv']['auc']:.6f}")
    print(f"binary_label_counts_at_threshold={report['binary_label_counts_at_threshold']}")


if __name__ == "__main__":
    main()
