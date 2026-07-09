from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forensic_compare.utils import write_json


ID_COLUMNS = ["id", "image_id", "filename", "file_name", "image_path", "path"]
SCORE_COLUMNS = ["fraud_score", "fake_score", "prob_fraud", "prob_fake", "score", "prediction", "label"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Package score-valued FREUID predictions into Kaggle id,label format. "
            "Partial image predictions may replace a full fallback submission."
        )
    )
    parser.add_argument("--sample-submission", required=True, help="Official Kaggle sample_submission.csv.")
    parser.add_argument("--predictions", required=True, help="Partial or full score CSV with ids and fraud scores.")
    parser.add_argument("--fallback-submission", default=None, help="Optional full id,label score fallback CSV.")
    parser.add_argument("--out-path", required=True, help="Submission CSV to write.")
    parser.add_argument("--manifest-out", default=None, help="Optional JSON manifest; defaults to <out-path>.manifest.json.")
    parser.add_argument("--id-column", default=None)
    parser.add_argument("--score-column", default=None)
    return parser.parse_args()


def _detect_column(frame: pd.DataFrame, requested: str | None, candidates: list[str], kind: str) -> str:
    if requested:
        if requested not in frame.columns:
            raise ValueError(f"Requested {kind} column {requested!r} is not present")
        return requested
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise ValueError(f"Could not detect {kind} column; tried {candidates}")


def _id_value(value: object, column: str) -> str:
    raw = str(value)
    if column in {"image_path", "path"}:
        return Path(raw).stem
    return raw


def _load_id_score_map(path: Path, id_column: str | None, score_column: str | None) -> tuple[pd.Series, str, str]:
    frame = pd.read_csv(path)
    used_id_column = _detect_column(frame, id_column, ID_COLUMNS, "id")
    used_score_column = _detect_column(frame, score_column, SCORE_COLUMNS, "score")
    ids = frame[used_id_column].map(lambda value: _id_value(value, used_id_column)).astype(str)
    if ids.duplicated().any():
        duplicates = ids[ids.duplicated()].head(5).tolist()
        raise ValueError(f"{path} contains duplicate prediction ids: {duplicates}")
    scores = pd.to_numeric(frame[used_score_column], errors="raise").astype(float)
    if not np.isfinite(scores.to_numpy()).all():
        raise ValueError(f"{path} contains non-finite scores")
    return pd.Series(scores.clip(0.0, 1.0).to_numpy(), index=ids), used_id_column, used_score_column


def package_score_submission(
    sample_submission_path: Path,
    predictions_path: Path,
    out_path: Path,
    manifest_path: Path,
    fallback_submission_path: Path | None = None,
    id_column: str | None = None,
    score_column: str | None = None,
) -> dict[str, object]:
    sample = pd.read_csv(sample_submission_path)
    if list(sample.columns) != ["id", "label"]:
        raise ValueError("FREUID sample submission must have exactly columns: id,label")
    sample["id"] = sample["id"].astype(str)
    if sample["id"].duplicated().any():
        raise ValueError("Sample submission contains duplicate ids")
    sample_ids = set(sample["id"])

    pred_map, used_id_column, used_score_column = _load_id_score_map(predictions_path, id_column, score_column)
    extra_prediction_ids = sorted(set(pred_map.index) - sample_ids)
    if extra_prediction_ids:
        raise ValueError(f"Prediction ids are not in sample submission: {extra_prediction_ids[:5]}")

    if fallback_submission_path is not None:
        fallback = pd.read_csv(fallback_submission_path)
        if list(fallback.columns) != ["id", "label"]:
            raise ValueError("Fallback submission must have exactly columns: id,label")
        fallback["id"] = fallback["id"].astype(str)
        if fallback["id"].duplicated().any():
            raise ValueError("Fallback submission contains duplicate ids")
        if set(fallback["id"]) != sample_ids:
            raise ValueError("Fallback submission id set must exactly match sample submission")
        fallback_scores = pd.to_numeric(fallback["label"], errors="raise").astype(float)
        if not np.isfinite(fallback_scores.to_numpy()).all():
            raise ValueError("Fallback scores contain non-finite values")
        score_map = pd.Series(fallback_scores.clip(0.0, 1.0).to_numpy(), index=fallback["id"])
        score_map.loc[pred_map.index] = pred_map
    else:
        missing_prediction_ids = sorted(sample_ids - set(pred_map.index))
        if missing_prediction_ids:
            raise ValueError(f"Predictions are not full coverage and no fallback was provided: {missing_prediction_ids[:5]}")
        score_map = pred_map

    submission = pd.DataFrame({"id": sample["id"], "label": sample["id"].map(score_map).astype(float)})
    if submission["label"].isna().any():
        raise ValueError("Submission contains missing labels")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)
    manifest = {
        "sample_submission_path": str(sample_submission_path),
        "predictions_path": str(predictions_path),
        "fallback_submission_path": str(fallback_submission_path) if fallback_submission_path is not None else None,
        "output_path": str(out_path),
        "n_rows": int(len(submission)),
        "n_prediction_rows": int(len(pred_map)),
        "n_fallback_rows": int(len(submission) - len(pred_map)) if fallback_submission_path is not None else 0,
        "id_column": used_id_column,
        "score_column": used_score_column,
        "label_min": float(submission["label"].min()),
        "label_max": float(submission["label"].max()),
        "label_mean": float(submission["label"].mean()),
        "prediction_score_min": float(pred_map.min()),
        "prediction_score_max": float(pred_map.max()),
        "prediction_score_mean": float(pred_map.mean()),
        "columns": list(submission.columns),
    }
    write_json(manifest, manifest_path)
    return manifest


def main() -> None:
    args = parse_args()
    out_path = Path(args.out_path)
    manifest_path = Path(args.manifest_out) if args.manifest_out else out_path.with_suffix(".manifest.json")
    manifest = package_score_submission(
        sample_submission_path=Path(args.sample_submission),
        predictions_path=Path(args.predictions),
        fallback_submission_path=Path(args.fallback_submission) if args.fallback_submission else None,
        out_path=out_path,
        manifest_path=manifest_path,
        id_column=args.id_column,
        score_column=args.score_column,
    )
    print(Path(args.out_path).resolve())
    print(Path(manifest_path).resolve())
    print(f"rows={manifest['n_rows']} predictions={manifest['n_prediction_rows']} fallback={manifest['n_fallback_rows']}")


if __name__ == "__main__":
    main()
