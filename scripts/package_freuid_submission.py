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
SCORE_COLUMNS = ["fraud_score", "fake_score", "prob_fraud", "prob_fake", "score", "prediction"]
LABEL_COLUMNS = ["label", "predicted_label", "target"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package predictions into Kaggle FREUID `id,label` submission format.")
    parser.add_argument("--sample-submission", required=True, help="Kaggle sample_submission.csv.")
    parser.add_argument("--out-path", required=True, help="Submission CSV to write.")
    parser.add_argument("--manifest-out", default=None, help="Optional JSON manifest; defaults to <out-path>.manifest.json.")
    parser.add_argument("--predictions", default=None, help="Optional prediction CSV with ids and labels or fraud scores.")
    parser.add_argument("--id-column", default=None)
    parser.add_argument("--score-column", default=None)
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--constant-label",
        type=int,
        choices=[0, 1],
        default=0,
        help="Fallback label when --predictions is omitted.",
    )
    return parser.parse_args()


def _detect_column(frame: pd.DataFrame, requested: str | None, candidates: list[str], kind: str) -> str | None:
    if requested:
        if requested not in frame.columns:
            raise ValueError(f"Requested {kind} column {requested!r} is not present")
        return requested
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def _id_value(value: object, column: str) -> str:
    raw = str(value)
    if column in {"image_path", "path"}:
        return Path(raw).stem
    return raw


def package_freuid_submission(
    sample_submission_path: Path,
    out_path: Path,
    manifest_path: Path,
    predictions_path: Path | None = None,
    id_column: str | None = None,
    score_column: str | None = None,
    label_column: str | None = None,
    threshold: float = 0.5,
    constant_label: int = 0,
) -> tuple[pd.DataFrame, dict[str, object]]:
    sample = pd.read_csv(sample_submission_path)
    if list(sample.columns) != ["id", "label"]:
        raise ValueError("FREUID sample submission must have exactly columns: id,label")
    if sample["id"].duplicated().any():
        raise ValueError("Sample submission contains duplicate ids")
    sample_ids = sample["id"].astype(str)

    if predictions_path is None:
        labels = pd.Series(np.full(len(sample), constant_label, dtype=int), index=sample.index)
        used_id_column = None
        used_score_column = None
        used_label_column = None
        matched_predictions = 0
    else:
        predictions = pd.read_csv(predictions_path)
        used_id_column = _detect_column(predictions, id_column, ID_COLUMNS, "id")
        if used_id_column is None:
            raise ValueError(f"Could not detect id column; tried {ID_COLUMNS}")
        prediction_ids = predictions[used_id_column].map(lambda value: _id_value(value, used_id_column))
        if prediction_ids.duplicated().any():
            duplicates = prediction_ids[prediction_ids.duplicated()].head(5).tolist()
            raise ValueError(f"Prediction ids contain duplicates: {duplicates}")

        used_label_column = _detect_column(predictions, label_column, LABEL_COLUMNS, "label")
        used_score_column = _detect_column(predictions, score_column, SCORE_COLUMNS, "score")
        if used_label_column is None and used_score_column is None:
            raise ValueError(f"Could not detect label or score columns; tried {LABEL_COLUMNS + SCORE_COLUMNS}")
        if used_label_column is not None:
            pred_labels = pd.to_numeric(predictions[used_label_column], errors="raise").astype(int)
        else:
            scores = pd.to_numeric(predictions[used_score_column], errors="raise").astype(float)
            if not np.isfinite(scores.to_numpy()).all():
                raise ValueError("Prediction scores contain NaN or infinite values")
            pred_labels = (scores >= threshold).astype(int)
        if not pred_labels.isin([0, 1]).all():
            raise ValueError("FREUID labels must be binary 0/1")

        label_map = pd.Series(pred_labels.to_numpy(), index=prediction_ids)
        missing = sorted(set(sample_ids) - set(label_map.index))
        extra = sorted(set(label_map.index) - set(sample_ids))
        if missing or extra:
            raise ValueError(f"Prediction ids do not match sample ids: missing={missing[:5]}, extra={extra[:5]}")
        labels = sample_ids.map(label_map).astype(int)
        matched_predictions = int(len(label_map))

    submission = pd.DataFrame({"id": sample_ids, "label": labels.astype(int)})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)
    manifest = {
        "sample_submission_path": str(sample_submission_path),
        "predictions_path": str(predictions_path) if predictions_path is not None else None,
        "output_path": str(out_path),
        "n_rows": int(len(submission)),
        "matched_predictions": matched_predictions,
        "id_column": used_id_column,
        "score_column": used_score_column,
        "label_column": used_label_column,
        "threshold": threshold,
        "constant_label": constant_label if predictions_path is None else None,
        "label_counts": {str(key): int(value) for key, value in submission["label"].value_counts().sort_index().items()},
        "columns": list(submission.columns),
    }
    write_json(manifest, manifest_path)
    return submission, manifest


def main() -> None:
    args = parse_args()
    out_path = Path(args.out_path)
    manifest_path = Path(args.manifest_out) if args.manifest_out else out_path.with_suffix(".manifest.json")
    package_freuid_submission(
        sample_submission_path=Path(args.sample_submission),
        out_path=out_path,
        manifest_path=manifest_path,
        predictions_path=Path(args.predictions) if args.predictions else None,
        id_column=args.id_column,
        score_column=args.score_column,
        label_column=args.label_column,
        threshold=args.threshold,
        constant_label=args.constant_label,
    )
    print(out_path.resolve())
    print(manifest_path.resolve())


if __name__ == "__main__":
    main()
