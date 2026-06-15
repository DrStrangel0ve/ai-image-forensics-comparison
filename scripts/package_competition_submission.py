from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forensic_compare.utils import write_json


ID_COLUMNS = [
    "image_id",
    "id",
    "filename",
    "file_name",
    "image",
    "image_path",
    "file_path",
    "filepath",
    "path",
]
PATH_LIKE_COLUMNS = {"image_path", "file_path", "filepath", "path"}
SCORE_COLUMNS = [
    "fake_score",
    "prob_fake",
    "fake_probability",
    "score",
    "prediction",
    "predicted_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package prediction scores into a competition-style AI-image detection submission."
    )
    parser.add_argument("--predictions", required=True, help="Input prediction CSV.")
    parser.add_argument("--out-path", required=True, help="Submission CSV to write.")
    parser.add_argument(
        "--manifest-out",
        default=None,
        help="Optional JSON manifest. Defaults to <out-path>.manifest.json.",
    )
    parser.add_argument("--id-column", default=None, help="Input image id/path column. Auto-detected by default.")
    parser.add_argument("--score-column", default=None, help="Input fake-score column. Auto-detected by default.")
    parser.add_argument(
        "--id-from-path",
        choices=["stem", "name", "path"],
        default="stem",
        help="How to convert path-like ids into image_id values.",
    )
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument("--real-threshold", type=float, default=0.2)
    parser.add_argument("--fake-threshold", type=float, default=0.8)
    parser.add_argument("--score-precision", type=int, default=6)
    parser.add_argument(
        "--sort-by-id",
        action="store_true",
        help="Sort rows by image_id instead of preserving input order.",
    )
    parser.add_argument(
        "--clip-scores",
        action="store_true",
        help="Clip scores to [0, 1] instead of failing when out-of-range scores appear.",
    )
    return parser.parse_args()


def _detect_column(frame: pd.DataFrame, requested: str | None, candidates: list[str], kind: str) -> str:
    if requested:
        if requested not in frame.columns:
            raise ValueError(f"Requested {kind} column {requested!r} is not present")
        return requested
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise ValueError(f"Could not detect a {kind} column; tried {candidates}")


def _image_id(value: object, column: str, id_from_path: str) -> str:
    raw = str(value)
    if column not in PATH_LIKE_COLUMNS or id_from_path == "path":
        return raw
    path = Path(raw)
    if id_from_path == "name":
        return path.name
    return path.stem


def _scores(series: pd.Series, clip_scores: bool) -> pd.Series:
    scores = pd.to_numeric(series, errors="raise").astype(float)
    if not np.isfinite(scores.to_numpy()).all():
        raise ValueError("Scores contain NaN or infinite values")
    if clip_scores:
        return scores.clip(0.0, 1.0)
    out_of_range = scores[(scores < 0.0) | (scores > 1.0)]
    if not out_of_range.empty:
        raise ValueError("Scores must be probabilities in [0, 1]; pass --clip-scores to clip them")
    return scores


def _triage_decision(score: float, real_threshold: float, fake_threshold: float) -> str:
    if score <= real_threshold:
        return "likely_real"
    if score >= fake_threshold:
        return "likely_fake"
    return "uncertain"


def package_submission(
    predictions_path: Path,
    out_path: Path,
    manifest_path: Path,
    id_column: str | None = None,
    score_column: str | None = None,
    id_from_path: str = "stem",
    decision_threshold: float = 0.5,
    real_threshold: float = 0.2,
    fake_threshold: float = 0.8,
    score_precision: int = 6,
    sort_by_id: bool = False,
    clip_scores: bool = False,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if real_threshold >= fake_threshold:
        raise ValueError("--real-threshold must be lower than --fake-threshold")
    frame = pd.read_csv(predictions_path)
    id_col = _detect_column(frame, id_column, ID_COLUMNS, "image id")
    score_col = _detect_column(frame, score_column, SCORE_COLUMNS, "fake score")
    scores = _scores(frame[score_col], clip_scores)

    submission = pd.DataFrame(
        {
            "image_id": [_image_id(value, id_col, id_from_path) for value in frame[id_col]],
            "fake_score": scores.round(score_precision),
            "predicted_label": (scores >= decision_threshold).astype(int),
            "predicted_label_name": np.where(scores >= decision_threshold, "fake", "real"),
            "confidence": (np.abs(scores - 0.5) * 2.0).round(score_precision),
            "triage_decision": [
                _triage_decision(float(score), real_threshold, fake_threshold) for score in scores
            ],
        }
    )
    if submission["image_id"].duplicated().any():
        duplicates = submission.loc[submission["image_id"].duplicated(), "image_id"].tolist()
        raise ValueError(f"Duplicate image_id values are not allowed: {duplicates[:5]}")
    if sort_by_id:
        submission = submission.sort_values("image_id", ignore_index=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)
    manifest = {
        "input_path": str(predictions_path),
        "output_path": str(out_path),
        "n_rows": int(len(submission)),
        "id_column": id_col,
        "score_column": score_col,
        "decision_threshold": decision_threshold,
        "real_threshold": real_threshold,
        "fake_threshold": fake_threshold,
        "score_min": float(scores.min()) if len(scores) else None,
        "score_max": float(scores.max()) if len(scores) else None,
        "label_counts": {
            str(key): int(value) for key, value in submission["predicted_label_name"].value_counts().items()
        },
        "triage_counts": {
            str(key): int(value) for key, value in submission["triage_decision"].value_counts().items()
        },
        "y_true_present_excluded": "y_true" in frame.columns,
        "columns": list(submission.columns),
    }
    write_json(manifest, manifest_path)
    return submission, manifest


def main() -> None:
    args = parse_args()
    out_path = Path(args.out_path)
    manifest_path = Path(args.manifest_out) if args.manifest_out else out_path.with_suffix(".manifest.json")
    package_submission(
        predictions_path=Path(args.predictions),
        out_path=out_path,
        manifest_path=manifest_path,
        id_column=args.id_column,
        score_column=args.score_column,
        id_from_path=args.id_from_path,
        decision_threshold=args.decision_threshold,
        real_threshold=args.real_threshold,
        fake_threshold=args.fake_threshold,
        score_precision=args.score_precision,
        sort_by_id=args.sort_by_id,
        clip_scores=args.clip_scores,
    )
    print(out_path.resolve())
    print(manifest_path.resolve())


if __name__ == "__main__":
    main()
