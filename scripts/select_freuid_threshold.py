from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forensic_compare.freuid import apcer_at_bpcer, audet_proxy
from forensic_compare.metrics import binary_metrics
from forensic_compare.utils import write_json


ID_COLUMNS = ["id", "image_id", "filename", "file_name", "image_path", "path"]
SCORE_COLUMNS = ["fraud_score", "fake_score", "prob_fraud", "prob_fake", "score", "prediction"]
TRUE_LABEL_COLUMNS = ["y_true", "true_label", "target"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select a FREUID validation threshold at a bounded BPCER operating point.")
    parser.add_argument("--predictions", required=True, help="Validation prediction CSV with true labels and fraud scores.")
    parser.add_argument("--out-json", required=True, help="Threshold/metric manifest to write.")
    parser.add_argument("--out-predictions", default=None, help="Optional thresholded validation prediction CSV.")
    parser.add_argument("--id-column", default=None)
    parser.add_argument("--label-column", default=None, help="True-label column; defaults to y_true/true_label/target.")
    parser.add_argument("--score-column", default=None)
    parser.add_argument("--bpcer-target", type=float, default=0.01)
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


def select_freuid_threshold(
    predictions_path: Path,
    out_json: Path,
    out_predictions: Path | None = None,
    id_column: str | None = None,
    label_column: str | None = None,
    score_column: str | None = None,
    bpcer_target: float = 0.01,
) -> dict[str, object]:
    frame = pd.read_csv(predictions_path)
    used_label_column = _detect_column(frame, label_column, TRUE_LABEL_COLUMNS, "true-label")
    used_score_column = _detect_column(frame, score_column, SCORE_COLUMNS, "score")
    used_id_column = _detect_column(frame, id_column, ID_COLUMNS, "id")
    if used_label_column is None:
        raise ValueError(f"Could not detect true-label column; tried {TRUE_LABEL_COLUMNS}")
    if used_score_column is None:
        raise ValueError(f"Could not detect score column; tried {SCORE_COLUMNS}")

    y_true = pd.to_numeric(frame[used_label_column], errors="raise").astype(int).to_numpy()
    scores = pd.to_numeric(frame[used_score_column], errors="raise").astype(float).to_numpy()
    if not np.isfinite(scores).all():
        raise ValueError("Prediction scores contain NaN or infinite values")

    point = apcer_at_bpcer(y_true, scores, bpcer_target=bpcer_target)
    labels = (scores >= point.threshold).astype(int)
    binary = binary_metrics(y_true, scores)
    manifest = {
        "predictions_path": str(predictions_path),
        "out_predictions": str(out_predictions) if out_predictions is not None else None,
        "n_rows": int(len(frame)),
        "id_column": used_id_column,
        "label_column": used_label_column,
        "score_column": used_score_column,
        "bpcer_target": float(bpcer_target),
        "threshold": float(point.threshold),
        "bpcer_at_operating_point": float(point.bpcer),
        "apcer_at_operating_point": float(point.apcer),
        "audet_proxy": float(audet_proxy(y_true, scores)),
        "roc_auc": float(binary["roc_auc"]),
        "accuracy_at_threshold": float(np.mean(labels == y_true)),
        "label_counts": {str(key): int(value) for key, value in pd.Series(labels).value_counts().sort_index().items()},
        "n_bona_fide": int(point.n_bona_fide),
        "n_attack": int(point.n_attack),
    }
    write_json(manifest, out_json)

    if out_predictions is not None:
        out_frame = pd.DataFrame(
            {
                "y_true": y_true,
                "fraud_score": scores,
                "label": labels,
            }
        )
        if used_id_column is not None:
            out_frame.insert(0, "id", frame[used_id_column].astype(str).to_numpy())
        out_predictions.parent.mkdir(parents=True, exist_ok=True)
        out_frame.to_csv(out_predictions, index=False)
    return manifest


def main() -> None:
    args = parse_args()
    manifest = select_freuid_threshold(
        predictions_path=Path(args.predictions),
        out_json=Path(args.out_json),
        out_predictions=Path(args.out_predictions) if args.out_predictions else None,
        id_column=args.id_column,
        label_column=args.label_column,
        score_column=args.score_column,
        bpcer_target=args.bpcer_target,
    )
    print(Path(args.out_json).resolve())
    if args.out_predictions:
        print(Path(args.out_predictions).resolve())
    print(
        "threshold="
        f"{manifest['threshold']:.8f} "
        f"apcer={manifest['apcer_at_operating_point']:.6f} "
        f"bpcer={manifest['bpcer_at_operating_point']:.6f} "
        f"audet_proxy={manifest['audet_proxy']:.6f}"
    )


if __name__ == "__main__":
    main()
