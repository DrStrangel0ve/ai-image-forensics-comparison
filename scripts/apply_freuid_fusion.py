from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forensic_compare.utils import read_json, write_json


ID_COLUMNS = ["id", "image_id", "filename", "file_name", "image_path", "path"]
SCORE_COLUMNS = ["fraud_score", "fake_score", "prob_fraud", "prob_fake", "score", "prediction"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply a saved FREUID score-fusion recipe to unlabeled predictions.")
    parser.add_argument("--predictions", nargs="+", required=True, help="Prediction CSVs with id and fraud score columns.")
    parser.add_argument("--names", nargs="+", default=None, help="Names matching fusion_summary.json source_names.")
    parser.add_argument("--fusion-summary", required=True, help="fusion_summary.json from scripts/fuse_freuid_scores.py.")
    parser.add_argument("--out-predictions", required=True, help="Output CSV with id,fraud_score,label.")
    parser.add_argument("--manifest-out", default=None, help="Optional JSON manifest; defaults to <out-predictions>.manifest.json.")
    parser.add_argument("--threshold-json", default=None, help="Threshold manifest from scripts/select_freuid_threshold.py.")
    parser.add_argument("--threshold", type=float, default=None, help="Manual threshold override.")
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="Optional inference-time weight override in source-name order; values are normalized to sum to one.",
    )
    parser.add_argument(
        "--normalization",
        choices=["raw", "minmax", "rank"],
        default=None,
        help="Optional normalization override for an explicit inference probe.",
    )
    parser.add_argument("--id-column", default=None)
    parser.add_argument("--score-column", default=None)
    return parser.parse_args()


def _source_name(path: Path) -> str:
    parent = path.parent.name
    return parent if parent else path.stem


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


def _normalize(values: np.ndarray, mode: str) -> np.ndarray:
    if mode == "raw":
        return values.astype(float)
    if mode == "minmax":
        result = np.zeros_like(values, dtype=float)
        for col in range(values.shape[1]):
            column = values[:, col]
            span = float(column.max() - column.min())
            result[:, col] = 0.5 if span < 1e-12 else (column - column.min()) / span
        return result
    if mode == "rank":
        return np.column_stack(
            [pd.Series(values[:, col]).rank(method="average", pct=True).to_numpy(dtype=float) for col in range(values.shape[1])]
        )
    raise ValueError(f"Unsupported normalization: {mode}")


def _load_prediction_frames(
    paths: list[Path],
    names: list[str] | None,
    id_column: str | None,
    score_column: str | None,
) -> tuple[pd.DataFrame, list[str]]:
    source_names = names or [_source_name(path) for path in paths]
    if len(source_names) != len(paths):
        raise ValueError("--names must have the same length as --predictions")
    if len(set(source_names)) != len(source_names):
        raise ValueError(f"Prediction source names must be unique: {source_names}")

    base: pd.DataFrame | None = None
    for source_name, path in zip(source_names, paths):
        frame = pd.read_csv(path)
        used_id_column = _detect_column(frame, id_column, ID_COLUMNS, "id")
        used_score_column = _detect_column(frame, score_column, SCORE_COLUMNS, "score")
        ids = frame[used_id_column].map(lambda value: _id_value(value, used_id_column))
        scores = pd.to_numeric(frame[used_score_column], errors="raise").astype(float)
        if ids.duplicated().any():
            duplicates = ids[ids.duplicated()].head(5).tolist()
            raise ValueError(f"{path} contains duplicate ids: {duplicates}")
        if not np.isfinite(scores.to_numpy()).all():
            raise ValueError(f"{path} contains NaN or infinite scores")
        keep = pd.DataFrame({"id": ids.astype(str), source_name: scores.to_numpy(dtype=float)})
        if base is None:
            base = keep
        else:
            base = base.merge(keep, on="id", how="inner", validate="one_to_one")
    if base is None or base.empty:
        raise ValueError("No prediction rows to fuse")
    return base, source_names


def _threshold_from_inputs(fusion_summary: dict[str, object], threshold_json: Path | None, threshold: float | None) -> float:
    if threshold is not None:
        return float(threshold)
    if threshold_json is not None:
        manifest = read_json(threshold_json)
        if "threshold" not in manifest:
            raise ValueError(f"{threshold_json} does not contain a threshold field")
        return float(manifest["threshold"])
    best = fusion_summary.get("best")
    if isinstance(best, dict) and "threshold_at_1pct_bpcer" in best:
        return float(best["threshold_at_1pct_bpcer"])
    raise ValueError("Provide --threshold, --threshold-json, or a fusion summary with best.threshold_at_1pct_bpcer")


def apply_freuid_fusion(
    prediction_paths: list[Path],
    fusion_summary_path: Path,
    out_predictions: Path,
    manifest_path: Path,
    names: list[str] | None = None,
    threshold_json: Path | None = None,
    threshold: float | None = None,
    weights_override: list[float] | None = None,
    normalization_override: str | None = None,
    id_column: str | None = None,
    score_column: str | None = None,
) -> dict[str, object]:
    fusion_summary = read_json(fusion_summary_path)
    best = fusion_summary.get("best")
    if not isinstance(best, dict):
        raise ValueError(f"{fusion_summary_path} is missing best fusion settings")
    expected_names = list(fusion_summary.get("source_names", []))
    weights = np.asarray(weights_override if weights_override is not None else best.get("weights", []), dtype=float)
    normalization = str(normalization_override or best.get("normalization", "raw"))
    if len(expected_names) != len(weights):
        raise ValueError("fusion_summary source_names and best.weights lengths differ")
    if not np.isfinite(weights).all() or np.any(weights < 0.0) or float(weights.sum()) <= 0.0:
        raise ValueError("Fusion weights must be finite, non-negative, and sum to a positive value")
    weights = weights / float(weights.sum())

    frame, source_names = _load_prediction_frames(prediction_paths, names, id_column, score_column)
    if source_names != expected_names:
        raise ValueError(f"Prediction names must match fusion source_names exactly: expected {expected_names}, got {source_names}")

    threshold_value = _threshold_from_inputs(fusion_summary, threshold_json, threshold)
    scores = _normalize(frame[source_names].to_numpy(dtype=float), normalization) @ weights
    labels = (scores >= threshold_value).astype(int)
    output = pd.DataFrame({"id": frame["id"], "fraud_score": scores, "label": labels})
    out_predictions.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_predictions, index=False)

    manifest = {
        "prediction_paths": [str(path) for path in prediction_paths],
        "fusion_summary_path": str(fusion_summary_path),
        "threshold_json": str(threshold_json) if threshold_json is not None else None,
        "out_predictions": str(out_predictions),
        "n_rows": int(len(output)),
        "source_names": source_names,
        "normalization": normalization,
        "weights": [float(value) for value in weights],
        "threshold": float(threshold_value),
        "label_counts": {str(key): int(value) for key, value in output["label"].value_counts().sort_index().items()},
    }
    write_json(manifest, manifest_path)
    return manifest


def main() -> None:
    args = parse_args()
    out_predictions = Path(args.out_predictions)
    manifest_path = Path(args.manifest_out) if args.manifest_out else out_predictions.with_suffix(".manifest.json")
    manifest = apply_freuid_fusion(
        prediction_paths=[Path(path) for path in args.predictions],
        names=list(args.names) if args.names else None,
        fusion_summary_path=Path(args.fusion_summary),
        out_predictions=out_predictions,
        manifest_path=manifest_path,
        threshold_json=Path(args.threshold_json) if args.threshold_json else None,
        threshold=args.threshold,
        weights_override=list(args.weights) if args.weights else None,
        normalization_override=args.normalization,
        id_column=args.id_column,
        score_column=args.score_column,
    )
    print(out_predictions.resolve())
    print(manifest_path.resolve())
    print(
        "fusion="
        f"normalization={manifest['normalization']} "
        f"threshold={manifest['threshold']:.8f} "
        f"label_counts={manifest['label_counts']}"
    )


if __name__ == "__main__":
    main()
