from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from forensic_compare.conventional import extract_feature_set
from forensic_compare.datasets import collect_labeled_images, discover_layout, stratified_split
from forensic_compare.metrics import binary_metrics
from forensic_compare.utils import ensure_dir, read_json


SCORE_MODES = ["global", "tile_mean", "tile_max", "tile_top2_mean"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a saved conventional feature model with native-image tile aggregation. "
            "The model is still the original resized-input model; only target scoring is tiled."
        )
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 29])
    parser.add_argument(
        "--model-template",
        default="runs/ms_cocoai_to_ishu_neural_fusion/combined_v3_seed{seed}",
        help="Template containing feature_model.joblib and metrics.json for each seed.",
    )
    parser.add_argument(
        "--target-template",
        default="data/raw/ishu_ai_vs_real_2026_seed{seed}_test",
        help="Template for each seed-specific target image-folder dataset.",
    )
    parser.add_argument("--target-split", choices=["all", "test"], default="all")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument(
        "--tile-size",
        type=int,
        default=512,
        help="Native-pixel square crop size before resizing each tile to --image-size.",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/ms_cocoai_to_ishu_combined_v3_native_tiling",
    )
    parser.add_argument("--summary-dir", default="reports/assets")
    parser.add_argument(
        "--report-path",
        default="reports/ms_cocoai_to_ishu_combined_v3_native_tiling_2026_06_13.md",
    )
    parser.add_argument("--skip-errors", action="store_true")
    return parser.parse_args()


def _target_records(target_dir: Path, target_split: str, val_fraction: float, seed: int):
    layout = discover_layout(target_dir)
    if target_split == "test" and layout.test:
        return collect_labeled_images(layout.test)
    if layout.single:
        records = collect_labeled_images(layout.single)
        if target_split == "test":
            _train, test = stratified_split(records, val_fraction, seed)
            return test
        return records
    if layout.train and layout.test:
        if target_split == "all":
            return collect_labeled_images(layout.train) + collect_labeled_images(layout.test)
        return collect_labeled_images(layout.test)
    raise ValueError(f"Unsupported dataset layout: {layout}")


def tile_boxes(width: int, height: int, tile_size: int) -> list[tuple[int, int, int, int]]:
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if max(width, height) <= tile_size:
        return [(0, 0, width, height)]

    side = min(tile_size, width, height)

    def positions(length: int) -> list[int]:
        if length <= side:
            return [0]
        return sorted({0, (length - side) // 2, length - side})

    boxes = []
    for top in positions(height):
        for left in positions(width):
            boxes.append((left, top, left + side, top + side))
    return boxes


def aggregate_scores(global_score: float, tile_scores: np.ndarray) -> dict[str, float]:
    if len(tile_scores) == 0:
        tile_scores = np.asarray([global_score], dtype=float)
    sorted_scores = np.sort(np.asarray(tile_scores, dtype=float))
    top_k = sorted_scores[-min(2, len(sorted_scores)) :]
    return {
        "global": float(global_score),
        "tile_mean": float(tile_scores.mean()),
        "tile_max": float(tile_scores.max()),
        "tile_top2_mean": float(top_k.mean()),
        "tile_std": float(tile_scores.std()),
    }


def _score_features(model, features: np.ndarray) -> float:
    matrix = np.asarray(features, dtype=np.float32).reshape(1, -1)
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(matrix)[0, 1])
    raw_score = float(model.decision_function(matrix)[0])
    return float(1.0 / (1.0 + np.exp(-raw_score)))


def _score_tile(
    model,
    tile: Image.Image,
    temp_path: Path,
    image_size: int,
    feature_set: str,
) -> float:
    tile.save(temp_path, format="PNG")
    features = extract_feature_set(temp_path, image_size=image_size, feature_set=feature_set)
    return _score_features(model, features)


def evaluate_seed(
    seed: int,
    model_dir: Path,
    target_dir: Path,
    target_split: str,
    val_fraction: float,
    image_size: int,
    tile_size: int,
    output_dir: Path,
    skip_errors: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_metrics = read_json(model_dir / "metrics.json")
    feature_set = str(source_metrics["feature_set"])
    model = joblib.load(model_dir / "feature_model.joblib")
    records = _target_records(target_dir, target_split, val_fraction, seed)

    rows = []
    skipped = []
    with tempfile.TemporaryDirectory(prefix=f"tiled_features_seed{seed}_") as temp_root:
        temp_path = Path(temp_root) / "tile.png"
        for path, label, class_name in tqdm(records, desc=f"seed{seed}/tiles"):
            try:
                global_features = extract_feature_set(
                    path,
                    image_size=image_size,
                    feature_set=feature_set,
                )
                global_score = _score_features(model, global_features)
                with Image.open(path) as image:
                    image = image.convert("RGB")
                    width, height = image.size
                    boxes = tile_boxes(width, height, tile_size)
                    tile_scores = np.asarray(
                        [
                            _score_tile(
                                model,
                                image.crop(box),
                                temp_path,
                                image_size,
                                feature_set,
                            )
                            for box in boxes
                        ],
                        dtype=float,
                    )
                aggregated = aggregate_scores(global_score, tile_scores)
                rows.append(
                    {
                        "seed": seed,
                        "path": str(path),
                        "class_name": class_name,
                        "y_true": int(label),
                        "width": width,
                        "height": height,
                        "tile_size": tile_size,
                        "n_tiles": len(tile_scores),
                        **{f"{key}_score": value for key, value in aggregated.items()},
                        "tile_max_minus_global": float(aggregated["tile_max"] - global_score),
                    }
                )
            except Exception as exc:
                if not skip_errors:
                    raise
                skipped.append({"seed": seed, "path": str(path), "error": repr(exc)})

    if not rows:
        raise ValueError(f"No usable target rows for seed={seed}")

    detail = pd.DataFrame(rows)
    metrics_rows = []
    y_true = detail["y_true"].to_numpy(dtype=int)
    for score_mode in SCORE_MODES:
        scores = detail[f"{score_mode}_score"].to_numpy(dtype=float)
        metrics = binary_metrics(y_true, scores)
        metrics_rows.append(
            {
                "seed": seed,
                "score_mode": score_mode,
                "feature_set": feature_set,
                "image_size": image_size,
                "tile_size": tile_size,
                "n_images": int(len(detail)),
                "mean_tiles": float(detail["n_tiles"].mean()),
                "accuracy": float(metrics["accuracy"]),
                "roc_auc": metrics["roc_auc"],
                "brier_score": float(metrics["brier_score"]),
                "expected_calibration_error": float(metrics["expected_calibration_error"]),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["f1"]),
                "predicted_fake_rate": float((scores >= 0.5).mean()),
            }
        )

    seed_output_dir = ensure_dir(output_dir / f"seed{seed}")
    detail.to_csv(seed_output_dir / "predictions.csv", index=False)
    pd.DataFrame(metrics_rows).to_csv(seed_output_dir / "metrics_by_score_mode.csv", index=False)
    if skipped:
        pd.DataFrame(skipped).to_csv(seed_output_dir / "skipped.csv", index=False)
    return detail, pd.DataFrame(metrics_rows)


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for score_mode, group in metrics.groupby("score_mode", sort=False):
        row = {
            "score_mode": score_mode,
            "n_seeds": int(group["seed"].nunique()),
            "n_images_total": int(group["n_images"].sum()),
            "mean_tiles": float(group["mean_tiles"].mean()),
        }
        for column in [
            "accuracy",
            "roc_auc",
            "brier_score",
            "expected_calibration_error",
            "precision",
            "recall",
            "f1",
            "predicted_fake_rate",
        ]:
            row[f"{column}_mean"] = float(group[column].astype(float).mean())
            row[f"{column}_std"] = float(group[column].astype(float).std(ddof=1))
        rows.append(row)
    order = {mode: index for index, mode in enumerate(SCORE_MODES)}
    return pd.DataFrame(rows).sort_values("score_mode", key=lambda s: s.map(order)).reset_index(drop=True)


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    display = frame[columns].copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{float(value):.4f}")
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in display.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_report(summary: pd.DataFrame, metrics: pd.DataFrame, args: argparse.Namespace) -> None:
    global_row = summary[summary["score_mode"] == "global"].iloc[0]
    best_accuracy = summary.sort_values("accuracy_mean", ascending=False).iloc[0]
    best_auc = summary.sort_values("roc_auc_mean", ascending=False).iloc[0]
    report_path = Path(args.report_path)
    lines = [
        "# MS COCOAI to Ishu combined_v3 Native-Tiling Diagnostic",
        "",
        (
            "This bounded diagnostic asks whether the MS-trained `combined_v3` conventional branch benefits "
            "from scoring native-resolution target crops instead of only the global resized view. It uses the "
            "same saved feature model; each tile is resized to the original feature extraction size before scoring."
        ),
        "",
        "## Summary",
        "",
        _markdown_table(
            summary,
            [
                "score_mode",
                "n_seeds",
                "accuracy_mean",
                "roc_auc_mean",
                "brier_score_mean",
                "expected_calibration_error_mean",
                "predicted_fake_rate_mean",
                "mean_tiles",
            ],
        ),
        "",
        "## Interpretation",
        "",
        (
            f"The global resized view reaches {global_row['accuracy_mean']:.4f} mean accuracy / "
            f"{global_row['roc_auc_mean']:.4f} mean AUC. The best default-threshold accuracy mode is "
            f"`{best_accuracy['score_mode']}` at {best_accuracy['accuracy_mean']:.4f}; the best ranking mode is "
            f"`{best_auc['score_mode']}` at {best_auc['roc_auc_mean']:.4f} AUC."
        ),
        (
            "Treat this as a branch-level diagnostic rather than a new SCP-Fusion result: tile aggregation can "
            "change sensitivity to local artifacts, but it also changes calibration and fake-call rate before any "
            "source-aware thresholding."
        ),
        "",
        "## Per-Seed Metrics",
        "",
        _markdown_table(
            metrics.sort_values(["score_mode", "seed"]),
            [
                "score_mode",
                "seed",
                "accuracy",
                "roc_auc",
                "brier_score",
                "expected_calibration_error",
                "predicted_fake_rate",
                "mean_tiles",
            ],
        ),
        "",
        "## Rebuild",
        "",
        "```powershell",
        ".\\.venv\\Scripts\\python.exe scripts\\evaluate_tiled_feature_model.py",
        "```",
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(Path(args.output_dir))
    all_detail = []
    all_metrics = []
    for seed in args.seeds:
        detail, metrics = evaluate_seed(
            seed=seed,
            model_dir=Path(args.model_template.format(seed=seed)),
            target_dir=Path(args.target_template.format(seed=seed)),
            target_split=args.target_split,
            val_fraction=args.val_fraction,
            image_size=args.image_size,
            tile_size=args.tile_size,
            output_dir=output_dir,
            skip_errors=args.skip_errors,
        )
        all_detail.append(detail)
        all_metrics.append(metrics)

    detail = pd.concat(all_detail, ignore_index=True)
    metrics = pd.concat(all_metrics, ignore_index=True)
    summary = summarize_metrics(metrics)

    summary_dir = ensure_dir(Path(args.summary_dir))
    detail_path = summary_dir / "ms_cocoai_to_ishu_combined_v3_native_tiling_detail.csv"
    metrics_path = summary_dir / "ms_cocoai_to_ishu_combined_v3_native_tiling_seed_metrics.csv"
    summary_path = summary_dir / "ms_cocoai_to_ishu_combined_v3_native_tiling_summary.csv"
    detail.to_csv(detail_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_report(summary, metrics, args)

    print(detail_path.resolve())
    print(metrics_path.resolve())
    print(summary_path.resolve())
    print(Path(args.report_path).resolve())


if __name__ == "__main__":
    main()
