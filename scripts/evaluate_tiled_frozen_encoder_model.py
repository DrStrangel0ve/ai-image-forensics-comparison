from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from forensic_compare.datasets import collect_labeled_images, discover_layout, limit_records, stratified_split
from forensic_compare.foundation import build_frozen_encoder, encode_batch, frozen_encoder_transform
from forensic_compare.metrics import binary_metrics
from forensic_compare.utils import ensure_dir, read_json, resolve_device


SCORE_MODES = ["global", "tile_mean", "tile_max", "tile_top2_mean"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a saved frozen-encoder classifier with native-image tile aggregation. "
            "The saved classifier is unchanged; only target scoring is tiled."
        )
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 29])
    parser.add_argument(
        "--model-template",
        default="runs/ms_cocoai_to_ishu_foundation/clip_vit_b_32_seed{seed}",
        help="Template containing classifier.joblib and metrics.json for each seed.",
    )
    parser.add_argument(
        "--target-template",
        default="data/raw/ishu_ai_vs_real_2026_seed{seed}_test",
        help="Template for each seed-specific target image-folder dataset.",
    )
    parser.add_argument("--target-split", choices=["all", "test"], default="all")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-target-samples", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default="runs/tiled_frozen_encoder_eval",
    )
    parser.add_argument("--summary-dir", default="reports/assets")
    parser.add_argument(
        "--report-path",
        default="reports/tiled_frozen_encoder_eval_2026_06_14.md",
    )
    return parser.parse_args()


def _target_records(
    target_dir: Path,
    target_split: str,
    val_fraction: float,
    seed: int,
    max_target_samples: int,
):
    layout = discover_layout(target_dir)
    if target_split == "test" and layout.test:
        records = collect_labeled_images(layout.test)
    elif layout.single:
        records = collect_labeled_images(layout.single)
        if target_split == "test":
            _train, records = stratified_split(records, val_fraction, seed)
    elif layout.train and layout.test:
        if target_split == "all":
            records = collect_labeled_images(layout.train) + collect_labeled_images(layout.test)
        else:
            records = collect_labeled_images(layout.test)
    else:
        raise ValueError(f"Unsupported dataset layout: {layout}")
    return limit_records(records, max_target_samples, seed)


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
    tile_scores = np.asarray(tile_scores, dtype=float)
    if len(tile_scores) == 0:
        tile_scores = np.asarray([global_score], dtype=float)
    sorted_scores = np.sort(tile_scores)
    top_k = sorted_scores[-min(2, len(sorted_scores)) :]
    return {
        "global": float(global_score),
        "tile_mean": float(tile_scores.mean()),
        "tile_max": float(tile_scores.max()),
        "tile_top2_mean": float(top_k.mean()),
        "tile_std": float(tile_scores.std()),
    }


def _score_embeddings(classifier, embeddings: np.ndarray) -> np.ndarray:
    if hasattr(classifier, "predict_proba"):
        return np.asarray(classifier.predict_proba(embeddings)[:, 1], dtype=float)
    raw_scores = classifier.decision_function(embeddings)
    return np.asarray(1.0 / (1.0 + np.exp(-raw_scores)), dtype=float)


def patch_sklearn_predict_proba_compat(estimator) -> None:
    """Patch known sklearn joblib compatibility gaps after loading older/newer objects."""

    seen: set[int] = set()

    def visit(node) -> None:
        node_id = id(node)
        if node_id in seen:
            return
        seen.add(node_id)
        if type(node).__name__ == "LogisticRegression" and not hasattr(node, "multi_class"):
            node.multi_class = "auto"
        for _name, child in getattr(node, "steps", []):
            visit(child)
        for child in getattr(node, "estimators_", []):
            visit(child)

    visit(estimator)


@torch.no_grad()
def _encode_pil_images(images: list[Image.Image], encoder, transform, device: torch.device, batch_size: int) -> np.ndarray:
    features = []
    encoder = encoder.to(device)
    for start in range(0, len(images), batch_size):
        batch_images = images[start : start + batch_size]
        tensor = torch.stack([transform(image) for image in batch_images], dim=0).to(device)
        embeddings = encode_batch(encoder, tensor).detach().cpu().numpy()
        features.append(embeddings)
    return np.vstack(features)


def _score_image(
    path: Path,
    encoder,
    classifier,
    transform,
    device: torch.device,
    batch_size: int,
    tile_size: int,
) -> dict[str, float | int]:
    with Image.open(path) as image:
        image = image.convert("RGB")
        width, height = image.size
        boxes = tile_boxes(width, height, tile_size)
        crops = [image.copy(), *[image.crop(box).copy() for box in boxes]]

    embeddings = _encode_pil_images(crops, encoder, transform, device, batch_size)
    scores = _score_embeddings(classifier, embeddings)
    global_score = float(scores[0])
    tile_scores = np.asarray(scores[1:], dtype=float)
    aggregated = aggregate_scores(global_score, tile_scores)
    return {
        "width": int(width),
        "height": int(height),
        "n_tiles": int(len(tile_scores)),
        **{f"{key}_score": value for key, value in aggregated.items()},
        "tile_max_minus_global": float(aggregated["tile_max"] - global_score),
    }


def evaluate_seed(
    seed: int,
    model_dir: Path,
    target_dir: Path,
    target_split: str,
    val_fraction: float,
    tile_size: int,
    batch_size: int,
    device_name: str,
    max_target_samples: int,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_metrics = read_json(model_dir / "metrics.json")
    encoder_name = str(source_metrics["encoder"])
    pretrained = bool(source_metrics.get("pretrained", False))
    spec = build_frozen_encoder(encoder_name, pretrained=pretrained)
    transform = frozen_encoder_transform(spec.image_size, spec.mean, spec.std)
    device = resolve_device(device_name)
    classifier = joblib.load(model_dir / "classifier.joblib")
    patch_sklearn_predict_proba_compat(classifier)
    records = _target_records(target_dir, target_split, val_fraction, seed, max_target_samples)

    rows = []
    for path, label, class_name in tqdm(records, desc=f"seed{seed}/tiled-{encoder_name}"):
        scored = _score_image(
            path=path,
            encoder=spec.model,
            classifier=classifier,
            transform=transform,
            device=device,
            batch_size=batch_size,
            tile_size=tile_size,
        )
        rows.append(
            {
                "seed": seed,
                "path": str(path),
                "class_name": class_name,
                "y_true": int(label),
                "encoder": encoder_name,
                "pretrained": pretrained,
                "classifier": source_metrics.get("classifier"),
                "image_size": int(spec.image_size),
                "tile_size": int(tile_size),
                **scored,
            }
        )

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
                "encoder": encoder_name,
                "pretrained": pretrained,
                "classifier": source_metrics.get("classifier"),
                "image_size": int(spec.image_size),
                "tile_size": int(tile_size),
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
    return pd.DataFrame(rows).sort_values("score_mode", key=lambda series: series.map(order)).reset_index(drop=True)


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
        "# Tiled Frozen-Encoder Evaluation",
        "",
        "This report evaluates a saved frozen-encoder classifier with deterministic native-image tile aggregation. The encoder and classifier are unchanged; target images are scored as a global resized view plus native crops.",
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
        "Treat this as a foundation-branch diagnostic until the tiled scores are folded into source-heldout SCP-Fusion.",
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
        ".\\.venv\\Scripts\\python.exe scripts\\evaluate_tiled_frozen_encoder_model.py --model-template <run_template> --target-template <target_template>",
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
            tile_size=args.tile_size,
            batch_size=args.batch_size,
            device_name=args.device,
            max_target_samples=args.max_target_samples,
            output_dir=output_dir,
        )
        all_detail.append(detail)
        all_metrics.append(metrics)

    detail = pd.concat(all_detail, ignore_index=True)
    metrics = pd.concat(all_metrics, ignore_index=True)
    summary = summarize_metrics(metrics)

    summary_dir = ensure_dir(Path(args.summary_dir))
    detail_path = summary_dir / "tiled_frozen_encoder_detail.csv"
    metrics_path = summary_dir / "tiled_frozen_encoder_seed_metrics.csv"
    summary_path = summary_dir / "tiled_frozen_encoder_summary.csv"
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
