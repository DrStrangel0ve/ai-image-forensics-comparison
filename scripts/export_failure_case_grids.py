from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

from forensic_compare.utils import ensure_dir
from scripts.summarize_source_holdout import _metadata_frame, _prediction_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export qualitative false-positive, false-negative, and disagreement grids."
    )
    parser.add_argument("--metadata", required=True, help="metadata.csv from export_hf_image_dataset.py.")
    parser.add_argument(
        "--predictions",
        action="append",
        required=True,
        help="Prediction in METHOD=PATH form. Repeat for multiple methods.",
    )
    parser.add_argument("--primary-method", required=True)
    parser.add_argument("--data-dir", default=None, help="Dataset root for reconstructing missing paths.")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--tile-size", type=int, default=224)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--min-disagreement", type=float, default=0.4)
    return parser.parse_args()


def _parse_prediction_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.parent.name or path.stem, path
    method, path = value.split("=", 1)
    if not method or not path:
        raise ValueError(f"Prediction arguments must be METHOD=PATH, got {value!r}")
    return method, Path(path)


def _resolve_image_path(value: str | Path) -> Path:
    path = Path(value)
    if path.exists():
        return path
    candidate = ROOT / path
    if candidate.exists():
        return candidate
    return path


def _repo_relative_text(value) -> str:
    text = str(value)
    normalized = text.replace("\\", "/")
    root = str(ROOT.resolve()).replace("\\", "/")
    if normalized.lower().startswith(root.lower() + "/"):
        return normalized[len(root) + 1 :]
    return normalized


def _export_frame(frame: pd.DataFrame) -> pd.DataFrame:
    exported = frame.copy()
    for column in [
        "path_key",
        "path_metadata",
        "prediction_path",
        "image_path",
    ]:
        if column in exported.columns:
            exported[column] = exported[column].map(_repo_relative_text)
    return exported


def _load_joined_predictions(
    metadata_path: Path,
    prediction_args: list[str],
    data_dir: str | None,
    split: str,
) -> tuple[pd.DataFrame, list[str]]:
    metadata = _metadata_frame(metadata_path, split)
    prediction_frames = []
    methods = []
    for method, path in map(_parse_prediction_arg, prediction_args):
        methods.append(method)
        prediction_frames.append(_prediction_frame(path, method, data_dir, split))
    predictions = pd.concat(prediction_frames, ignore_index=True)
    joined = predictions.merge(metadata, on="path_key", suffixes=("_prediction", "_metadata"))
    if len(joined) != len(predictions):
        raise ValueError(f"Matched {len(joined)} prediction rows but expected {len(predictions)}")
    label_mismatches = joined[joined["y_true"].astype(int) != joined["label"].astype(int)]
    if not label_mismatches.empty:
        raise ValueError(
            f"Prediction labels disagree with metadata labels for {len(label_mismatches)} rows"
        )

    score_wide = joined.pivot_table(
        index="path_key", columns="method", values="fake_score", aggfunc="first"
    )
    path_wide = joined.pivot_table(
        index="path_key", columns="method", values="path_prediction", aggfunc="first"
    )
    metadata_columns = [
        column
        for column in ["path_key", "path_metadata", "label", "source_label", "source_key", "source_name", "caption"]
        if column in joined.columns
    ]
    base = joined[metadata_columns].drop_duplicates("path_key").set_index("path_key")
    wide = base.join(score_wide, how="inner").reset_index()
    first_paths = path_wide[methods[0]].rename("prediction_path")
    wide = wide.merge(first_paths, on="path_key", how="left")
    path_source = "prediction_path" if "prediction_path" in wide.columns else "path_metadata"
    wide["image_path"] = wide[path_source].map(lambda value: str(_resolve_image_path(value)))
    return wide, methods


def _select_cases(
    frame: pd.DataFrame,
    methods: list[str],
    primary_method: str,
    threshold: float,
    top_k: int,
    min_disagreement: float,
) -> dict[str, pd.DataFrame]:
    if primary_method not in methods:
        raise ValueError(f"Primary method {primary_method!r} not found in predictions: {methods}")
    selected = frame.copy()
    selected["primary_score"] = selected[primary_method].astype(float)
    selected["primary_predicted_fake"] = selected["primary_score"] >= threshold
    selected["score_min"] = selected[methods].min(axis=1)
    selected["score_max"] = selected[methods].max(axis=1)
    selected["score_spread"] = selected["score_max"] - selected["score_min"]
    selected["lowest_method"] = selected[methods].idxmin(axis=1)
    selected["highest_method"] = selected[methods].idxmax(axis=1)

    false_positives = (
        selected[(selected["label"].astype(int) == 0) & selected["primary_predicted_fake"]]
        .sort_values(["primary_score", "score_spread"], ascending=[False, False])
        .head(top_k)
    )
    false_negatives = (
        selected[(selected["label"].astype(int) == 1) & ~selected["primary_predicted_fake"]]
        .sort_values(["primary_score", "score_spread"], ascending=[True, False])
        .head(top_k)
    )
    disagreements = (
        selected[selected["score_spread"] >= min_disagreement]
        .sort_values("score_spread", ascending=False)
        .head(top_k)
    )
    return {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "disagreements": disagreements,
    }


def _font(size: int) -> ImageFont.ImageFont:
    for name in ["arial.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _wrap_text(text: str, font: ImageFont.ImageFont, max_width: int, max_lines: int) -> list[str]:
    words = str(text).replace("\n", " ").split()
    lines: list[str] = []
    current = ""
    probe_image = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(probe_image)
    for word in words:
        candidate = word if not current else f"{current} {word}"
        width = draw.textbbox((0, 0), candidate, font=font)[2]
        if width <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
        if len(lines) >= max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(current)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
    return lines


def _caption(row: pd.Series, methods: list[str], primary_method: str) -> str:
    label = "fake" if int(row["label"]) == 1 else "real"
    source = row.get("source_name", "")
    score_parts = [f"{method}:{float(row[method]):.2f}" for method in methods]
    headline = f"{label} / {source} / {primary_method}:{float(row[primary_method]):.2f}"
    return f"{headline} | " + " ".join(score_parts)


def _make_grid(
    rows: pd.DataFrame,
    methods: list[str],
    primary_method: str,
    title: str,
    out_path: Path,
    tile_size: int,
    columns: int,
) -> None:
    if rows.empty:
        image = Image.new("RGB", (tile_size * columns, 120), "white")
        draw = ImageDraw.Draw(image)
        draw.text((12, 12), f"{title}: no cases", fill="black", font=_font(18))
        image.save(out_path)
        return

    title_font = _font(20)
    caption_font = _font(12)
    caption_height = 86
    title_height = 44
    rows_count = int((len(rows) + columns - 1) / columns)
    width = columns * tile_size
    height = title_height + rows_count * (tile_size + caption_height)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), title, fill="black", font=title_font)

    for index, (_, row) in enumerate(rows.iterrows()):
        grid_x = (index % columns) * tile_size
        grid_y = title_height + (index // columns) * (tile_size + caption_height)
        try:
            image = Image.open(row["image_path"]).convert("RGB")
            image = ImageOps.exif_transpose(image)
            image.thumbnail((tile_size, tile_size), Image.Resampling.LANCZOS)
            tile = Image.new("RGB", (tile_size, tile_size), "#f2f2f2")
            tile.paste(image, ((tile_size - image.width) // 2, (tile_size - image.height) // 2))
        except (OSError, FileNotFoundError):
            tile = Image.new("RGB", (tile_size, tile_size), "#eeeeee")
            tile_draw = ImageDraw.Draw(tile)
            tile_draw.text((8, 8), "missing image", fill="black", font=caption_font)
        canvas.paste(tile, (grid_x, grid_y))
        caption = _caption(row, methods, primary_method)
        for line_index, line in enumerate(_wrap_text(caption, caption_font, tile_size - 10, 4)):
            draw.text(
                (grid_x + 5, grid_y + tile_size + 5 + line_index * 16),
                line,
                fill="black",
                font=caption_font,
            )
    canvas.save(out_path)


def _write_markdown_report(
    out_dir: Path,
    primary_method: str,
    threshold: float,
    cases: dict[str, pd.DataFrame],
    methods: list[str],
) -> None:
    def line(name: str) -> str:
        return f"- `{name}`: {len(cases[name])} cases, grid `{name}_grid.png`, CSV `{name}.csv`"

    report = [
        "# Qualitative Failure Cases",
        "",
        f"Primary method: `{primary_method}`",
        f"Threshold: `{threshold:.2f}`",
        f"Compared methods: {', '.join(f'`{method}`' for method in methods)}",
        "",
        line("false_positives"),
        line("false_negatives"),
        line("disagreements"),
        "",
        "## Grids",
        "",
        "![False positives](false_positives_grid.png)",
        "",
        "![False negatives](false_negatives_grid.png)",
        "",
        "![Model disagreements](disagreements_grid.png)",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1")
    if args.columns < 1:
        raise ValueError("--columns must be at least 1")
    out_dir = ensure_dir(args.out_dir)
    joined, methods = _load_joined_predictions(
        Path(args.metadata),
        args.predictions,
        args.data_dir,
        args.split,
    )
    cases = _select_cases(
        joined,
        methods,
        args.primary_method,
        args.threshold,
        args.top_k,
        args.min_disagreement,
    )
    _export_frame(joined).to_csv(out_dir / "joined_predictions_wide.csv", index=False)
    for name, frame in cases.items():
        _export_frame(frame).to_csv(out_dir / f"{name}.csv", index=False)
        _make_grid(
            frame,
            methods,
            args.primary_method,
            name.replace("_", " ").title(),
            out_dir / f"{name}_grid.png",
            args.tile_size,
            args.columns,
        )

    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["artifact", "path"])
        for artifact in [
            "joined_predictions_wide.csv",
            "false_positives.csv",
            "false_positives_grid.png",
            "false_negatives.csv",
            "false_negatives_grid.png",
            "disagreements.csv",
            "disagreements_grid.png",
            "report.md",
        ]:
            writer.writerow([artifact, str((out_dir / artifact).resolve())])
    _write_markdown_report(out_dir, args.primary_method, args.threshold, cases, methods)
    print(f"Wrote qualitative failure grids to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
