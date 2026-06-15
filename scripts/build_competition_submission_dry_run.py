from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from lint_competition_submission import lint_submission
from package_competition_submission import package_submission


DEFAULT_RUN_DATE = date.today()
DEFAULT_SOURCE = Path("reports/assets/ms_cocoai_to_ishu_combined_v3_native_tiling_detail.csv")
DEFAULT_OUT_DIR = Path("reports/assets/competition_dry_run")
DEFAULT_REPORT = Path("reports/competition_submission_dry_run_2026_06_15.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a small checked-in competition-submission dry run from an existing score table."
    )
    parser.add_argument("--source-detail", default=str(DEFAULT_SOURCE), help="Per-image score CSV to sample.")
    parser.add_argument("--score-column", default="tile_max_score", help="Score column to package as fake_score.")
    parser.add_argument("--seed", type=int, default=7, help="Seed slice to use when the source has a seed column.")
    parser.add_argument("--limit", type=int, default=120, help="Maximum rows to include in the dry-run bundle.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Directory for generated CSV/JSON assets.")
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT), help="Markdown report to write.")
    parser.add_argument(
        "--run-date",
        default=DEFAULT_RUN_DATE.isoformat(),
        help="Date to stamp into the report, in YYYY-MM-DD format. Defaults to today's local date.",
    )
    return parser.parse_args()


def _slug(value: object) -> str:
    text = str(value).strip().lower()
    chars = [char if char.isalnum() else "_" for char in text]
    slug = "".join(chars).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "image"


def _balanced_sample(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    if limit <= 0 or len(frame) <= limit:
        return frame.copy()
    if "class_name" not in frame.columns:
        return frame.head(limit).copy()
    groups = list(frame.groupby("class_name", sort=True))
    if not groups:
        return frame.head(limit).copy()
    per_group = max(1, limit // len(groups))
    sampled_indices: list[int] = []
    for _name, group in groups:
        sampled_indices.extend(group.head(per_group).index.tolist())
    if len(sampled_indices) < limit:
        remaining = frame.loc[~frame.index.isin(sampled_indices)].head(limit - len(sampled_indices))
        sampled_indices.extend(remaining.index.tolist())
    return frame.loc[sampled_indices].head(limit).copy()


def _image_ids(frame: pd.DataFrame) -> list[str]:
    ids: list[str] = []
    counts: dict[str, int] = {}
    for row in frame.itertuples(index=False):
        path_value = getattr(row, "path", "")
        path = Path(str(path_value))
        class_value = getattr(row, "class_name", path.parent.name)
        base = f"{_slug(class_value)}_{_slug(path.stem)}"
        seen = counts.get(base, 0)
        counts[base] = seen + 1
        ids.append(base if seen == 0 else f"{base}_{seen + 1}")
    return ids


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame[columns].itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def build_dry_run(
    source_detail: Path,
    score_column: str,
    seed: int,
    limit: int,
    out_dir: Path,
    report_path: Path,
    run_date: date = DEFAULT_RUN_DATE,
) -> str:
    source = pd.read_csv(source_detail)
    required = {"path", score_column}
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"Missing required source columns: {missing}")
    if "seed" in source.columns:
        source = source[source["seed"] == seed].copy()
    if source.empty:
        raise ValueError("No rows available for the requested dry-run slice")
    source = source.sort_values([column for column in ["class_name", "path"] if column in source.columns])
    source = _balanced_sample(source, limit)

    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "predictions.csv"
    expected_ids_path = out_dir / "expected_ids.csv"
    submission_path = out_dir / "submission.csv"
    package_manifest_path = out_dir / "submission_manifest.json"
    lint_manifest_path = out_dir / "submission_lint.json"

    predictions = pd.DataFrame(
        {
            "image_id": _image_ids(source),
            "fake_score": pd.to_numeric(source[score_column], errors="raise").astype(float).tolist(),
        }
    )
    if "y_true" in source.columns:
        predictions["y_true"] = source["y_true"].astype(int).tolist()
    predictions.to_csv(predictions_path, index=False)
    predictions[["image_id"]].to_csv(expected_ids_path, index=False)

    _submission, package_manifest = package_submission(
        predictions_path=predictions_path,
        out_path=submission_path,
        manifest_path=package_manifest_path,
        id_column="image_id",
        score_column="fake_score",
        sort_by_id=True,
    )
    passed, lint_manifest = lint_submission(
        submission_path=submission_path,
        manifest_path=lint_manifest_path,
        expected_ids_path=expected_ids_path,
        expected_id_column="image_id",
    )
    if not passed:
        raise ValueError("Dry-run submission lint failed")

    summary = pd.DataFrame(
        [
            {"asset": "predictions", "path": predictions_path.as_posix(), "rows": len(predictions)},
            {"asset": "expected ids", "path": expected_ids_path.as_posix(), "rows": len(predictions)},
            {"asset": "submission", "path": submission_path.as_posix(), "rows": package_manifest["n_rows"]},
            {"asset": "package manifest", "path": package_manifest_path.as_posix(), "rows": 1},
            {"asset": "lint manifest", "path": lint_manifest_path.as_posix(), "rows": 1},
        ]
    )
    lint_checks = pd.DataFrame(lint_manifest["checks"])[["check", "passed", "detail"]]
    lines = [
        "# Competition Submission Dry Run",
        "",
        f"Run date: {run_date.isoformat()}",
        "",
        "Generated by `scripts/build_competition_submission_dry_run.py` from a checked-in per-image score table.",
        "",
        "This is a packaging smoke test, not an official NTIRE/ImageCLEF submission. It proves that an existing score table can be converted into an upload-shaped CSV, that label-like columns are excluded from the submission, and that expected IDs match before upload.",
        "",
        "## Source Slice",
        "",
        f"- Source detail: `{source_detail.as_posix()}`",
        f"- Score column: `{score_column}`",
        f"- Seed: `{seed}`",
        f"- Rows: `{len(predictions)}`",
        f"- Score range: `{float(predictions['fake_score'].min()):.6f}` to `{float(predictions['fake_score'].max()):.6f}`",
        f"- y_true in prediction fixture and excluded from submission: `{bool(package_manifest['y_true_present_excluded'])}`",
        "",
        "## Generated Assets",
        "",
        _markdown_table(summary, ["asset", "path", "rows"]),
        "",
        "## Lint Checks",
        "",
        _markdown_table(lint_checks, ["check", "passed", "detail"]),
        "",
    ]
    text = "\n".join(lines)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    args = parse_args()
    build_dry_run(
        source_detail=Path(args.source_detail),
        score_column=args.score_column,
        seed=args.seed,
        limit=args.limit,
        out_dir=Path(args.out_dir),
        report_path=Path(args.report_path),
        run_date=date.fromisoformat(args.run_date),
    )
    print(Path(args.report_path).resolve())
    print(Path(args.out_dir).resolve())


if __name__ == "__main__":
    main()
