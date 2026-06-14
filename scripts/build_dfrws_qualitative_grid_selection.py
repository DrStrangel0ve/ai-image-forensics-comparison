from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image


RUN_DATE = "2026-06-14"

MODEL_SCORE_COLUMNS = [
    "combined_v3",
    "convnext_tiny_frozen",
    "physics_guided",
    "resnet18",
    "scp_fusion_v0",
]

CANDIDATES = [
    {
        "seed": 17,
        "image_path": "reports/assets/qualitative_seed17_scp_fusion_false_negatives.png",
        "csv_path": "reports/assets/qualitative_seed17_scp_fusion_false_negatives.csv",
        "recommendation": "appendix_backup",
        "poster_readability_score": 3.5,
        "artifact_story_score": 3.5,
        "visual_rationale": (
            "Good confidence-miss evidence, but several examples are visually subtle at poster scale "
            "and the panel reads more like broad error sampling than a single forensic story."
        ),
    },
    {
        "seed": 29,
        "image_path": "reports/assets/qualitative_seed29_scp_fusion_false_negatives.png",
        "csv_path": "reports/assets/qualitative_seed29_scp_fusion_false_negatives.csv",
        "recommendation": "selected_for_dfrws_poster",
        "poster_readability_score": 5.0,
        "artifact_story_score": 5.0,
        "visual_rationale": (
            "Best poster panel: readable text/sign failures, object-label oddities, compositing artifacts, "
            "and repeated low fake scores across source families make the failure mode obvious quickly."
        ),
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select the qualitative false-negative grid to use in the DFRWS poster package."
    )
    parser.add_argument("--repo-root", default=".", help="Repository root for resolving artifact paths.")
    parser.add_argument(
        "--out-path",
        default="reports/dfrws_qualitative_grid_selection_2026_06_14.md",
        help="Markdown selection report to write.",
    )
    parser.add_argument(
        "--csv-out",
        default="reports/assets/dfrws_qualitative_grid_selection.csv",
        help="Machine-readable candidate audit to write.",
    )
    return parser.parse_args()


def _source_counts(frame: pd.DataFrame) -> str:
    counts = frame["source_name"].value_counts().sort_index()
    return ", ".join(f"{source}={count}" for source, count in counts.items())


def _format_float(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.4f}"


def _markdown_escape(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).replace("\n", " ").replace("|", "\\|")


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame[columns].itertuples(index=False):
        lines.append("| " + " | ".join(_markdown_escape(value) for value in row) + " |")
    return "\n".join(lines)


def _candidate_row(repo_root: Path, candidate: dict[str, object]) -> dict[str, object]:
    image_path = repo_root / str(candidate["image_path"])
    csv_path = repo_root / str(candidate["csv_path"])
    frame = pd.read_csv(csv_path)
    score_frame = frame[MODEL_SCORE_COLUMNS]
    with Image.open(image_path) as image:
        width, height = image.size

    source_coverage_score = min(5.0, frame["source_name"].nunique() / 4.0 * 5.0)
    poster_score = (
        0.45 * float(candidate["poster_readability_score"])
        + 0.35 * float(candidate["artifact_story_score"])
        + 0.20 * source_coverage_score
    )
    return {
        "seed": candidate["seed"],
        "recommendation": candidate["recommendation"],
        "image_path": candidate["image_path"],
        "csv_path": candidate["csv_path"],
        "rows": len(frame),
        "sources": frame["source_name"].nunique(),
        "source_counts": _source_counts(frame),
        "image_width": width,
        "image_height": height,
        "mean_scp_fusion_score": frame["scp_fusion_v0"].mean(),
        "median_scp_fusion_score": frame["scp_fusion_v0"].median(),
        "max_scp_fusion_score": frame["scp_fusion_v0"].max(),
        "mean_branch_score": score_frame.mean(axis=1).mean(),
        "mean_score_spread": frame["score_spread"].mean(),
        "poster_readability_score": candidate["poster_readability_score"],
        "artifact_story_score": candidate["artifact_story_score"],
        "source_coverage_score": source_coverage_score,
        "poster_selection_score": poster_score,
        "visual_rationale": candidate["visual_rationale"],
    }


def build_selection_report(repo_root: Path) -> tuple[str, pd.DataFrame]:
    rows = [_candidate_row(repo_root, candidate) for candidate in CANDIDATES]
    audit = pd.DataFrame(rows).sort_values(
        ["poster_selection_score", "seed"], ascending=[False, True]
    )
    selected = audit.iloc[0]
    display = audit.copy()
    for column in [
        "mean_scp_fusion_score",
        "median_scp_fusion_score",
        "max_scp_fusion_score",
        "mean_branch_score",
        "mean_score_spread",
        "poster_readability_score",
        "artifact_story_score",
        "source_coverage_score",
        "poster_selection_score",
    ]:
        display[column] = display[column].map(_format_float)

    lines = [
        "# DFRWS Qualitative Grid Selection",
        "",
        f"Run date: {RUN_DATE}",
        "",
        (
            f"Decision: use the seed-{selected['seed']} SCP-Fusion false-negative grid as the "
            "DFRWS poster qualitative panel; keep the other grid as appendix or backup material."
        ),
        "",
        "This selection intentionally weights poster readability and forensic story clarity ahead of raw miss confidence. "
        "The seed-17 examples have slightly lower SCP-Fusion scores, but the seed-29 panel communicates the failure mode faster.",
        "",
        "## Candidate Audit",
        "",
        _markdown_table(
            display,
            [
                "seed",
                "recommendation",
                "rows",
                "sources",
                "source_counts",
                "mean_scp_fusion_score",
                "mean_score_spread",
                "poster_readability_score",
                "artifact_story_score",
                "poster_selection_score",
            ],
        ),
        "",
        "## Poster Use",
        "",
        f"- Selected panel: `{selected['image_path']}`.",
        "- Caption angle: source-heldout fusion can miss generated images even when visible text, signage, object labels, or compositing cues look suspicious to a human reviewer.",
        "- Keep seed 17 as a reproducibility/appendix panel because it shows the same failure class under a different split seed.",
        "- Do not imply these are representative of all misses; they are illustrative examples selected from checked-in false-negative manifests.",
        "",
        "## Candidate Notes",
        "",
    ]
    for row in audit.itertuples(index=False):
        lines.append(f"- Seed {row.seed}: {row.visual_rationale}")
    lines.append("")
    return "\n".join(lines), audit


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    text, audit = build_selection_report(repo_root)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")

    csv_out = Path(args.csv_out)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(csv_out, index=False)

    print(out_path.resolve())
    print(csv_out.resolve())


if __name__ == "__main__":
    main()
