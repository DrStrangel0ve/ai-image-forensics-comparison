from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


RUN_DATE = "2026-06-14"
CLEAN_ID = "ms_to_ishu_tuned_fusion_constraint_sweep_best"
TRANSFORM_ROWS = [
    ("jpeg70", "ms_to_ishu_tuned_fusion_jpeg70"),
    ("jpeg50", "ms_to_ishu_tuned_fusion_jpeg50"),
    ("jpeg30", "ms_to_ishu_tuned_fusion_jpeg30"),
    ("blur1", "ms_to_ishu_tuned_fusion_blur1"),
    ("resize_half", "ms_to_ishu_tuned_fusion_resize_half"),
    ("crop85", "ms_to_ishu_tuned_fusion_crop85"),
    ("noise3", "ms_to_ishu_tuned_fusion_noise3"),
    ("screenshot", "ms_to_ishu_tuned_fusion_screenshot"),
    ("social_square", "ms_to_ishu_tuned_fusion_social_square"),
    ("social_720p", "ms_to_ishu_tuned_fusion_social_720p"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank target-transform robustness failures for the reverse tuned-fusion SCP-Fusion setting."
    )
    parser.add_argument(
        "--core-results",
        default="reports/assets/publication_core_results.csv",
        help="Canonical publication result table.",
    )
    parser.add_argument(
        "--out-path",
        default="reports/robustness_failure_ranking_2026_06_14.md",
        help="Markdown robustness ranking report to write.",
    )
    parser.add_argument(
        "--csv-out",
        default="reports/assets/robustness_failure_ranking.csv",
        help="Machine-readable robustness ranking CSV to write.",
    )
    return parser.parse_args()


def _stress_tier(delta_accuracy: float, delta_auc: float) -> str:
    worst = min(delta_accuracy, delta_auc)
    if worst <= -0.045:
        return "major_drop"
    if worst <= -0.02:
        return "moderate_drop"
    if delta_accuracy >= 0.005 and delta_auc >= 0.005:
        return "apparent_gain"
    return "stable_or_mild"


def _interpretation(transform: str, row: pd.Series) -> str:
    tier = row["stress_tier"] if isinstance(row, pd.Series) else row.stress_tier
    if transform == "jpeg30":
        return "Harsh JPEG is the largest accuracy stressor and shifts the fake-call rate downward."
    if transform == "blur1":
        return "Blur is a joint ranking and calibration stressor, with the worst Brier/ECE in this table."
    if transform == "resize_half":
        return "Half-resolution roundtrip is the largest AUC stressor and should be a main failure-mode example."
    if transform == "screenshot":
        return "Screenshot-style resampling hurts ranking more than default accuracy."
    if transform.startswith("social"):
        return "Social-media-like processing is comparatively survivable under this source-selected policy."
    if transform == "noise3":
        return "Mild noise is not a failure here; it improves ranking on this bounded proxy split."
    if tier == "moderate_drop":
        return "Meaningful but not catastrophic drop versus the clean source-selected anchor."
    if tier == "apparent_gain":
        return "Apparent gain on this proxy split; do not overinterpret without more transforms."
    return "Stable or mild shift versus the clean source-selected anchor."


def _require_row(core: pd.DataFrame, finding_id: str) -> pd.Series:
    matches = core[core["finding_id"] == finding_id]
    if matches.empty:
        raise ValueError(f"Missing finding_id={finding_id!r}")
    return matches.iloc[0]


def build_failure_ranking(core_results_path: Path) -> tuple[str, pd.DataFrame]:
    core = pd.read_csv(core_results_path)
    clean = _require_row(core, CLEAN_ID)
    rows = []
    for transform, finding_id in TRANSFORM_ROWS:
        result = _require_row(core, finding_id)
        delta_accuracy = float(result["accuracy"]) - float(clean["accuracy"])
        delta_auc = float(result["auc"]) - float(clean["auc"])
        rows.append(
            {
                "transform": transform,
                "finding_id": finding_id,
                "method": result["method"],
                "accuracy": float(result["accuracy"]),
                "delta_accuracy_vs_clean": delta_accuracy,
                "auc": float(result["auc"]),
                "delta_auc_vs_clean": delta_auc,
                "brier": float(result["brier"]),
                "ece": float(result["ece"]),
                "fake_call_rate": float(result["predicted_fake_rate"]),
                "stress_tier": _stress_tier(delta_accuracy, delta_auc),
                "source": result["source"],
            }
        )
    ranking = pd.DataFrame(rows)
    ranking["interpretation"] = [
        _interpretation(row.transform, row) for row in ranking.itertuples(index=False)
    ]
    ranking = ranking.sort_values(
        ["delta_auc_vs_clean", "delta_accuracy_vs_clean", "transform"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    ranking.insert(0, "rank_by_auc_drop", range(1, len(ranking) + 1))
    text = _write_markdown(clean, ranking)
    return text, ranking


def _format_float(value: float) -> str:
    return f"{value:.4f}"


def _markdown_escape(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float):
        return _format_float(value)
    return str(value).replace("\n", " ").replace("|", "\\|")


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame[columns].itertuples(index=False):
        lines.append("| " + " | ".join(_markdown_escape(value) for value in row) + " |")
    return "\n".join(lines)


def _write_markdown(clean: pd.Series, ranking: pd.DataFrame) -> str:
    worst_accuracy = ranking.sort_values("delta_accuracy_vs_clean").iloc[0]
    worst_auc = ranking.iloc[0]
    tier_summary = ranking.groupby("stress_tier", sort=False).size().reset_index(name="count")
    lines = [
        "# Robustness Failure Ranking",
        "",
        f"Run date: {RUN_DATE}",
        "",
        "Generated by `scripts/build_robustness_failure_ranking.py` from `reports/assets/publication_core_results.csv`.",
        "",
        "This is a target-transform stress ranking for the reverse tuned-fusion SCP-Fusion operating point. It is proxy robustness evidence, not an official NTIRE/ImageCLEF challenge score.",
        "",
        "## Clean Anchor",
        "",
        (
            f"`{CLEAN_ID}`: accuracy {_format_float(float(clean['accuracy']))}, "
            f"AUC {_format_float(float(clean['auc']))}, Brier {_format_float(float(clean['brier']))}, "
            f"ECE {_format_float(float(clean['ece']))}, fake-call rate {_format_float(float(clean['predicted_fake_rate']))}."
        ),
        "",
        "## Stress Tier Summary",
        "",
        _markdown_table(tier_summary, ["stress_tier", "count"]),
        "",
        "## Ranked By AUC Drop",
        "",
        _markdown_table(
            ranking,
            [
                "rank_by_auc_drop",
                "transform",
                "accuracy",
                "delta_accuracy_vs_clean",
                "auc",
                "delta_auc_vs_clean",
                "brier",
                "ece",
                "fake_call_rate",
                "stress_tier",
                "interpretation",
            ],
        ),
        "",
        "## Paper Takeaways",
        "",
        f"- Largest AUC stressor: `{worst_auc['transform']}` ({_format_float(float(worst_auc['delta_auc_vs_clean']))} AUC delta).",
        f"- Largest accuracy stressor: `{worst_accuracy['transform']}` ({_format_float(float(worst_accuracy['delta_accuracy_vs_clean']))} accuracy delta).",
        "- Blur, half-resolution resize, screenshot-style resampling, and harsh JPEG should be the failure-mode examples in WIFS/DFF.",
        "- Mild noise and social-square processing look survivable on this bounded proxy split, but should not be written as universal robustness.",
        "- Keep the claim phrasing as `source-selected proxy transform stress`, not `robust detector`.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    text, ranking = build_failure_ranking(Path(args.core_results))
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    csv_path = Path(args.csv_out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(csv_path, index=False)
    print(out_path.resolve())
    print(csv_path.resolve())


if __name__ == "__main__":
    main()
