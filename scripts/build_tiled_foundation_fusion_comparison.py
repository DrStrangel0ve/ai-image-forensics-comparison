from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


RUN_DATE = "2026-06-14"

DEFAULT_INPUTS = [
    (
        "combined_v3",
        "reports/assets/ms_cocoai_to_ishu_tuned_fusion_native_tiling_summary.csv",
        "reports/ms_cocoai_to_ishu_tuned_fusion_native_tiling_2026_06_13.md",
    ),
    (
        "clip_vit_b_32",
        "reports/assets/ms_cocoai_to_ishu_tuned_fusion_tiled_clip_summary.csv",
        "reports/ms_cocoai_to_ishu_tuned_fusion_tiled_clip_2026_06_14.md",
    ),
    (
        "dinov2_vits14",
        "reports/assets/ms_cocoai_to_ishu_tuned_fusion_tiled_dinov2_summary.csv",
        "reports/ms_cocoai_to_ishu_tuned_fusion_tiled_dinov2_2026_06_14.md",
    ),
    (
        "convnext_tiny",
        "reports/assets/ms_cocoai_to_ishu_tuned_fusion_tiled_convnext_summary.csv",
        "reports/ms_cocoai_to_ishu_tuned_fusion_tiled_convnext_2026_06_14.md",
    ),
]

DISPLAY_NAMES = {
    "combined_v3": "combined_v3 conventional",
    "clip_vit_b_32": "CLIP ViT-B/32",
    "dinov2_vits14": "DINOv2-small",
    "convnext_tiny": "ConvNeXt-Tiny",
}

METRIC_COLUMNS = [
    "target_accuracy_mean",
    "target_roc_auc_mean",
    "target_brier_score_mean",
    "target_expected_calibration_error_mean",
    "target_predicted_positive_rate_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a comparison of reverse tuned-fusion runs with tiled target branches."
    )
    parser.add_argument(
        "--out-path",
        default="reports/ms_cocoai_to_ishu_tiled_foundation_fusion_comparison_2026_06_14.md",
        help="Markdown report to write.",
    )
    parser.add_argument(
        "--csv-out",
        default="reports/assets/ms_cocoai_to_ishu_tiled_foundation_fusion_comparison.csv",
        help="Machine-readable comparison table to write.",
    )
    return parser.parse_args()


def _read_summary(branch: str, summary_path: Path, report_path: str) -> pd.DataFrame:
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing fused tiled summary for {branch}: {summary_path}")
    frame = pd.read_csv(summary_path)
    required = {"score_mode", "constraint_policy", "n_seeds", *METRIC_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"{summary_path} is missing required columns: {', '.join(sorted(missing))}"
        )
    frame = frame.copy()
    frame.insert(0, "branch", branch)
    frame.insert(1, "branch_label", DISPLAY_NAMES.get(branch, branch))
    frame["source_report"] = report_path
    return frame


def load_comparison() -> pd.DataFrame:
    frames = [
        _read_summary(branch, Path(summary_path), report_path)
        for branch, summary_path, report_path in DEFAULT_INPUTS
    ]
    comparison = pd.concat(frames, ignore_index=True)
    clean = comparison[
        (comparison["branch"].eq("combined_v3")) & (comparison["score_mode"].eq("global"))
    ].iloc[0]
    previous_best = comparison[comparison["branch"].eq("combined_v3")].sort_values(
        ["target_accuracy_mean", "target_roc_auc_mean", "target_predicted_positive_rate_mean"],
        ascending=[False, False, True],
    ).iloc[0]
    for column in METRIC_COLUMNS:
        comparison[f"{column}_delta_vs_clean_global"] = comparison[column] - float(clean[column])
        comparison[f"{column}_delta_vs_previous_tiled_v3"] = (
            comparison[column] - float(previous_best[column])
        )
    return comparison


def _format_value(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("\n", " ").replace("|", "\\|")


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame[columns].itertuples(index=False):
        lines.append("| " + " | ".join(_format_value(value) for value in row) + " |")
    return "\n".join(lines)


def build_report(comparison: pd.DataFrame) -> str:
    best_accuracy = comparison.sort_values(
        ["target_accuracy_mean", "target_roc_auc_mean", "target_predicted_positive_rate_mean"],
        ascending=[False, False, True],
    ).iloc[0]
    best_auc = comparison.sort_values(
        ["target_roc_auc_mean", "target_accuracy_mean", "target_predicted_positive_rate_mean"],
        ascending=[False, False, True],
    ).iloc[0]
    previous_best = comparison[comparison["branch"].eq("combined_v3")].sort_values(
        ["target_accuracy_mean", "target_roc_auc_mean", "target_predicted_positive_rate_mean"],
        ascending=[False, False, True],
    ).iloc[0]

    best_by_branch = []
    for branch, group in comparison.groupby("branch", sort=False):
        acc_row = group.sort_values(
            ["target_accuracy_mean", "target_roc_auc_mean", "target_predicted_positive_rate_mean"],
            ascending=[False, False, True],
        ).iloc[0]
        auc_row = group.sort_values(
            ["target_roc_auc_mean", "target_accuracy_mean", "target_predicted_positive_rate_mean"],
            ascending=[False, False, True],
        ).iloc[0]
        best_by_branch.append(
            {
                "branch_label": acc_row["branch_label"],
                "best_accuracy_mode": acc_row["score_mode"],
                "best_accuracy": acc_row["target_accuracy_mean"],
                "accuracy_delta_vs_clean": acc_row[
                    "target_accuracy_mean_delta_vs_clean_global"
                ],
                "best_auc_mode": auc_row["score_mode"],
                "best_auc": auc_row["target_roc_auc_mean"],
                "auc_delta_vs_clean": auc_row["target_roc_auc_mean_delta_vs_clean_global"],
            }
        )
    best_frame = pd.DataFrame(best_by_branch)

    display_columns = [
        "branch_label",
        "score_mode",
        "target_accuracy_mean",
        "target_accuracy_mean_delta_vs_clean_global",
        "target_accuracy_mean_delta_vs_previous_tiled_v3",
        "target_roc_auc_mean",
        "target_roc_auc_mean_delta_vs_clean_global",
        "target_roc_auc_mean_delta_vs_previous_tiled_v3",
        "target_brier_score_mean",
        "target_expected_calibration_error_mean",
        "target_predicted_positive_rate_mean",
    ]

    lines = [
        "# MS COCOAI to Ishu Tiled Foundation Fusion Comparison",
        "",
        f"Run date: {RUN_DATE}",
        "",
        "This compares reverse tuned-fusion diagnostics after replacing one target branch at a time with native-tiled target scores. Source training rows, selected fusion configurations, and source threshold policy stay fixed.",
        "",
        "## Headline",
        "",
        f"- Previous fused native-tiling frontier: `{previous_best['branch_label']}` with `{previous_best['score_mode']}` at `{previous_best['target_accuracy_mean']:.4f}` accuracy / `{previous_best['target_roc_auc_mean']:.4f}` AUC.",
        f"- Best accuracy-preserving tiled foundation replacement: `{best_accuracy['branch_label']}` with `{best_accuracy['score_mode']}` at `{best_accuracy['target_accuracy_mean']:.4f}` accuracy / `{best_accuracy['target_roc_auc_mean']:.4f}` AUC.",
        f"- Best AUC-only tiled foundation replacement: `{best_auc['branch_label']}` with `{best_auc['score_mode']}` at `{best_auc['target_accuracy_mean']:.4f}` accuracy / `{best_auc['target_roc_auc_mean']:.4f}` AUC.",
        "",
        "The gain is real but small: tiled foundation replacement nudges the reverse fused frontier above the previous tiled conventional branch, but it does not close the official SOTA gap and still needs source-heldout and transform stress checks.",
        "",
        "## Best Mode Per Replaced Branch",
        "",
        _markdown_table(
            best_frame,
            [
                "branch_label",
                "best_accuracy_mode",
                "best_accuracy",
                "accuracy_delta_vs_clean",
                "best_auc_mode",
                "best_auc",
                "auc_delta_vs_clean",
            ],
        ),
        "",
        "## All Score Modes",
        "",
        _markdown_table(comparison, display_columns),
        "",
        "## Interpretation",
        "",
        "- DINOv2 `tile_top2_mean` gives the best accuracy-first fused operating point.",
        "- ConvNeXt `tile_mean` gives the best AUC-only fused operating point but lowers default accuracy relative to the previous tiled-v3 frontier.",
        "- CLIP tiling helps ranking inside fusion, but its strong standalone tiled AUC does not translate into the best fused branch replacement.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    comparison = load_comparison()

    csv_path = Path(args.csv_out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(csv_path, index=False)

    report_path = Path(args.out_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_report(comparison), encoding="utf-8")

    print(report_path.resolve())
    print(csv_path.resolve())


if __name__ == "__main__":
    main()
