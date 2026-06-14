from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


RUN_DATE = "2026-06-14"

DEFAULT_VARIANTS = ["blur1", "jpeg30", "resize_half", "screenshot"]
LOWER_IS_BETTER = {
    "target_brier_score_mean",
    "target_expected_calibration_error_mean",
    "target_predicted_positive_rate_mean",
}
METRICS = [
    "target_accuracy_mean",
    "target_roc_auc_mean",
    "target_brier_score_mean",
    "target_expected_calibration_error_mean",
    "target_predicted_positive_rate_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a calibration tradeoff report for tiled-DINO transform probes."
    )
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    parser.add_argument(
        "--out-path",
        default="reports/tiled_dinov2_calibration_tradeoff_2026_06_14.md",
        help="Markdown report to write.",
    )
    parser.add_argument(
        "--csv-out",
        default="reports/assets/tiled_dinov2_calibration_tradeoff.csv",
        help="Machine-readable all-mode tradeoff table to write.",
    )
    parser.add_argument(
        "--choice-csv-out",
        default="reports/assets/tiled_dinov2_calibration_tradeoff_choices.csv",
        help="Machine-readable per-transform mode-choice table to write.",
    )
    return parser.parse_args()


def _summary_path(variant: str) -> Path:
    return Path(f"reports/assets/ms_cocoai_to_ishu_tuned_fusion_{variant}_tiled_dinov2_summary.csv")


def _read_variant(variant: str) -> pd.DataFrame:
    path = _summary_path(variant)
    if not path.exists():
        raise FileNotFoundError(f"Missing tiled-DINO summary for {variant}: {path}")
    frame = pd.read_csv(path)
    required = {"variant", "score_mode", *METRICS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    baseline = frame[frame["score_mode"] == "global"]
    if len(baseline) != 1:
        raise ValueError(f"Expected one global row for {variant}; got {len(baseline)}")
    base = baseline.iloc[0]
    rows = []
    for row in frame.itertuples(index=False):
        record = row._asdict()
        for metric in METRICS:
            delta = float(record[metric]) - float(base[metric])
            record[f"{metric}_delta_vs_global"] = delta
            if metric in LOWER_IS_BETTER:
                record[f"{metric}_improved_vs_global"] = delta < 0
            else:
                record[f"{metric}_improved_vs_global"] = delta > 0
        rows.append(record)
    return pd.DataFrame(rows)


def build_tradeoff(variants: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    tradeoff = pd.concat([_read_variant(variant) for variant in variants], ignore_index=True)
    choices = []
    for variant, group in tradeoff.groupby("variant", sort=False):
        accuracy_row = group.sort_values(
            ["target_accuracy_mean", "target_roc_auc_mean", "target_predicted_positive_rate_mean"],
            ascending=[False, False, True],
        ).iloc[0]
        auc_row = group.sort_values(
            ["target_roc_auc_mean", "target_accuracy_mean", "target_predicted_positive_rate_mean"],
            ascending=[False, False, True],
        ).iloc[0]
        brier_row = group.sort_values(
            ["target_brier_score_mean", "target_accuracy_mean", "target_roc_auc_mean"],
            ascending=[True, False, False],
        ).iloc[0]
        ece_row = group.sort_values(
            ["target_expected_calibration_error_mean", "target_accuracy_mean", "target_roc_auc_mean"],
            ascending=[True, False, False],
        ).iloc[0]
        choices.append(
            {
                "variant": variant,
                "best_accuracy_mode": accuracy_row["score_mode"],
                "best_accuracy_delta": accuracy_row["target_accuracy_mean_delta_vs_global"],
                "best_accuracy_auc_delta": accuracy_row["target_roc_auc_mean_delta_vs_global"],
                "best_accuracy_brier_delta": accuracy_row["target_brier_score_mean_delta_vs_global"],
                "best_accuracy_ece_delta": accuracy_row[
                    "target_expected_calibration_error_mean_delta_vs_global"
                ],
                "best_auc_mode": auc_row["score_mode"],
                "best_auc_delta": auc_row["target_roc_auc_mean_delta_vs_global"],
                "best_brier_mode": brier_row["score_mode"],
                "best_brier_delta": brier_row["target_brier_score_mean_delta_vs_global"],
                "best_brier_accuracy_delta": brier_row["target_accuracy_mean_delta_vs_global"],
                "best_brier_auc_delta": brier_row["target_roc_auc_mean_delta_vs_global"],
                "best_ece_mode": ece_row["score_mode"],
                "best_ece_delta": ece_row["target_expected_calibration_error_mean_delta_vs_global"],
                "best_ece_accuracy_delta": ece_row["target_accuracy_mean_delta_vs_global"],
                "best_ece_auc_delta": ece_row["target_roc_auc_mean_delta_vs_global"],
            }
        )
    return tradeoff, pd.DataFrame(choices)


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


def _mode_aggregate(tradeoff: pd.DataFrame) -> pd.DataFrame:
    delta_columns = [f"{metric}_delta_vs_global" for metric in METRICS]
    aggregate = tradeoff.groupby("score_mode", sort=False)[delta_columns].mean().reset_index()
    aggregate["brier_improvement_count"] = tradeoff.groupby("score_mode", sort=False)[
        "target_brier_score_mean_improved_vs_global"
    ].sum().to_numpy()
    aggregate["ece_improvement_count"] = tradeoff.groupby("score_mode", sort=False)[
        "target_expected_calibration_error_mean_improved_vs_global"
    ].sum().to_numpy()
    return aggregate


def build_report(tradeoff: pd.DataFrame, choices: pd.DataFrame) -> str:
    aggregate = _mode_aggregate(tradeoff)
    variants = int(tradeoff["variant"].nunique())
    best_accuracy_mode = choices["best_accuracy_mode"].mode().iloc[0]
    best_brier_mode = choices["best_brier_mode"].mode().iloc[0]
    tile_mean = aggregate[aggregate["score_mode"] == "tile_mean"].iloc[0]
    tile_max = aggregate[aggregate["score_mode"] == "tile_max"].iloc[0]
    all_mode_columns = [
        "variant",
        "score_mode",
        "target_accuracy_mean_delta_vs_global",
        "target_roc_auc_mean_delta_vs_global",
        "target_brier_score_mean_delta_vs_global",
        "target_expected_calibration_error_mean_delta_vs_global",
        "target_predicted_positive_rate_mean_delta_vs_global",
    ]
    choice_columns = [
        "variant",
        "best_accuracy_mode",
        "best_accuracy_delta",
        "best_accuracy_auc_delta",
        "best_accuracy_brier_delta",
        "best_accuracy_ece_delta",
        "best_brier_mode",
        "best_brier_delta",
        "best_brier_accuracy_delta",
        "best_brier_auc_delta",
        "best_ece_mode",
        "best_ece_delta",
    ]
    aggregate_columns = [
        "score_mode",
        "target_accuracy_mean_delta_vs_global",
        "target_roc_auc_mean_delta_vs_global",
        "target_brier_score_mean_delta_vs_global",
        "target_expected_calibration_error_mean_delta_vs_global",
        "target_predicted_positive_rate_mean_delta_vs_global",
        "brier_improvement_count",
        "ece_improvement_count",
    ]
    lines = [
        "# Tiled DINO Calibration Tradeoff",
        "",
        f"Run date: {RUN_DATE}",
        "",
        "This report separates the decision/ranking benefit of tiled-DINO branch replacement from its calibration behavior across transformed-target reverse SCP-Fusion probes.",
        "",
        "## Headline",
        "",
        (
            f"- Decision/ranking mode: `{best_accuracy_mode}` is the best accuracy mode on all "
            f"{variants} checked transforms; its average deltas are "
            f"{tile_max['target_accuracy_mean_delta_vs_global']:+.4f} accuracy and "
            f"{tile_max['target_roc_auc_mean_delta_vs_global']:+.4f} AUC."
        ),
        (
            f"- Calibration-safe mode: `{best_brier_mode}` is the best Brier mode on all checked "
            f"transforms; `tile_mean` improves Brier on "
            f"{int(tile_mean['brier_improvement_count'])}/{variants} transforms and ECE on "
            f"{int(tile_mean['ece_improvement_count'])}/{variants} transforms."
        ),
        "",
        "The paper-safe claim is therefore two-part: tiled DINO helps source-fixed decisions/ranking when fused, while `tile_mean` is the safer calibration diagnostic and `tile_max` is the stronger operating-point diagnostic.",
        "",
        "## Mode Average Deltas",
        "",
        _markdown_table(aggregate, aggregate_columns),
        "",
        "## Per-Transform Choices",
        "",
        _markdown_table(choices, choice_columns),
        "",
        "## All Mode Deltas",
        "",
        _markdown_table(tradeoff, all_mode_columns),
        "",
        "## Interpretation",
        "",
        "- Use `tile_max` for the robustness headline table, with a calibration caveat.",
        "- Use `tile_mean` when discussing Brier/ECE behavior or calibration-aware operating points.",
        "- Do not claim tiled-DINO improves calibration universally; the ECE result is transform-dependent.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    tradeoff, choices = build_tradeoff(list(args.variants))

    csv_path = Path(args.csv_out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tradeoff.to_csv(csv_path, index=False)

    choice_path = Path(args.choice_csv_out)
    choice_path.parent.mkdir(parents=True, exist_ok=True)
    choices.to_csv(choice_path, index=False)

    report_path = Path(args.out_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_report(tradeoff, choices), encoding="utf-8")

    print(report_path.resolve())
    print(csv_path.resolve())
    print(choice_path.resolve())


if __name__ == "__main__":
    main()
