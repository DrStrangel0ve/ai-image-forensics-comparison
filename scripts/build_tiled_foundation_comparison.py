from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


RUN_DATE = "2026-06-14"

DEFAULT_INPUTS = [
    (
        "CLIP ViT-B/32",
        "reports/assets/tiled_clip_reverse_transfer_summary.csv",
        "reports/tiled_clip_reverse_transfer_2026_06_14.md",
    ),
    (
        "DINOv2-small",
        "reports/assets/tiled_dinov2_reverse_transfer_summary.csv",
        "reports/tiled_dinov2_reverse_transfer_2026_06_14.md",
    ),
    (
        "ConvNeXt-Tiny",
        "reports/assets/tiled_convnext_reverse_transfer_summary.csv",
        "reports/tiled_convnext_reverse_transfer_2026_06_14.md",
    ),
]

METRIC_COLUMNS = [
    "accuracy_mean",
    "roc_auc_mean",
    "brier_score_mean",
    "expected_calibration_error_mean",
    "predicted_fake_rate_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a compact comparison report for tiled frozen-foundation reverse-transfer runs."
    )
    parser.add_argument(
        "--out-path",
        default="reports/tiled_foundation_reverse_transfer_comparison_2026_06_14.md",
        help="Markdown report to write.",
    )
    parser.add_argument(
        "--csv-out",
        default="reports/assets/tiled_foundation_reverse_transfer_comparison.csv",
        help="Machine-readable comparison table to write.",
    )
    return parser.parse_args()


def _read_summary(encoder: str, summary_path: Path, report_path: str) -> pd.DataFrame:
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing tiled summary for {encoder}: {summary_path}")
    frame = pd.read_csv(summary_path)
    required = {"score_mode", "n_seeds", "n_images_total", "mean_tiles", *METRIC_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"{summary_path} is missing required columns: {', '.join(sorted(missing))}"
        )
    frame = frame.copy()
    frame.insert(0, "encoder", encoder)
    frame["source_report"] = report_path
    return frame


def load_comparison() -> pd.DataFrame:
    frames = [
        _read_summary(encoder, Path(summary_path), report_path)
        for encoder, summary_path, report_path in DEFAULT_INPUTS
    ]
    comparison = pd.concat(frames, ignore_index=True)
    global_rows = (
        comparison.loc[comparison["score_mode"] == "global", ["encoder", *METRIC_COLUMNS]]
        .set_index("encoder")
        .add_prefix("global_")
    )
    comparison = comparison.join(global_rows, on="encoder")
    for column in METRIC_COLUMNS:
        comparison[f"{column}_delta_vs_global"] = (
            comparison[column] - comparison[f"global_{column}"]
        )
    return comparison


def _markdown_escape(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).replace("\n", " ").replace("|", "\\|")


def _format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return _markdown_escape(value)


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame[columns].itertuples(index=False):
        lines.append("| " + " | ".join(_format_value(value) for value in row) + " |")
    return "\n".join(lines)


def build_report(comparison: pd.DataFrame) -> str:
    best_accuracy = comparison.sort_values("accuracy_mean", ascending=False).iloc[0]
    best_auc = comparison.sort_values("roc_auc_mean", ascending=False).iloc[0]
    best_brier = comparison.sort_values("brier_score_mean", ascending=True).iloc[0]
    best_ece = comparison.sort_values(
        "expected_calibration_error_mean", ascending=True
    ).iloc[0]

    best_by_encoder = []
    for encoder, group in comparison.groupby("encoder", sort=False):
        acc_row = group.sort_values("accuracy_mean", ascending=False).iloc[0]
        auc_row = group.sort_values("roc_auc_mean", ascending=False).iloc[0]
        best_by_encoder.append(
            {
                "encoder": encoder,
                "best_accuracy_mode": acc_row["score_mode"],
                "best_accuracy": acc_row["accuracy_mean"],
                "accuracy_delta_vs_global": acc_row["accuracy_mean_delta_vs_global"],
                "best_auc_mode": auc_row["score_mode"],
                "best_auc": auc_row["roc_auc_mean"],
                "auc_delta_vs_global": auc_row["roc_auc_mean_delta_vs_global"],
                "best_brier_for_accuracy_mode": acc_row["brier_score_mean"],
                "source_report": acc_row["source_report"],
            }
        )
    best_frame = pd.DataFrame(best_by_encoder)

    display_columns = [
        "encoder",
        "score_mode",
        "accuracy_mean",
        "accuracy_mean_delta_vs_global",
        "roc_auc_mean",
        "roc_auc_mean_delta_vs_global",
        "brier_score_mean",
        "expected_calibration_error_mean",
        "predicted_fake_rate_mean",
    ]
    lines = [
        "# Tiled Foundation Reverse-Transfer Comparison",
        "",
        f"Run date: {RUN_DATE}",
        "",
        "This report compares native-image tile aggregation for the three saved MS-COCOAI-to-Ishu frozen-encoder classifiers. The classifiers are unchanged; only target-side scoring changes from one global resized image to deterministic native tiles.",
        "",
        "## Headline",
        "",
        f"- Best default-threshold accuracy: `{best_accuracy['encoder']}` with `{best_accuracy['score_mode']}` at `{best_accuracy['accuracy_mean']:.4f}` accuracy.",
        f"- Best ranking AUC: `{best_auc['encoder']}` with `{best_auc['score_mode']}` at `{best_auc['roc_auc_mean']:.4f}` AUC.",
        f"- Best Brier score: `{best_brier['encoder']}` with `{best_brier['score_mode']}` at `{best_brier['brier_score_mean']:.4f}`.",
        f"- Best ECE: `{best_ece['encoder']}` with `{best_ece['score_mode']}` at `{best_ece['expected_calibration_error_mean']:.4f}`.",
        "",
        "ConvNeXt-Tiny is the strongest tiled forced-decision encoder in this reverse direction, while CLIP remains the strongest ranking encoder. DINOv2 improves under tiling, but its reverse-transfer global baseline is too weak to catch the other two encoders.",
        "",
        "## Best Mode Per Encoder",
        "",
        _markdown_table(
            best_frame,
            [
                "encoder",
                "best_accuracy_mode",
                "best_accuracy",
                "accuracy_delta_vs_global",
                "best_auc_mode",
                "best_auc",
                "auc_delta_vs_global",
                "best_brier_for_accuracy_mode",
                "source_report",
            ],
        ),
        "",
        "## All Score Modes",
        "",
        _markdown_table(comparison, display_columns),
        "",
        "## Interpretation",
        "",
        "- `tile_mean` is the safer operating-point aggregator: it improves accuracy/calibration for all three encoders.",
        "- `tile_top2_mean` is the stronger ranking aggregator for all three encoders, but it tends to raise the predicted fake rate and should be threshold-calibrated before deployment claims.",
        "- The next SCP-Fusion step is to feed the best tiled foundation score modes into the reverse fusion stack and test whether they beat the current native-tiled conventional branch result of 0.7749 accuracy / 0.8472 AUC.",
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
