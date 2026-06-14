from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


RUN_DATE = "2026-06-14"

TRANSFER_OBJECTIVES = [
    ("threshold_accuracy", "accuracy", "max", "Default 0.5 threshold decisions on Ishu -> MS COCOAI."),
    ("ranking_auc", "auc", "max", "Cross-domain ranking independent of the default threshold."),
    ("probability_brier", "brier", "min", "Squared probability error; lower is better."),
    ("reliability_ece", "ece", "min", "Expected calibration error; lower is better."),
]

SOURCE_OBJECTIVES = [
    (
        "source_holdout_accuracy",
        "mean_calibrated_accuracy",
        "max",
        "Leave-one-source-out calibrated decision accuracy.",
    ),
    (
        "source_holdout_brier",
        "mean_calibrated_brier_score",
        "min",
        "Leave-one-source-out calibrated Brier score.",
    ),
    (
        "source_holdout_ece",
        "mean_calibrated_ece",
        "min",
        "Leave-one-source-out expected calibration error.",
    ),
    (
        "source_holdout_fake_detection",
        "mean_calibrated_fake_detection",
        "max",
        "Generated-image recall after source-heldout calibration.",
    ),
]

TILED_MODE_COLUMNS = [
    ("tiled_dino_accuracy", "best_accuracy_mode", "best_accuracy_delta"),
    ("tiled_dino_auc", "best_auc_mode", "best_auc_delta"),
    ("tiled_dino_brier", "best_brier_mode", "best_brier_delta"),
    ("tiled_dino_ece", "best_ece_mode", "best_ece_delta"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a paper-facing calibration and operating-mode synthesis."
    )
    parser.add_argument(
        "--core-results",
        default="reports/assets/publication_core_results.csv",
        help="Canonical publication result table.",
    )
    parser.add_argument(
        "--source-holdout-calibration",
        default="reports/assets/score_fusion_source_holdout_calibration_summary.csv",
        help="Source-heldout calibration summary table.",
    )
    parser.add_argument(
        "--tiled-dino-choices",
        default="reports/assets/tiled_dinov2_calibration_tradeoff_choices.csv",
        help="Tiled-DINO calibration/accuracy mode-choice table.",
    )
    parser.add_argument(
        "--out-path",
        default="reports/calibration_operating_modes_2026_06_14.md",
        help="Markdown report to write.",
    )
    parser.add_argument(
        "--csv-out",
        default="reports/assets/calibration_operating_modes.csv",
        help="Machine-readable operating-mode table to write.",
    )
    return parser.parse_args()


def _numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _best_row(frame: pd.DataFrame, metric: str, direction: str) -> pd.Series:
    candidates = frame.dropna(subset=[metric])
    if candidates.empty:
        raise ValueError(f"No rows with metric {metric!r}")
    index = candidates[metric].idxmax() if direction == "max" else candidates[metric].idxmin()
    return candidates.loc[index]


def _transfer_rows(core_results: pd.DataFrame) -> list[dict[str, object]]:
    core = _numeric(
        core_results,
        ["accuracy", "auc", "brier", "ece", "predicted_fake_rate"],
    )
    transfer = core[
        core["setting"].fillna("").str.startswith("Ishu -> source-balanced MS COCOAI")
        & core["auc"].notna()
    ].copy()
    rows = []
    for objective, metric, direction, recommendation in TRANSFER_OBJECTIVES:
        best = _best_row(transfer, metric, direction)
        rows.append(
            {
                "evidence_group": "ishu_to_ms_transfer",
                "objective": objective,
                "selected_method": best["method"],
                "selected_mode": "default_score",
                "metric": metric,
                "metric_value": float(best[metric]),
                "secondary_metric": "predicted_fake_rate",
                "secondary_value": (
                    float(best["predicted_fake_rate"]) if pd.notna(best.get("predicted_fake_rate")) else pd.NA
                ),
                "evidence": best["source"],
                "recommendation": recommendation,
            }
        )
    return rows


def _source_holdout_rows(summary: pd.DataFrame) -> list[dict[str, object]]:
    summary = _numeric(
        summary,
        [
            "mean_calibrated_accuracy",
            "mean_calibrated_brier_score",
            "mean_calibrated_ece",
            "mean_calibrated_fake_detection",
            "mean_calibrated_real_fpr",
            "mean_calibrated_roc_auc",
        ],
    )
    rows = []
    for objective, metric, direction, recommendation in SOURCE_OBJECTIVES:
        best = _best_row(summary, metric, direction)
        rows.append(
            {
                "evidence_group": "source_holdout_calibration",
                "objective": objective,
                "selected_method": best["method"],
                "selected_mode": best["calibrator"],
                "metric": metric,
                "metric_value": float(best[metric]),
                "secondary_metric": "mean_calibrated_real_fpr",
                "secondary_value": float(best["mean_calibrated_real_fpr"]),
                "evidence": "reports/assets/score_fusion_source_holdout_calibration_summary.csv",
                "recommendation": recommendation,
            }
        )
    return rows


def _mode_count_text(values: pd.Series) -> str:
    counts = values.value_counts().sort_index()
    return ", ".join(f"{mode}={count}" for mode, count in counts.items())


def _tiled_mode_rows(choices: pd.DataFrame) -> list[dict[str, object]]:
    choices = _numeric(choices, [column for _objective, _mode, column in TILED_MODE_COLUMNS])
    rows = []
    for objective, mode_column, delta_column in TILED_MODE_COLUMNS:
        counts = choices[mode_column].value_counts()
        selected_mode = counts.idxmax()
        selected = choices[choices[mode_column] == selected_mode]
        rows.append(
            {
                "evidence_group": "tiled_dinov2_transform_stress",
                "objective": objective,
                "selected_method": "tiled DINOv2 reverse fusion",
                "selected_mode": selected_mode,
                "metric": delta_column,
                "metric_value": float(selected[delta_column].mean()),
                "secondary_metric": "mode_counts",
                "secondary_value": _mode_count_text(choices[mode_column]),
                "evidence": "reports/assets/tiled_dinov2_calibration_tradeoff_choices.csv",
                "recommendation": (
                    "Use the majority mode for this objective; report mixed modes explicitly when counts are split."
                ),
            }
        )
    return rows


def build_calibration_operating_modes(
    core_results_path: Path,
    source_holdout_path: Path,
    tiled_choices_path: Path,
) -> tuple[str, pd.DataFrame]:
    core_results = pd.read_csv(core_results_path)
    source_holdout = pd.read_csv(source_holdout_path)
    tiled_choices = pd.read_csv(tiled_choices_path)
    rows = (
        _transfer_rows(core_results)
        + _source_holdout_rows(source_holdout)
        + _tiled_mode_rows(tiled_choices)
    )
    modes = pd.DataFrame(rows)
    return _write_markdown(modes), modes


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


def _display(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "objective": frame["objective"],
            "selected_method": frame["selected_method"],
            "mode": frame["selected_mode"],
            "metric": frame["metric"],
            "value": frame["metric_value"].map(lambda value: f"{float(value):.4f}"),
            "secondary": frame["secondary_metric"].astype(str) + "=" + frame["secondary_value"].astype(str),
            "recommendation": frame["recommendation"],
        }
    )


def _write_markdown(frame: pd.DataFrame) -> str:
    lines = [
        "# Calibration Operating Modes",
        "",
        f"Run date: {RUN_DATE}",
        "",
        "Generated by `scripts/build_calibration_operating_modes.py` from checked-in calibration and result tables.",
        "",
        "This report separates ranking, calibration, and thresholded-decision objectives so SCP-Fusion is not oversold as one universal operating point.",
    ]
    for group, title in [
        ("ishu_to_ms_transfer", "Ishu To MS COCOAI Transfer"),
        ("source_holdout_calibration", "Source-Heldout Calibration"),
        ("tiled_dinov2_transform_stress", "Tiled-DINO Transform Stress"),
    ]:
        subset = frame[frame["evidence_group"] == group]
        lines.extend(["", f"## {title}", "", _markdown_table(_display(subset), list(_display(subset).columns))])
    lines.extend(
        [
            "",
            "## Paper Guidance",
            "",
            "- Use frozen CLIP as the current transfer ranking/accuracy frontier when the question is Ishu -> MS COCOAI generalization.",
            "- Use SCP-Fusion + CLIP when discussing probability-error improvement, but keep the ECE caveat visible.",
            "- Use source-heldout calibration as the decision-policy story: it improves Brier/ECE without claiming universal detector superiority.",
            "- Use tiled-DINO `tile_max` for decision/ranking stress headlines and `tile_mean` for Brier/ECE-sensitive wording, noting resize-half ECE is neutral in the current checked-in evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    text, modes = build_calibration_operating_modes(
        Path(args.core_results),
        Path(args.source_holdout_calibration),
        Path(args.tiled_dino_choices),
    )
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    csv_path = Path(args.csv_out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    modes.to_csv(csv_path, index=False)
    print(out_path.resolve())
    print(csv_path.resolve())


if __name__ == "__main__":
    main()
