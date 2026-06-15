from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "objective",
    "selected_method",
    "selected_mode",
    "metric",
    "metric_value",
}


REQUIRED_OBJECTIVES = {
    "ranking_auc",
    "probability_brier",
    "reliability_ece",
    "source_holdout_ece",
    "tiled_dino_accuracy",
    "tiled_dino_brier",
}


def load_calibration_operating_modes(path: Path) -> pd.DataFrame:
    modes = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(modes.columns)
    if missing:
        raise ValueError(
            f"Calibration operating modes table is missing columns: {', '.join(sorted(missing))}"
        )
    missing_objectives = REQUIRED_OBJECTIVES - set(modes["objective"].astype(str))
    if missing_objectives:
        raise ValueError(
            "Calibration operating modes table is missing objectives: "
            + ", ".join(sorted(missing_objectives))
        )
    return modes


def objective_row(modes: pd.DataFrame, objective: str) -> pd.Series:
    matches = modes[modes["objective"].astype(str) == objective]
    if matches.empty:
        raise ValueError(f"Missing calibration operating-mode objective={objective!r}")
    return matches.iloc[0]


def objective_summary(modes: pd.DataFrame, objective: str) -> tuple[str, str, str, str]:
    row = objective_row(modes, objective)
    return (
        str(row["selected_method"]),
        str(row["selected_mode"]),
        str(row["metric"]),
        f"{float(row['metric_value']):.4f}",
    )


def operating_mode_guidance_sentence(modes: pd.DataFrame) -> str:
    ranking_method, _ranking_mode, _ranking_metric, ranking_value = objective_summary(
        modes, "ranking_auc"
    )
    brier_method, _brier_mode, _brier_metric, brier_value = objective_summary(
        modes, "probability_brier"
    )
    ece_method, _ece_mode, _ece_metric, ece_value = objective_summary(modes, "reliability_ece")
    source_method, source_mode, _source_metric, source_value = objective_summary(
        modes, "source_holdout_ece"
    )
    tile_acc_method, tile_acc_mode, _tile_acc_metric, tile_acc_value = objective_summary(
        modes, "tiled_dino_accuracy"
    )
    _tile_brier_method, tile_brier_mode, _tile_brier_metric, tile_brier_value = objective_summary(
        modes, "tiled_dino_brier"
    )
    return (
        "The operating-mode audit is deliberately objective-specific: "
        f"{ranking_method} leads transfer AUC at {ranking_value}; "
        f"{brier_method} gives the best transfer Brier score at {brier_value}; "
        f"{ece_method} gives the lowest transfer ECE at {ece_value}; "
        f"{source_method} with `{source_mode}` anchors source-heldout ECE at {source_value}; "
        f"and {tile_acc_method} separates `{tile_acc_mode}` for decision/ranking stress "
        f"from `{tile_brier_mode}` for Brier-sensitive stress ({tile_brier_value})."
    )


def operating_mode_results_sentence(modes: pd.DataFrame) -> str:
    brier_method, _brier_mode, _brier_metric, brier_value = objective_summary(
        modes, "probability_brier"
    )
    ece_method, _ece_mode, _ece_metric, ece_value = objective_summary(modes, "reliability_ece")
    return (
        "The calibration audit keeps operating modes separate: "
        f"{brier_method} leads Brier at {brier_value}, while {ece_method} leads ECE at {ece_value}."
    )
