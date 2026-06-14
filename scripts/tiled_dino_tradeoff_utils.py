from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = {
    "variant",
    "score_mode",
    "target_accuracy_mean_delta_vs_global",
    "target_roc_auc_mean_delta_vs_global",
    "target_brier_score_mean_improved_vs_global",
    "target_expected_calibration_error_mean_improved_vs_global",
}


def _bool_count(series: pd.Series) -> int:
    return int(series.astype(str).str.lower().isin(["true", "1", "yes"]).sum())


def _signed_metric(value: float) -> str:
    return f"{float(value):+.4f}"


def load_tiled_dino_tradeoff_summary(
    tradeoff_path: Path,
    *,
    allow_missing: bool = False,
) -> dict[str, str] | None:
    if allow_missing and not tradeoff_path.exists():
        return None

    tradeoff = pd.read_csv(tradeoff_path)
    missing = REQUIRED_COLUMNS - set(tradeoff.columns)
    if missing:
        raise ValueError(f"Missing tiled-DINO tradeoff columns: {sorted(missing)}")

    n_transforms = int(tradeoff["variant"].nunique())
    mode_means = (
        tradeoff.groupby("score_mode", as_index=True)[
            ["target_accuracy_mean_delta_vs_global", "target_roc_auc_mean_delta_vs_global"]
        ]
        .mean()
        .to_dict("index")
    )
    for mode in ["tile_max", "tile_mean"]:
        if mode not in mode_means:
            raise ValueError(f"Missing tiled-DINO score_mode={mode!r}")

    tile_mean = tradeoff[tradeoff["score_mode"].eq("tile_mean")]
    return {
        "n_transforms": str(n_transforms),
        "tile_max_acc_delta": _signed_metric(mode_means["tile_max"]["target_accuracy_mean_delta_vs_global"]),
        "tile_max_auc_delta": _signed_metric(mode_means["tile_max"]["target_roc_auc_mean_delta_vs_global"]),
        "tile_mean_brier_count": f"{_bool_count(tile_mean['target_brier_score_mean_improved_vs_global'])}/{n_transforms}",
        "tile_mean_ece_count": f"{_bool_count(tile_mean['target_expected_calibration_error_mean_improved_vs_global'])}/{n_transforms}",
    }


def tiled_dino_sentence(summary: dict[str, Any], *, terminal_label: str = "transforms") -> str:
    return (
        f"Across {summary['n_transforms']} transform-stress probes, tiled-DINO `tile_max` gives "
        f"{summary['tile_max_acc_delta']} accuracy and {summary['tile_max_auc_delta']} AUC average deltas, "
        f"while `tile_mean` improves Brier on {summary['tile_mean_brier_count']} and ECE on "
        f"{summary['tile_mean_ece_count']} {terminal_label}."
    )


def tiled_dino_evidence_summary(summary: dict[str, Any]) -> str:
    return (
        "tiled_dinov2_calibration_tradeoff "
        f"(tile_max: mean acc_delta={summary['tile_max_acc_delta']}, "
        f"mean AUC_delta={summary['tile_max_auc_delta']} "
        f"across {summary['n_transforms']} transform-stress probes; "
        f"tile_mean: Brier improves on {summary['tile_mean_brier_count']}, "
        f"ECE improves on {summary['tile_mean_ece_count']})"
    )
