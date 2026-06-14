from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


RUN_DATE = "2026-06-14"

DEFAULT_VARIANTS = ["blur1", "jpeg30", "resize_half", "screenshot"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a compact comparison of tiled-DINO transform stress probes."
    )
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    parser.add_argument(
        "--out-path",
        default="reports/tiled_dinov2_transform_stress_comparison_2026_06_14.md",
        help="Markdown comparison report to write.",
    )
    parser.add_argument(
        "--csv-out",
        default="reports/assets/tiled_dinov2_transform_stress_comparison.csv",
        help="Machine-readable comparison table to write.",
    )
    return parser.parse_args()


def _baseline_path(variant: str) -> Path:
    return Path(f"reports/assets/ms_cocoai_to_ishu_tuned_fusion_{variant}_robustness_summary.csv")


def _fused_tiled_path(variant: str) -> Path:
    return Path(f"reports/assets/ms_cocoai_to_ishu_tuned_fusion_{variant}_tiled_dinov2_summary.csv")


def _standalone_tiled_path(variant: str) -> Path:
    return Path(f"reports/assets/tiled_dinov2_{variant}_reverse_transfer_summary.csv")


def _report_path(variant: str) -> str:
    return f"reports/ms_cocoai_to_ishu_tuned_fusion_{variant}_tiled_dinov2_2026_06_14.md"


def _read_one_row(path: Path, label: str) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"{label} is empty: {path}")
    return frame.iloc[0]


def _read_variant(variant: str) -> dict[str, object]:
    baseline = _read_one_row(_baseline_path(variant), f"{variant} baseline robustness summary")
    fused = pd.read_csv(_fused_tiled_path(variant))
    standalone = pd.read_csv(_standalone_tiled_path(variant))

    if fused.empty:
        raise ValueError(f"Fused tiled-DINO summary is empty for {variant}")
    if standalone.empty:
        raise ValueError(f"Standalone tiled-DINO summary is empty for {variant}")

    best_accuracy = fused.sort_values(
        ["target_accuracy_mean", "target_roc_auc_mean", "target_predicted_positive_rate_mean"],
        ascending=[False, False, True],
    ).iloc[0]
    best_auc = fused.sort_values(
        ["target_roc_auc_mean", "target_accuracy_mean", "target_predicted_positive_rate_mean"],
        ascending=[False, False, True],
    ).iloc[0]
    standalone_best_accuracy = standalone.sort_values(
        ["accuracy_mean", "roc_auc_mean", "predicted_fake_rate_mean"],
        ascending=[False, False, True],
    ).iloc[0]
    standalone_best_auc = standalone.sort_values(
        ["roc_auc_mean", "accuracy_mean", "predicted_fake_rate_mean"],
        ascending=[False, False, True],
    ).iloc[0]

    baseline_accuracy = float(baseline["target_accuracy_mean"])
    baseline_auc = float(baseline["target_roc_auc_mean"])
    baseline_brier = float(baseline["target_brier_score_mean"])
    baseline_ece = float(baseline["target_expected_calibration_error_mean"])

    return {
        "variant": variant,
        "baseline_accuracy": baseline_accuracy,
        "baseline_auc": baseline_auc,
        "baseline_brier": baseline_brier,
        "baseline_ece": baseline_ece,
        "best_accuracy_mode": best_accuracy["score_mode"],
        "best_accuracy": float(best_accuracy["target_accuracy_mean"]),
        "best_accuracy_auc": float(best_accuracy["target_roc_auc_mean"]),
        "accuracy_delta": float(best_accuracy["target_accuracy_mean"]) - baseline_accuracy,
        "auc_delta_for_best_accuracy": float(best_accuracy["target_roc_auc_mean"]) - baseline_auc,
        "brier_delta_for_best_accuracy": float(best_accuracy["target_brier_score_mean"]) - baseline_brier,
        "ece_delta_for_best_accuracy": (
            float(best_accuracy["target_expected_calibration_error_mean"]) - baseline_ece
        ),
        "best_auc_mode": best_auc["score_mode"],
        "best_auc": float(best_auc["target_roc_auc_mean"]),
        "best_auc_accuracy": float(best_auc["target_accuracy_mean"]),
        "auc_delta": float(best_auc["target_roc_auc_mean"]) - baseline_auc,
        "accuracy_delta_for_best_auc": float(best_auc["target_accuracy_mean"]) - baseline_accuracy,
        "standalone_best_accuracy_mode": standalone_best_accuracy["score_mode"],
        "standalone_best_accuracy": float(standalone_best_accuracy["accuracy_mean"]),
        "standalone_best_auc_mode": standalone_best_auc["score_mode"],
        "standalone_best_auc": float(standalone_best_auc["roc_auc_mean"]),
        "source_report": _report_path(variant),
    }


def build_comparison(variants: list[str]) -> pd.DataFrame:
    return pd.DataFrame([_read_variant(variant) for variant in variants])


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
        ["accuracy_delta", "auc_delta_for_best_accuracy"], ascending=False
    ).iloc[0]
    best_auc = comparison.sort_values(["auc_delta", "accuracy_delta_for_best_auc"], ascending=False).iloc[0]
    display_columns = [
        "variant",
        "baseline_accuracy",
        "baseline_auc",
        "best_accuracy_mode",
        "best_accuracy",
        "best_accuracy_auc",
        "accuracy_delta",
        "auc_delta_for_best_accuracy",
        "brier_delta_for_best_accuracy",
        "ece_delta_for_best_accuracy",
        "best_auc_mode",
        "best_auc",
        "auc_delta",
        "standalone_best_accuracy",
        "standalone_best_auc",
    ]
    lines = [
        "# Tiled DINO Transform Stress Comparison",
        "",
        f"Run date: {RUN_DATE}",
        "",
        "This compares transformed-target reverse SCP-Fusion robustness before and after replacing the transformed DINOv2-small target branch with native-tile DINOv2 scores. Source rows, selected fusion configurations, and source threshold policy stay fixed.",
        "",
        "## Headline",
        "",
        (
            f"- Largest accuracy lift: `{best_accuracy['variant']}` via "
            f"`{best_accuracy['best_accuracy_mode']}` "
            f"({best_accuracy['accuracy_delta']:+.4f} accuracy, "
            f"{best_accuracy['auc_delta_for_best_accuracy']:+.4f} AUC)."
        ),
        (
            f"- Largest AUC lift: `{best_auc['variant']}` via `{best_auc['best_auc_mode']}` "
            f"({best_auc['auc_delta']:+.4f} AUC, "
            f"{best_auc['accuracy_delta_for_best_auc']:+.4f} accuracy)."
        ),
        "",
        "Checked transforms show modest accuracy/AUC improvements from tiled-DINO fusion, but calibration is mixed: Brier usually improves for `tile_mean`, while the best accuracy/AUC modes can increase ECE or fake-call rate. Treat this as robustness support for the ranking/decision branch, not a calibration upgrade.",
        "",
        "## Comparison",
        "",
        _markdown_table(comparison, display_columns),
        "",
        "## Interpretation",
        "",
        "- `tile_max` is currently the strongest transformed-target fusion mode for all checked transforms.",
        "- Standalone tiled DINO remains weak under the checked transforms; the gain appears only when DINO is fused with physical/conventional and other neural/foundation branches.",
        "- The remaining high-resolution tiling gap is no longer these proxy transforms; it is official or paper-compatible external high-resolution benchmark evidence.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    comparison = build_comparison(list(args.variants))

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
