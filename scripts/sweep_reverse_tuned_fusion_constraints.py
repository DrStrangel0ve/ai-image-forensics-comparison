from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tune_reverse_fusion_source_holdout import (
    _format_float_token,
    _markdown_table,
    _parse_cap_list,
    _parse_dropout_configs,
    _parse_float_list,
    evaluate_final_config,
    evaluate_source_holdout_grid,
    load_seed_frames,
    select_config,
    summarize_grid,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep source fake-call constraints for reverse source-heldout tuned fusion."
        )
    )
    parser.add_argument("--run-root", default="runs")
    parser.add_argument(
        "--metadata",
        default="data/raw/ms_cocoai_2026_validation_source_balanced_100/metadata.csv",
    )
    parser.add_argument("--summary-dir", default="reports/assets")
    parser.add_argument(
        "--report-path",
        default="reports/ms_cocoai_to_ishu_tuned_fusion_constraint_sweep_2026_06_13.md",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 29])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "combined_v3",
            "resnet18",
            "physics_guided",
            "convnext_tiny",
            "clip_vit_b_32",
            "dinov2_vits14",
        ],
    )
    parser.add_argument("--fusion-cs", default="1,0.3,0.1,0.03,0.01")
    parser.add_argument("--dropout-configs", default="none,mean0p35x8")
    parser.add_argument("--source-fake-rate-caps", default="0.40,0.42,0.45,0.48,0.50")
    parser.add_argument("--selection-score", choices=["mean", "min"], default="min")
    parser.add_argument("--real-validation-fraction", type=float, default=0.5)
    parser.add_argument("--fake-detection-weight", type=float, default=1.0)
    parser.add_argument("--real-clearance-weight", type=float, default=1.0)
    parser.add_argument("--real-fpr-penalty", type=float, default=4.0)
    parser.add_argument("--fake-miss-penalty", type=float, default=1.5)
    parser.add_argument("--threshold-tiebreak", choices=["higher", "lower", "near_half"], default="higher")
    return parser.parse_args()


def cap_label(cap: float | None) -> str:
    if cap is None:
        return "uncapped"
    return f"cap_{_format_float_token(cap)}"


def _cap_equals(series: pd.Series, cap: float | None) -> pd.Series:
    if cap is None:
        return series.isna()
    return (series.astype(float) - float(cap)).abs() < 1e-12


def summarize_by_cap(selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, group in selected.groupby("constraint_policy", sort=False):
        row = {"constraint_policy": label, "n_seeds": int(group["seed"].nunique())}
        for column in [
            "target_accuracy",
            "target_roc_auc",
            "target_brier_score",
            "target_expected_calibration_error",
            "target_precision",
            "target_recall",
            "target_f1",
            "target_predicted_positive_rate",
            "source_predicted_positive_rate",
            "threshold_source_predicted_positive_rate",
            "selection_validation_utility_mean",
            "selection_validation_utility_min",
        ]:
            values = pd.to_numeric(group[column], errors="coerce")
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_std"] = float(values.std(ddof=1)) if len(values.dropna()) > 1 else 0.0
        row["selected_configs"] = "; ".join(
            f"seed{int(item.seed)}:C{item.fusion_c:g}:{item.dropout_config}:cap{item.source_fake_rate_cap}"
            for item in group.sort_values("seed").itertuples(index=False)
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["target_accuracy_mean", "target_predicted_positive_rate_mean"],
        ascending=[False, True],
    )


def run_sweep(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fusion_cs = _parse_float_list(args.fusion_cs)
    dropouts = _parse_dropout_configs(args.dropout_configs)
    dropout_by_label = {dropout.label: dropout for dropout in dropouts}
    caps = _parse_cap_list(args.source_fake_rate_caps)
    selected_rows = []
    grid_rows = []
    for seed in args.seeds:
        source, target, methods = load_seed_frames(
            Path(args.run_root),
            Path(args.metadata),
            list(args.methods),
            seed,
        )
        folds = evaluate_source_holdout_grid(
            source,
            methods,
            fusion_cs,
            dropouts,
            caps,
            seed,
            real_validation_fraction=args.real_validation_fraction,
            threshold_tiebreak=args.threshold_tiebreak,
            fake_detection_weight=args.fake_detection_weight,
            real_clearance_weight=args.real_clearance_weight,
            real_fpr_penalty=args.real_fpr_penalty,
            fake_miss_penalty=args.fake_miss_penalty,
        )
        grid = summarize_grid(folds)
        grid_rows.append(grid)
        for cap in caps:
            cap_grid = grid[_cap_equals(grid["source_fake_rate_cap"], cap)].copy()
            selected_config = select_config(cap_grid, args.selection_score)
            selected = evaluate_final_config(
                source,
                target,
                methods,
                selected_config,
                dropout_by_label,
                seed,
                threshold_tiebreak=args.threshold_tiebreak,
                fake_detection_weight=args.fake_detection_weight,
                real_clearance_weight=args.real_clearance_weight,
                real_fpr_penalty=args.real_fpr_penalty,
                fake_miss_penalty=args.fake_miss_penalty,
            )
            selected["constraint_policy"] = cap_label(cap)
            selected["selection_score"] = args.selection_score
            selected_rows.append(selected)
    selected = pd.DataFrame(selected_rows)
    grid_summary = pd.concat(grid_rows, ignore_index=True)
    summary = summarize_by_cap(selected)
    return grid_summary, selected, summary


def write_report(summary: pd.DataFrame, selected: pd.DataFrame, grid: pd.DataFrame, out_path: Path) -> None:
    summary_columns = [
        "constraint_policy",
        "n_seeds",
        "target_accuracy_mean",
        "target_roc_auc_mean",
        "target_brier_score_mean",
        "target_expected_calibration_error_mean",
        "target_predicted_positive_rate_mean",
        "source_predicted_positive_rate_mean",
        "selected_configs",
    ]
    selected_columns = [
        "constraint_policy",
        "seed",
        "fusion_c",
        "dropout_config",
        "source_fake_rate_cap",
        "target_accuracy",
        "target_roc_auc",
        "target_predicted_positive_rate",
        "threshold_source_predicted_positive_rate",
    ]
    grid_columns = [
        "seed",
        "fusion_c",
        "dropout_config",
        "source_fake_rate_cap",
        "validation_utility_mean",
        "validation_utility_min",
        "validation_accuracy_mean",
        "validation_predicted_positive_rate_mean",
    ]
    best = summary.iloc[0]
    lowest_fake = summary.sort_values(
        ["target_predicted_positive_rate_mean", "target_accuracy_mean"],
        ascending=[True, False],
    ).iloc[0]
    lines = [
        "# MS COCOAI to Ishu Tuned-Fusion Constraint Sweep",
        "",
        (
            "This report sweeps the source fake-call cap used by source-heldout tuned fusion. "
            "Each cap is selected independently by worst-source utility, then retrained on all "
            "source rows and evaluated on Ishu."
        ),
        "",
        "## Constraint Summary",
        "",
        _markdown_table(summary, summary_columns),
        "",
        "## Selected Runs",
        "",
        _markdown_table(selected.sort_values(["constraint_policy", "seed"]), selected_columns),
        "",
        "## Grid Frontier",
        "",
        _markdown_table(
            grid.sort_values(["validation_utility_min", "validation_utility_mean"], ascending=[False, False]).head(12),
            grid_columns,
        ),
        "",
        "## Read",
        "",
        (
            f"The best accuracy/AUC constraint is `{best['constraint_policy']}` at "
            f"{best['target_accuracy_mean']:.4f} accuracy / {best['target_roc_auc_mean']:.4f} AUC, "
            f"with a {best['target_predicted_positive_rate_mean']:.4f} target fake-call rate."
        ),
        (
            f"The lowest target fake-call policy is `{lowest_fake['constraint_policy']}` at "
            f"{lowest_fake['target_predicted_positive_rate_mean']:.4f}, with "
            f"{lowest_fake['target_accuracy_mean']:.4f} accuracy."
        ),
        (
            "The frontier is therefore useful for the paper: source caps reduce source fake calls, "
            "but target fake-call bias does not move monotonically with the source cap under this "
            "small validation set."
        ),
        "",
        "## Rebuild",
        "",
        "```powershell",
        ".\\.venv\\Scripts\\python.exe scripts\\sweep_reverse_tuned_fusion_constraints.py",
        "```",
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    grid, selected, summary = run_sweep(args)
    summary_dir = Path(args.summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    grid_path = summary_dir / "ms_cocoai_to_ishu_tuned_fusion_constraint_sweep_grid.csv"
    selected_path = summary_dir / "ms_cocoai_to_ishu_tuned_fusion_constraint_sweep_selected.csv"
    summary_path = summary_dir / "ms_cocoai_to_ishu_tuned_fusion_constraint_sweep_summary.csv"
    grid.to_csv(grid_path, index=False)
    selected.to_csv(selected_path, index=False)
    summary.to_csv(summary_path, index=False)
    report_path = Path(args.report_path)
    write_report(summary, selected, grid, report_path)
    print(grid_path.resolve())
    print(selected_path.resolve())
    print(summary_path.resolve())
    print(report_path.resolve())


if __name__ == "__main__":
    main()
