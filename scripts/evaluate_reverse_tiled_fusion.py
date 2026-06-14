from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sweep_reverse_tuned_fusion_constraints import summarize_by_cap
from tune_reverse_fusion_source_holdout import (
    _markdown_table,
    _parse_dropout_configs,
    _path_key,
    decision_metrics,
    fit_scores,
    load_seed_frames,
    select_threshold,
)


DEFAULT_METHODS = [
    "combined_v3",
    "resnet18",
    "physics_guided",
    "convnext_tiny",
    "clip_vit_b_32",
    "dinov2_vits14",
]
DEFAULT_TILE_SCORE_MODES = ["global", "tile_mean", "tile_max", "tile_top2_mean"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate reverse tuned fusion after replacing one target branch with native-tile scores."
        )
    )
    parser.add_argument("--run-root", default="runs")
    parser.add_argument(
        "--metadata",
        default="data/raw/ms_cocoai_2026_validation_source_balanced_100/metadata.csv",
    )
    parser.add_argument(
        "--selected-configs",
        default="reports/assets/ms_cocoai_to_ishu_tuned_fusion_constraint_sweep_selected.csv",
    )
    parser.add_argument(
        "--tile-detail",
        default="reports/assets/ms_cocoai_to_ishu_combined_v3_native_tiling_detail.csv",
    )
    parser.add_argument("--tile-branch", default="combined_v3")
    parser.add_argument("--tile-score-modes", nargs="+", default=DEFAULT_TILE_SCORE_MODES)
    parser.add_argument("--constraint-policy", default="cap_0p4")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 29])
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--dropout-configs", default="none,mean0p35x8")
    parser.add_argument("--threshold-tiebreak", choices=["higher", "lower", "near_half"], default="higher")
    parser.add_argument("--fake-detection-weight", type=float, default=1.0)
    parser.add_argument("--real-clearance-weight", type=float, default=1.0)
    parser.add_argument("--real-fpr-penalty", type=float, default=4.0)
    parser.add_argument("--fake-miss-penalty", type=float, default=1.5)
    parser.add_argument("--alignment-tolerance", type=float, default=1e-5)
    parser.add_argument("--summary-dir", default="reports/assets")
    parser.add_argument(
        "--asset-prefix",
        default="ms_cocoai_to_ishu_tuned_fusion_native_tiling",
        help="Prefix for detail and summary CSV assets written under --summary-dir.",
    )
    parser.add_argument(
        "--report-path",
        default="reports/ms_cocoai_to_ishu_tuned_fusion_native_tiling_2026_06_13.md",
    )
    return parser.parse_args()


def _score_column(score_mode: str) -> str:
    return score_mode if score_mode.endswith("_score") else f"{score_mode}_score"


def _selected_config(selected: pd.DataFrame, seed: int, policy: str) -> pd.Series:
    matches = selected[
        (selected["seed"].astype(int) == int(seed))
        & (selected["constraint_policy"] == policy)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one selected config for seed={seed}, policy={policy}; got {len(matches)}")
    return matches.iloc[0]


def load_tiled_target(
    clean_target: pd.DataFrame,
    tile_detail: pd.DataFrame,
    seed: int,
    branch: str,
    score_mode: str,
    alignment_tolerance: float = 1e-5,
) -> pd.DataFrame:
    if branch not in clean_target.columns:
        raise ValueError(f"Target frame does not contain branch {branch!r}")
    score_column = _score_column(score_mode)
    if score_column not in tile_detail.columns:
        raise ValueError(f"Tile detail is missing score column {score_column!r}")
    seed_detail = tile_detail[tile_detail["seed"].astype(int) == int(seed)].reset_index(drop=True)
    target = clean_target.reset_index(drop=True).copy()
    if len(seed_detail) != len(target):
        raise ValueError(
            f"Seed {seed} tile detail has {len(seed_detail)} rows, target has {len(target)}"
        )

    detail_by_path = seed_detail.assign(path_key=seed_detail["path"].map(_path_key))
    target_keys = set(target["path_key"])
    if set(detail_by_path["path_key"]) == target_keys:
        replacement = target[["path_key"]].merge(
            detail_by_path[["path_key", "y_true", score_column]],
            on="path_key",
            how="left",
            validate="one_to_one",
        )
        if not (replacement["y_true"].astype(int).to_numpy() == target["y_true"].astype(int).to_numpy()).all():
            raise ValueError(f"Seed {seed} path-aligned tile labels do not match target labels")
        target[branch] = replacement[score_column].astype(float).to_numpy()
        return target

    labels_match = seed_detail["y_true"].astype(int).to_numpy() == target["y_true"].astype(int).to_numpy()
    if not labels_match.all():
        raise ValueError(
            f"Seed {seed} tile detail paths do not match and ordered labels differ at "
            f"{int((~labels_match).sum())} rows"
        )
    if "global_score" not in seed_detail.columns:
        raise ValueError("Order-based tile alignment requires a global_score column")
    max_delta = float(
        np.max(
            np.abs(
                seed_detail["global_score"].astype(float).to_numpy()
                - target[branch].astype(float).to_numpy()
            )
        )
    )
    if max_delta > alignment_tolerance:
        raise ValueError(
            f"Seed {seed} order alignment failed: max global-score delta {max_delta:.6g} "
            f"exceeds tolerance {alignment_tolerance:.6g}"
        )
    target[branch] = seed_detail[score_column].astype(float).to_numpy()
    return target


def evaluate_seed(
    args: argparse.Namespace,
    seed: int,
    selected_config: pd.Series,
    dropout_by_label: dict,
    tile_detail: pd.DataFrame,
) -> list[dict[str, float | str | int]]:
    source, clean_target, method_order = load_seed_frames(
        Path(args.run_root),
        Path(args.metadata),
        list(args.methods),
        seed,
    )
    cap = selected_config["source_fake_rate_cap"]
    cap = None if pd.isna(cap) else float(cap)
    rows = []
    for score_mode in args.tile_score_modes:
        tiled_target = load_tiled_target(
            clean_target,
            tile_detail,
            seed,
            args.tile_branch,
            score_mode,
            alignment_tolerance=args.alignment_tolerance,
        )
        _model, source_scores, target_scores = fit_scores(
            source,
            tiled_target,
            method_order,
            float(selected_config["fusion_c"]),
            dropout_by_label[str(selected_config["dropout_config"])],
            seed + 9091,
        )
        threshold, source_utility, source_positive_rate = select_threshold(
            source["y_true"].to_numpy(dtype=int),
            source_scores,
            cap,
            tiebreak=args.threshold_tiebreak,
            fake_detection_weight=args.fake_detection_weight,
            real_clearance_weight=args.real_clearance_weight,
            real_fpr_penalty=args.real_fpr_penalty,
            fake_miss_penalty=args.fake_miss_penalty,
        )
        source_metrics = decision_metrics(
            source["y_true"].to_numpy(dtype=int),
            source_scores,
            threshold,
            fake_detection_weight=args.fake_detection_weight,
            real_clearance_weight=args.real_clearance_weight,
            real_fpr_penalty=args.real_fpr_penalty,
            fake_miss_penalty=args.fake_miss_penalty,
        )
        target_metrics = decision_metrics(
            tiled_target["y_true"].to_numpy(dtype=int),
            target_scores,
            threshold,
            fake_detection_weight=args.fake_detection_weight,
            real_clearance_weight=args.real_clearance_weight,
            real_fpr_penalty=args.real_fpr_penalty,
            fake_miss_penalty=args.fake_miss_penalty,
        )
        rows.append(
            {
                "score_mode": score_mode,
                "constraint_policy": args.constraint_policy,
                "seed": seed,
                "tile_branch": args.tile_branch,
                "fusion_c": float(selected_config["fusion_c"]),
                "dropout_config": str(selected_config["dropout_config"]),
                "source_fake_rate_cap": cap,
                "selection_validation_utility_mean": float(
                    selected_config["selection_validation_utility_mean"]
                ),
                "selection_validation_utility_min": float(
                    selected_config["selection_validation_utility_min"]
                ),
                "threshold": threshold,
                "threshold_source_utility": source_utility,
                "threshold_source_predicted_positive_rate": source_positive_rate,
                **{f"source_{key}": value for key, value in source_metrics.items()},
                **{f"target_{key}": value for key, value in target_metrics.items()},
            }
        )
    return rows


def summarize_by_score_mode(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (score_mode, policy), group in detail.groupby(["score_mode", "constraint_policy"], sort=False):
        row = {
            "score_mode": score_mode,
            "constraint_policy": policy,
            "n_seeds": int(group["seed"].nunique()),
        }
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
        ]:
            values = pd.to_numeric(group[column], errors="coerce")
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_std"] = float(values.std(ddof=1)) if len(values.dropna()) > 1 else 0.0
        row["selected_configs"] = "; ".join(
            f"seed{int(item.seed)}:C{item.fusion_c:g}:{item.dropout_config}:cap{item.source_fake_rate_cap}"
            for item in group.sort_values("seed").itertuples(index=False)
        )
        rows.append(row)
    return pd.DataFrame(rows)


def run_tiled_fusion(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = pd.read_csv(args.selected_configs)
    tile_detail = pd.read_csv(args.tile_detail)
    dropouts = _parse_dropout_configs(args.dropout_configs)
    dropout_by_label = {dropout.label: dropout for dropout in dropouts}
    rows = []
    for seed in args.seeds:
        config = _selected_config(selected, seed, args.constraint_policy)
        rows.extend(evaluate_seed(args, seed, config, dropout_by_label, tile_detail))
    detail = pd.DataFrame(rows)
    summary = summarize_by_score_mode(detail)
    return detail, summary


def write_report(
    summary: pd.DataFrame,
    detail: pd.DataFrame,
    clean_selected: pd.DataFrame,
    out_path: Path,
    rebuild_command: str,
) -> None:
    clean_policy = detail["constraint_policy"].iloc[0]
    clean_summary = summarize_by_cap(clean_selected)
    clean = clean_summary[clean_summary["constraint_policy"] == clean_policy].iloc[0]
    best_accuracy = summary.sort_values("target_accuracy_mean", ascending=False).iloc[0]
    best_auc = summary.sort_values("target_roc_auc_mean", ascending=False).iloc[0]
    branch = detail["tile_branch"].iloc[0]
    summary_columns = [
        "score_mode",
        "constraint_policy",
        "n_seeds",
        "target_accuracy_mean",
        "target_roc_auc_mean",
        "target_brier_score_mean",
        "target_expected_calibration_error_mean",
        "target_predicted_positive_rate_mean",
        "source_predicted_positive_rate_mean",
    ]
    detail_columns = [
        "score_mode",
        "seed",
        "fusion_c",
        "dropout_config",
        "threshold",
        "target_accuracy",
        "target_roc_auc",
        "target_predicted_positive_rate",
    ]
    lines = [
        "# MS COCOAI to Ishu Tuned-Fusion Native-Tiling Diagnostic",
        "",
        (
            f"This evaluates the selected reverse tuned-fusion operating point after replacing "
            f"the target-side `{branch}` branch with native-tile aggregate scores. All other "
            "branch scores, source training rows, selected fusion configurations, and source "
            "threshold policy stay fixed."
        ),
        "",
        "## Summary",
        "",
        _markdown_table(summary, summary_columns),
        "",
        "## Clean Comparator",
        "",
        (
            f"The clean `{clean_policy}` tuned-fusion result was {clean['target_accuracy_mean']:.4f} "
            f"accuracy / {clean['target_roc_auc_mean']:.4f} AUC with a "
            f"{clean['target_predicted_positive_rate_mean']:.4f} target fake-call rate."
        ),
        (
            f"The best native-tiling accuracy mode is `{best_accuracy['score_mode']}` at "
            f"{best_accuracy['target_accuracy_mean']:.4f}; the best ranking mode is "
            f"`{best_auc['score_mode']}` at {best_auc['target_roc_auc_mean']:.4f} AUC."
        ),
        (
            "This is a fused-stack diagnostic, not retraining of every visual branch on tiles. "
            "It tests whether local crop evidence helps after score fusion when the other "
            "branches remain at their normal global-image scores."
        ),
        "",
        "## Per-Seed Detail",
        "",
        _markdown_table(detail.sort_values(["score_mode", "seed"]), detail_columns),
        "",
        "## Rebuild",
        "",
        "```powershell",
        rebuild_command,
        "```",
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    detail, summary = run_tiled_fusion(args)
    summary_dir = Path(args.summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    detail_path = summary_dir / f"{args.asset_prefix}_detail.csv"
    summary_path = summary_dir / f"{args.asset_prefix}_summary.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    selected = pd.read_csv(args.selected_configs)
    report_path = Path(args.report_path)
    rebuild_command = (
        ".\\.venv\\Scripts\\python.exe scripts\\evaluate_reverse_tiled_fusion.py "
        f"--tile-branch {args.tile_branch} "
        f"--tile-detail {args.tile_detail} "
        f"--asset-prefix {args.asset_prefix} "
        f"--report-path {args.report_path}"
    )
    if float(args.alignment_tolerance) != 1e-5:
        rebuild_command += f" --alignment-tolerance {args.alignment_tolerance:g}"
    write_report(summary, detail, selected, report_path, rebuild_command)
    print(detail_path.resolve())
    print(summary_path.resolve())
    print(report_path.resolve())


if __name__ == "__main__":
    main()
