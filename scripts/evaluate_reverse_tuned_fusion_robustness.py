from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from sweep_reverse_tuned_fusion_constraints import summarize_by_cap
from tune_reverse_fusion_source_holdout import (
    _aligned_matrix,
    _markdown_table,
    _parse_dropout_configs,
    decision_metrics,
    fit_scores,
    load_seed_frames,
    select_threshold,
)


METHOD_DIRS = {
    "combined_v3": "combined_v3",
    "resnet18": "resnet18",
    "physics_guided": "physics_guided",
    "convnext_tiny": "convnext_tiny",
    "clip_vit_b_32": "clip_vit_b_32",
    "dinov2_vits14": "dinov2_vits14",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate source-selected reverse tuned-fusion configs on transformed Ishu targets."
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
        "--robust-root",
        default=None,
        help=(
            "Root containing seed*/method/predictions.csv folders. Defaults to "
            "runs/ms_cocoai_to_ishu_{variant}_robustness."
        ),
    )
    parser.add_argument("--variant", default="jpeg70")
    parser.add_argument("--constraint-policy", default="cap_0p4")
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
    parser.add_argument("--dropout-configs", default="none,mean0p35x8")
    parser.add_argument("--threshold-tiebreak", choices=["higher", "lower", "near_half"], default="higher")
    parser.add_argument("--fake-detection-weight", type=float, default=1.0)
    parser.add_argument("--real-clearance-weight", type=float, default=1.0)
    parser.add_argument("--real-fpr-penalty", type=float, default=4.0)
    parser.add_argument("--fake-miss-penalty", type=float, default=1.5)
    parser.add_argument("--summary-dir", default="reports/assets")
    parser.add_argument(
        "--report-path",
        default=None,
        help=(
            "Markdown report path. Defaults to "
            "reports/ms_cocoai_to_ishu_tuned_fusion_{variant}_robustness_2026_06_13.md."
        ),
    )
    return parser.parse_args()


def variant_slug(variant: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", variant.strip()).strip("_").lower()
    if not slug:
        raise ValueError("Variant slug is empty")
    return slug


def resolved_robust_root(args: argparse.Namespace) -> Path:
    if args.robust_root:
        return Path(args.robust_root)
    return Path(f"runs/ms_cocoai_to_ishu_{variant_slug(args.variant)}_robustness")


def resolved_report_path(args: argparse.Namespace) -> Path:
    if args.report_path:
        return Path(args.report_path)
    return Path(
        f"reports/ms_cocoai_to_ishu_tuned_fusion_{variant_slug(args.variant)}_robustness_2026_06_13.md"
    )


def _robust_target_paths(robust_root: Path, seed: int, methods: list[str]) -> list[tuple[str, Path]]:
    paths = []
    for method in methods:
        method_dir = METHOD_DIRS.get(method, method)
        paths.append((method, robust_root / f"seed{seed}" / method_dir / "predictions.csv"))
    return paths


def load_robust_target(robust_root: Path, seed: int, methods: list[str]) -> pd.DataFrame:
    target, target_methods = _aligned_matrix(_robust_target_paths(robust_root, seed, methods))
    if target_methods != methods:
        raise ValueError(f"Robust target methods {target_methods} do not match expected {methods}")
    return target


def _selected_config(selected: pd.DataFrame, seed: int, policy: str) -> pd.Series:
    matches = selected[
        (selected["seed"].astype(int) == int(seed))
        & (selected["constraint_policy"] == policy)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one selected config for seed={seed}, policy={policy}; got {len(matches)}")
    return matches.iloc[0]


def evaluate_seed(
    args: argparse.Namespace,
    seed: int,
    selected_config: pd.Series,
    dropout_by_label,
) -> dict[str, float | str | int]:
    source, _clean_target, method_order = load_seed_frames(
        Path(args.run_root),
        Path(args.metadata),
        list(args.methods),
        seed,
    )
    robust_target = load_robust_target(resolved_robust_root(args), seed, method_order)
    _model, source_scores, robust_scores = fit_scores(
        source,
        robust_target,
        method_order,
        float(selected_config["fusion_c"]),
        dropout_by_label[str(selected_config["dropout_config"])],
        seed + 9091,
    )
    cap = selected_config["source_fake_rate_cap"]
    cap = None if pd.isna(cap) else float(cap)
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
        robust_target["y_true"].to_numpy(dtype=int),
        robust_scores,
        threshold,
        fake_detection_weight=args.fake_detection_weight,
        real_clearance_weight=args.real_clearance_weight,
        real_fpr_penalty=args.real_fpr_penalty,
        fake_miss_penalty=args.fake_miss_penalty,
    )
    return {
        "variant": args.variant,
        "constraint_policy": args.constraint_policy,
        "seed": seed,
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


def run_robustness(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = pd.read_csv(args.selected_configs)
    dropouts = _parse_dropout_configs(args.dropout_configs)
    dropout_by_label = {dropout.label: dropout for dropout in dropouts}
    rows = []
    for seed in args.seeds:
        config = _selected_config(selected, seed, args.constraint_policy)
        rows.append(evaluate_seed(args, seed, config, dropout_by_label))
    detail = pd.DataFrame(rows)
    summary = summarize_by_cap(detail)
    summary = summary.rename(columns={"constraint_policy": "variant_policy"})
    summary.insert(0, "variant", args.variant)
    return detail, summary


def write_report(summary: pd.DataFrame, detail: pd.DataFrame, clean_selected: pd.DataFrame, out_path: Path) -> None:
    clean_summary = summarize_by_cap(clean_selected)
    clean_policy = detail["constraint_policy"].iloc[0]
    clean = clean_summary[clean_summary["constraint_policy"] == clean_policy].iloc[0]
    robust = summary.iloc[0]
    variant = str(detail["variant"].iloc[0])
    summary_columns = [
        "variant",
        "variant_policy",
        "n_seeds",
        "target_accuracy_mean",
        "target_roc_auc_mean",
        "target_brier_score_mean",
        "target_expected_calibration_error_mean",
        "target_predicted_positive_rate_mean",
        "source_predicted_positive_rate_mean",
    ]
    detail_columns = [
        "variant",
        "seed",
        "fusion_c",
        "dropout_config",
        "threshold",
        "target_accuracy",
        "target_roc_auc",
        "target_predicted_positive_rate",
    ]
    lines = [
        f"# MS COCOAI to Ishu Tuned-Fusion {variant} Robustness",
        "",
        (
            "This evaluates the already-selected reverse tuned-fusion operating point on "
            f"`{variant}`-transformed Ishu target splits. The fusion configuration and source "
            f"threshold are selected from clean MS source scores only; `{variant}` labels are used "
            "only for final evaluation."
        ),
        "",
        "## Robustness Summary",
        "",
        _markdown_table(summary, summary_columns),
        "",
        "## Per-Seed Detail",
        "",
        _markdown_table(detail.sort_values("seed"), detail_columns),
        "",
        "## Clean Comparator",
        "",
        (
            f"The clean `{clean_policy}` tuned-fusion result was {clean['target_accuracy_mean']:.4f} "
            f"accuracy / {clean['target_roc_auc_mean']:.4f} AUC with a "
            f"{clean['target_predicted_positive_rate_mean']:.4f} target fake-call rate."
        ),
        (
            f"Under `{variant}`, the same source-selected policy reaches "
            f"{robust['target_accuracy_mean']:.4f} accuracy / {robust['target_roc_auc_mean']:.4f} AUC "
            f"with a {robust['target_predicted_positive_rate_mean']:.4f} target fake-call rate."
        ),
        (
            "This is a bounded robustness check for the tuned-fusion cap frontier. "
            "Interpret it together with the clean-target result and the other transformed "
            "target variants before treating the 0.40 cap as robust rather than clean-target optimized."
        ),
        "",
        "## Rebuild",
        "",
        (
            "Prerequisite branch prediction folders are expected under "
            f"`{resolved_robust_root(argparse.Namespace(robust_root=None, variant=variant))}/seed*/`. "
            "They are produced "
            "by evaluating the saved MS-trained `combined_v3`, ResNet-18, physics-guided, "
            f"ConvNeXt, CLIP, and DINOv2 branches on the seed-specific Ishu `{variant}` folders."
        ),
        "",
        "```powershell",
        f".\\.venv\\Scripts\\python.exe scripts\\evaluate_reverse_tuned_fusion_robustness.py --variant {variant}",
        "```",
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    detail, summary = run_robustness(args)
    summary_dir = Path(args.summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    slug = variant_slug(args.variant)
    detail_path = summary_dir / f"ms_cocoai_to_ishu_tuned_fusion_{slug}_robustness_detail.csv"
    summary_path = summary_dir / f"ms_cocoai_to_ishu_tuned_fusion_{slug}_robustness_summary.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    clean_selected = pd.read_csv(args.selected_configs)
    report_path = resolved_report_path(args)
    write_report(summary, detail, clean_selected, report_path)
    print(detail_path.resolve())
    print(summary_path.resolve())
    print(report_path.resolve())


if __name__ == "__main__":
    main()
