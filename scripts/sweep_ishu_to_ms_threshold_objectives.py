from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tune_reverse_fusion_source_holdout import (
    _format_float_token,
    _markdown_table,
    _parse_cap_list,
    _parse_float_list,
    decision_metrics,
    select_threshold,
)


METRIC_COLUMNS = [
    "target_accuracy",
    "target_roc_auc",
    "target_brier_score",
    "target_expected_calibration_error",
    "target_precision",
    "target_recall",
    "target_f1",
    "target_predicted_positive_rate",
    "source_accuracy",
    "source_predicted_positive_rate",
    "threshold",
    "threshold_source_utility",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep source-threshold utility objectives for saved Ishu-to-MS "
            "SCP-Fusion prediction scores."
        )
    )
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--score-family", default="score_fusion_foundation_aligned")
    parser.add_argument("--run-template", default="ishu_seed{seed}_to_ms_cocoai_all6")
    parser.add_argument("--target-variant", default="ms_cocoai")
    parser.add_argument("--score-column", default="fake_score")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 29])
    parser.add_argument("--source-fake-rate-caps", default="none,0.45,0.50,0.55,0.60")
    parser.add_argument("--fake-detection-weights", default="0.5,1,1.5,2,3")
    parser.add_argument("--real-clearance-weights", default="1")
    parser.add_argument("--real-fpr-penalties", default="0.5,1,2,4")
    parser.add_argument("--fake-miss-penalties", default="1,1.5,2,3,4,6,8")
    parser.add_argument("--threshold-tiebreak", choices=["higher", "lower", "near_half"], default="higher")
    parser.add_argument("--summary-dir", default="reports/assets")
    parser.add_argument(
        "--report-path",
        default="reports/ishu_to_ms_threshold_objective_sweep_2026_06_13.md",
    )
    return parser.parse_args()


def policy_label(
    cap: float | None,
    fake_detection_weight: float,
    real_clearance_weight: float,
    real_fpr_penalty: float,
    fake_miss_penalty: float,
) -> str:
    cap_token = "uncapped" if cap is None else f"cap{_format_float_token(cap)}"
    return (
        f"{cap_token}_fd{_format_float_token(fake_detection_weight)}"
        f"_rc{_format_float_token(real_clearance_weight)}"
        f"_rfp{_format_float_token(real_fpr_penalty)}"
        f"_fmp{_format_float_token(fake_miss_penalty)}"
    )


def _prediction_path(
    run_root: Path,
    score_family: str,
    run_template: str,
    seed: int,
    split: str,
) -> Path:
    return run_root / score_family / run_template.format(seed=seed) / split / "predictions.csv"


def _read_scores(path: Path, score_column: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"y_true", score_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return pd.DataFrame(
        {
            "y_true": frame["y_true"].astype(int),
            "score": frame[score_column].astype(float),
        }
    )


def _metrics(prefix: str, y_true, scores, threshold: float, weights: tuple[float, float, float, float]) -> dict[str, float]:
    fake_detection_weight, real_clearance_weight, real_fpr_penalty, fake_miss_penalty = weights
    metrics = decision_metrics(
        y_true,
        scores,
        threshold,
        fake_detection_weight=fake_detection_weight,
        real_clearance_weight=real_clearance_weight,
        real_fpr_penalty=real_fpr_penalty,
        fake_miss_penalty=fake_miss_penalty,
    )
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def evaluate_fixed_baseline(args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    run_root = Path(args.run_root)
    for seed in args.seeds:
        source = _read_scores(
            _prediction_path(run_root, args.score_family, args.run_template, seed, "train"),
            args.score_column,
        )
        target = _read_scores(
            _prediction_path(
                run_root,
                args.score_family,
                args.run_template,
                seed,
                args.target_variant,
            ),
            args.score_column,
        )
        threshold = 0.5
        weights = (1.0, 1.0, 1.0, 1.0)
        rows.append(
            {
                "policy": "fixed_0p5",
                "seed": seed,
                "source_fake_rate_cap": None,
                "fake_detection_weight": 1.0,
                "real_clearance_weight": 1.0,
                "real_fpr_penalty": 1.0,
                "fake_miss_penalty": 1.0,
                "threshold": threshold,
                "threshold_source_utility": float("nan"),
                **_metrics(
                    "source",
                    source["y_true"].to_numpy(dtype=int),
                    source["score"].to_numpy(dtype=float),
                    threshold,
                    weights,
                ),
                **_metrics(
                    "target",
                    target["y_true"].to_numpy(dtype=int),
                    target["score"].to_numpy(dtype=float),
                    threshold,
                    weights,
                ),
            }
        )
    return pd.DataFrame(rows)


def run_sweep(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_root = Path(args.run_root)
    caps = _parse_cap_list(args.source_fake_rate_caps)
    fake_detection_weights = _parse_float_list(args.fake_detection_weights)
    real_clearance_weights = _parse_float_list(args.real_clearance_weights)
    real_fpr_penalties = _parse_float_list(args.real_fpr_penalties)
    fake_miss_penalties = _parse_float_list(args.fake_miss_penalties)
    rows = []
    for seed in args.seeds:
        source = _read_scores(
            _prediction_path(run_root, args.score_family, args.run_template, seed, "train"),
            args.score_column,
        )
        target = _read_scores(
            _prediction_path(
                run_root,
                args.score_family,
                args.run_template,
                seed,
                args.target_variant,
            ),
            args.score_column,
        )
        source_y = source["y_true"].to_numpy(dtype=int)
        source_scores = source["score"].to_numpy(dtype=float)
        target_y = target["y_true"].to_numpy(dtype=int)
        target_scores = target["score"].to_numpy(dtype=float)
        for cap in caps:
            for fake_detection_weight in fake_detection_weights:
                for real_clearance_weight in real_clearance_weights:
                    for real_fpr_penalty in real_fpr_penalties:
                        for fake_miss_penalty in fake_miss_penalties:
                            weights = (
                                fake_detection_weight,
                                real_clearance_weight,
                                real_fpr_penalty,
                                fake_miss_penalty,
                            )
                            threshold, source_utility, source_positive_rate = select_threshold(
                                source_y,
                                source_scores,
                                cap,
                                tiebreak=args.threshold_tiebreak,
                                fake_detection_weight=fake_detection_weight,
                                real_clearance_weight=real_clearance_weight,
                                real_fpr_penalty=real_fpr_penalty,
                                fake_miss_penalty=fake_miss_penalty,
                            )
                            rows.append(
                                {
                                    "policy": policy_label(cap, *weights),
                                    "seed": seed,
                                    "source_fake_rate_cap": cap,
                                    "fake_detection_weight": fake_detection_weight,
                                    "real_clearance_weight": real_clearance_weight,
                                    "real_fpr_penalty": real_fpr_penalty,
                                    "fake_miss_penalty": fake_miss_penalty,
                                    "threshold": threshold,
                                    "threshold_source_utility": source_utility,
                                    "threshold_source_predicted_positive_rate": source_positive_rate,
                                    **_metrics("source", source_y, source_scores, threshold, weights),
                                    **_metrics("target", target_y, target_scores, threshold, weights),
                                }
                            )
    detail = pd.concat([evaluate_fixed_baseline(args), pd.DataFrame(rows)], ignore_index=True)
    return detail, summarize(detail)


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = [
        "policy",
        "source_fake_rate_cap",
        "fake_detection_weight",
        "real_clearance_weight",
        "real_fpr_penalty",
        "fake_miss_penalty",
    ]
    for keys, group in detail.groupby(group_columns, dropna=False, sort=False):
        (
            policy,
            cap,
            fake_detection_weight,
            real_clearance_weight,
            real_fpr_penalty,
            fake_miss_penalty,
        ) = keys
        row = {
            "policy": policy,
            "n_seeds": int(group["seed"].nunique()),
            "source_fake_rate_cap": cap,
            "fake_detection_weight": fake_detection_weight,
            "real_clearance_weight": real_clearance_weight,
            "real_fpr_penalty": real_fpr_penalty,
            "fake_miss_penalty": fake_miss_penalty,
        }
        for column in METRIC_COLUMNS:
            values = pd.to_numeric(group[column], errors="coerce")
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_std"] = float(values.std(ddof=1)) if len(values.dropna()) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["target_accuracy_mean", "target_predicted_positive_rate_mean"],
        ascending=[False, False],
    )


def write_report(summary: pd.DataFrame, detail: pd.DataFrame, out_path: Path) -> None:
    top = summary.iloc[0]
    fixed = summary[summary["policy"] == "fixed_0p5"].iloc[0]
    reverse_like = summary[
        (summary["fake_detection_weight"] == 1.0)
        & (summary["real_clearance_weight"] == 1.0)
        & (summary["real_fpr_penalty"] == 4.0)
        & (summary["fake_miss_penalty"] == 1.5)
    ].head(8)
    display_summary = summary.drop_duplicates(
        subset=[
            "target_accuracy_mean",
            "target_predicted_positive_rate_mean",
            "source_predicted_positive_rate_mean",
            "threshold_mean",
        ]
    )
    summary_columns = [
        "policy",
        "n_seeds",
        "target_accuracy_mean",
        "target_roc_auc_mean",
        "target_brier_score_mean",
        "target_expected_calibration_error_mean",
        "target_predicted_positive_rate_mean",
        "source_predicted_positive_rate_mean",
        "threshold_mean",
    ]
    detail_columns = [
        "policy",
        "seed",
        "threshold",
        "target_accuracy",
        "target_roc_auc",
        "target_predicted_positive_rate",
        "source_predicted_positive_rate",
    ]
    lines = [
        "# Ishu to MS Threshold Objective Sweep",
        "",
        (
            "This diagnostic sweeps source-selected decision objectives on saved "
            "Ishu-trained all-foundation SCP-Fusion scores. It does not retrain branches; "
            "it asks whether the fixed score ranker has a better source-derived operating point."
        ),
        "",
        "## Top Objective Sensitivity Rows",
        "",
        _markdown_table(display_summary.head(12), summary_columns),
        "",
        "## Fixed Threshold Baseline",
        "",
        _markdown_table(summary[summary["policy"] == "fixed_0p5"], summary_columns),
        "",
        "## Reverse-Like Utility Family",
        "",
        _markdown_table(reverse_like, summary_columns),
        "",
        "## Per-Seed Rows for Best Policy",
        "",
        _markdown_table(detail[detail["policy"] == top["policy"]].sort_values("seed"), detail_columns),
        "",
        "## Read",
        "",
        (
            f"The strongest target-accuracy diagnostic row is `{top['policy']}` at "
            f"{top['target_accuracy_mean']:.4f} accuracy / {top['target_roc_auc_mean']:.4f} AUC, "
            f"with a {top['target_predicted_positive_rate_mean']:.4f} target fake-call rate."
        ),
        (
            f"The fixed 0.5 threshold baseline is {fixed['target_accuracy_mean']:.4f} accuracy / "
            f"{fixed['target_roc_auc_mean']:.4f} AUC with a "
            f"{fixed['target_predicted_positive_rate_mean']:.4f} target fake-call rate."
        ),
        (
            "The reverse-transfer utility family is too conservative in this direction: source fake-rate "
            "caps mostly raise thresholds and make the model miss generated images. A less punitive "
            "real-FPR objective improves decision accuracy, but this sweep should be treated as a "
            "sensitivity result until the utility family is selected without target feedback."
        ),
        "",
        "## Rebuild",
        "",
        "```powershell",
        ".\\.venv\\Scripts\\python.exe scripts\\sweep_ishu_to_ms_threshold_objectives.py",
        "```",
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    detail, summary = run_sweep(args)
    summary_dir = Path(args.summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    detail_path = summary_dir / "ishu_to_ms_threshold_objective_sweep_detail.csv"
    summary_path = summary_dir / "ishu_to_ms_threshold_objective_sweep_summary.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    report_path = Path(args.report_path)
    write_report(summary, detail, report_path)
    print(detail_path.resolve())
    print(summary_path.resolve())
    print(report_path.resolve())


if __name__ == "__main__":
    main()
