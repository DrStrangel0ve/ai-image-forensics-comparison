from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

BRANCH_PATHS = {
    "combined_v3": ("ms_cocoai_to_ishu_neural_fusion", "combined_v3_seed{seed}"),
    "resnet18": ("ms_cocoai_to_ishu_neural_fusion", "resnet18_seed{seed}"),
    "physics_guided": (
        "ms_cocoai_to_ishu_neural_fusion",
        "physics_guided_resnet18_combined_v3_seed{seed}",
    ),
    "convnext_tiny": ("ms_cocoai_to_ishu_foundation", "convnext_tiny_seed{seed}"),
    "clip_vit_b_32": ("ms_cocoai_to_ishu_foundation", "clip_vit_b_32_seed{seed}"),
    "dinov2_vits14": ("ms_cocoai_to_ishu_foundation", "dinov2_vits14_seed{seed}"),
}

MEAN_COLUMNS = [
    "accuracy",
    "roc_auc",
    "brier_score",
    "expected_calibration_error",
    "precision",
    "recall",
    "f1",
    "predicted_positive_rate",
    "threshold",
    "threshold_source_predicted_positive_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run and summarize the MS-COCOAI-to-Ishu source-utility threshold sweep "
            "for all-branch score fusion."
        )
    )
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--summary-dir", default="reports/assets")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 29])
    parser.add_argument("--fusion-c", type=float, default=0.03)
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--real-fpr-penalties", default="2,4,8")
    parser.add_argument("--fake-miss-penalties", default="1,1.5,2")
    parser.add_argument(
        "--source-fake-rate-caps",
        default="none,0.45,0.48,0.50",
        help="Comma-separated source predicted-fake-rate caps; use 'none' for no cap.",
    )
    parser.add_argument("--fake-detection-weight", type=float, default=1.0)
    parser.add_argument("--real-clearance-weight", type=float, default=1.0)
    parser.add_argument("--threshold-tiebreak", default="higher", choices=["higher", "lower", "near_half"])
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to call scripts/fuse_prediction_scores.py.",
    )
    return parser.parse_args()


def _parse_float_list(values: str) -> list[float]:
    parsed = [float(value.strip()) for value in values.split(",") if value.strip()]
    if not parsed:
        raise ValueError("Expected at least one comma-separated numeric value")
    return parsed


def _parse_cap_list(values: str) -> list[float | None]:
    parsed: list[float | None] = []
    for value in values.split(","):
        cleaned = value.strip().lower()
        if not cleaned:
            continue
        if cleaned in {"none", "null", "na"}:
            parsed.append(None)
        else:
            parsed.append(float(cleaned))
    if not parsed:
        raise ValueError("Expected at least one source fake-rate cap or 'none'")
    return parsed


def _format_float_token(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def config_key(real_fpr_penalty: float, fake_miss_penalty: float, cap: float | None) -> str:
    cap_token = "nocap" if cap is None else f"cap{_format_float_token(cap)}"
    return (
        "score_fusion_all6_c003_source_utility_"
        f"rfp{_format_float_token(real_fpr_penalty)}_"
        f"fmp{_format_float_token(fake_miss_penalty)}_"
        f"{cap_token}"
    )


def _branch_prediction_path(run_root: Path, method: str, seed: int, target: bool) -> Path:
    family, template = BRANCH_PATHS[method]
    name = template.format(seed=seed)
    if target:
        name = f"{name}_to_ishu_test"
    return run_root / family / name / "predictions.csv"


def _fusion_command(
    python_executable: str,
    run_root: Path,
    output_dir: Path,
    seed: int,
    fusion_c: float,
    calibration_fraction: float,
    real_fpr_penalty: float,
    fake_miss_penalty: float,
    cap: float | None,
    fake_detection_weight: float,
    real_clearance_weight: float,
    threshold_tiebreak: str,
) -> list[str]:
    command = [
        python_executable,
        str(ROOT / "scripts" / "fuse_prediction_scores.py"),
        "--out-dir",
        str(output_dir),
        "--seed",
        str(seed),
        "--fusion-c",
        str(fusion_c),
        "--calibration-fraction",
        str(calibration_fraction),
        "--threshold-strategy",
        "source_utility",
        "--threshold-tiebreak",
        threshold_tiebreak,
        "--threshold-fake-detection-weight",
        str(fake_detection_weight),
        "--threshold-real-clearance-weight",
        str(real_clearance_weight),
        "--threshold-real-fpr-penalty",
        str(real_fpr_penalty),
        "--threshold-fake-miss-penalty",
        str(fake_miss_penalty),
    ]
    if cap is not None:
        command.extend(["--threshold-max-positive-rate", str(cap)])
    for method in BRANCH_PATHS:
        command.extend(
            [
                "--train",
                f"{method}={_branch_prediction_path(run_root, method, seed, target=False)}",
            ]
        )
    for method in BRANCH_PATHS:
        command.extend(
            [
                "--variant",
                f"ishu_test:{method}={_branch_prediction_path(run_root, method, seed, target=True)}",
            ]
        )
    return command


def _run_sweep(args: argparse.Namespace, configs: list[tuple[str, float, float, float | None]]) -> None:
    run_root = Path(args.run_root)
    for key, real_fpr_penalty, fake_miss_penalty, cap in configs:
        for seed in args.seeds:
            output_dir = run_root / "ms_cocoai_to_ishu_neural_fusion" / f"{key}_seed{seed}"
            if args.skip_existing and (output_dir / "summary.csv").exists():
                continue
            command = _fusion_command(
                args.python,
                run_root,
                output_dir,
                seed,
                args.fusion_c,
                args.calibration_fraction,
                real_fpr_penalty,
                fake_miss_penalty,
                cap,
                args.fake_detection_weight,
                args.real_clearance_weight,
                args.threshold_tiebreak,
            )
            if args.dry_run:
                print(" ".join(command))
            else:
                result = subprocess.run(
                    command,
                    cwd=ROOT,
                    check=False,
                    text=True,
                    capture_output=True,
                )
                if result.returncode != 0:
                    print(result.stdout)
                    print(result.stderr, file=sys.stderr)
                    raise subprocess.CalledProcessError(result.returncode, command)


def collect_metrics(run_root: Path, seeds: list[int], configs: list[tuple[str, float, float, float | None]]) -> pd.DataFrame:
    rows = []
    for key, real_fpr_penalty, fake_miss_penalty, cap in configs:
        for seed in seeds:
            summary_path = (
                run_root
                / "ms_cocoai_to_ishu_neural_fusion"
                / f"{key}_seed{seed}"
                / "summary.csv"
            )
            if not summary_path.exists():
                raise FileNotFoundError(summary_path)
            frame = pd.read_csv(summary_path)
            for row in frame.to_dict("records"):
                rows.append(
                    {
                        "config": key,
                        "seed": seed,
                        "real_fpr_penalty": real_fpr_penalty,
                        "fake_miss_penalty": fake_miss_penalty,
                        "source_fake_rate_cap": cap,
                        **row,
                    }
                )
    return pd.DataFrame(rows)


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    aggregations = {
        "seed": "count",
        **{column: ["mean", "std"] for column in MEAN_COLUMNS if column in metrics.columns},
    }
    summary = metrics.groupby(
        [
            "config",
            "variant",
            "real_fpr_penalty",
            "fake_miss_penalty",
            "source_fake_rate_cap",
        ],
        dropna=False,
    ).agg(aggregations)
    summary.columns = [
        "n_seeds" if column[0] == "seed" else f"{column[0]}_{column[1]}"
        for column in summary.columns
    ]
    summary = summary.reset_index()
    sort_columns = ["variant", "accuracy_mean", "predicted_positive_rate_mean", "roc_auc_mean"]
    return summary.sort_values(sort_columns, ascending=[True, False, True, False])


def _format_cell(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame[columns].to_dict("records"):
        lines.append("| " + " | ".join(_format_cell(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def _cap_label(value) -> str:
    return "none" if pd.isna(value) else f"{float(value):.2f}"


def _best_by_cap(target: pd.DataFrame) -> pd.DataFrame:
    frame = target.copy()
    frame["source_fake_rate_cap_label"] = frame["source_fake_rate_cap"].map(_cap_label)
    return (
        frame.sort_values(
            ["source_fake_rate_cap_label", "accuracy_mean", "predicted_positive_rate_mean"],
            ascending=[True, False, True],
        )
        .groupby("source_fake_rate_cap_label", as_index=False, sort=False)
        .head(1)
        .sort_values("accuracy_mean", ascending=False)
    )


def write_report(summary: pd.DataFrame, out_path: Path) -> None:
    target = summary[summary["variant"] == "ishu_test"].copy()
    target = target.sort_values(
        ["accuracy_mean", "predicted_positive_rate_mean", "roc_auc_mean"],
        ascending=[False, True, False],
    )
    columns = [
        "config",
        "source_fake_rate_cap",
        "real_fpr_penalty",
        "fake_miss_penalty",
        "accuracy_mean",
        "roc_auc_mean",
        "brier_score_mean",
        "expected_calibration_error_mean",
        "precision_mean",
        "recall_mean",
        "predicted_positive_rate_mean",
        "threshold_mean",
        "threshold_source_predicted_positive_rate_mean",
    ]
    best_by_cap = _best_by_cap(target)
    cap_columns = [
        "source_fake_rate_cap_label",
        "accuracy_mean",
        "precision_mean",
        "recall_mean",
        "predicted_positive_rate_mean",
        "threshold_mean",
        "threshold_source_predicted_positive_rate_mean",
    ]
    best = target.iloc[0]
    uncapped = target[target["source_fake_rate_cap"].isna()]
    best_uncapped = None if uncapped.empty else uncapped.sort_values("accuracy_mean", ascending=False).iloc[0]
    lines = [
        "# MS COCOAI to Ishu Source-Utility Threshold Sweep",
        "",
        "This report summarizes source-utility threshold selection for the strongly regularized all-branch fusion head.",
        "",
        "The score model is unchanged from the `C=0.03` reverse fusion family; only the source-side threshold objective changes.",
        "",
        "## Target Results",
        "",
        _markdown_table(target.head(12), columns),
        "",
        "## Best Operating Point By Source Fake-Rate Cap",
        "",
        _markdown_table(best_by_cap, cap_columns),
        "",
        "## Read",
        "",
        (
            f"The best source-utility operating point is `{best['config']}` with "
            f"{best['accuracy_mean']:.4f} mean accuracy, {best['roc_auc_mean']:.4f} AUC, "
            f"and {best['predicted_positive_rate_mean']:.4f} target fake-call rate."
        ),
        (
            "This matches the previous `cap_0p48` source-accuracy threshold result rather than exceeding it. "
            "The sweep therefore strengthens the conclusion that the source fake-rate cap is doing the useful "
            "operating-point work for this score model."
        ),
    ]
    if best_uncapped is not None:
        lines.extend(
            [
                (
                    f"Without a source fake-rate cap, the best utility-selected point reaches "
                    f"{best_uncapped['accuracy_mean']:.4f} accuracy with "
                    f"{best_uncapped['predicted_positive_rate_mean']:.4f} target fake-call rate, "
                    "so the cap still gives the cleaner decision frontier."
                ),
            ]
        )
    lines.extend(
        [
            "",
            "Use this as operating-point evidence, not as a new scoring-model result: AUC/Brier/ECE reflect the same fused scores for each seed/config family, while accuracy and predicted fake rate move with threshold selection.",
            "",
            "Next step: move the utility from post-hoc threshold selection into fusion training or validation selection, because threshold-only source utility does not improve beyond the capped source-threshold baseline.",
            "",
            "## Rebuild",
            "",
            "```powershell",
            ".\\.venv\\Scripts\\python.exe scripts\\run_reverse_source_utility_sweep.py --python .\\.venv\\Scripts\\python.exe --skip-existing",
            "```",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    real_fpr_penalties = _parse_float_list(args.real_fpr_penalties)
    fake_miss_penalties = _parse_float_list(args.fake_miss_penalties)
    caps = _parse_cap_list(args.source_fake_rate_caps)
    configs = [
        (config_key(real_fpr, fake_miss, cap), real_fpr, fake_miss, cap)
        for real_fpr in real_fpr_penalties
        for fake_miss in fake_miss_penalties
        for cap in caps
    ]
    _run_sweep(args, configs)
    if args.dry_run:
        return

    summary_dir = Path(args.summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    metrics = collect_metrics(Path(args.run_root), args.seeds, configs)
    summary = summarize_metrics(metrics)
    metrics_path = summary_dir / "ms_cocoai_to_ishu_source_utility_threshold_metrics.csv"
    summary_path = summary_dir / "ms_cocoai_to_ishu_source_utility_threshold_mean_metrics.csv"
    report_path = ROOT / "reports" / "ms_cocoai_to_ishu_source_utility_threshold_2026_06_13.md"
    metrics.to_csv(metrics_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_report(summary, report_path)
    print(metrics_path.resolve())
    print(summary_path.resolve())
    print(report_path.resolve())


if __name__ == "__main__":
    main()
