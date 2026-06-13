from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CANDIDATES = [
    "score_fusion_all6",
    "score_fusion_all6_temp_balanced",
    "score_fusion_all6_c01",
    "score_fusion_all6_c01_temp_balanced",
    "score_fusion_all6_c003",
    "score_fusion_all6_dropout_mean_r35x8",
    "score_fusion_all6_dropout_mean_r35x8_temp_balanced",
    "score_fusion_all6_c003_source_acc_cap_0p45",
    "score_fusion_all6_c003_source_acc_cap_0p48",
    "score_fusion_all6_c003_source_acc_cap_0p50",
    "score_fusion_all6_c003_source_utility_rfp2_fmp1_cap0p45",
    "score_fusion_all6_c003_source_utility_rfp2_fmp1_cap0p48",
    "score_fusion_all6_c003_source_utility_rfp2_fmp1_cap0p5",
    "score_fusion_all6_c003_source_utility_rfp2_fmp1_nocap",
]

TARGET_METRICS = [
    "target_accuracy",
    "target_roc_auc",
    "target_brier_score",
    "target_expected_calibration_error",
    "target_precision",
    "target_recall",
    "target_f1",
    "target_predicted_positive_rate",
]

SOURCE_METRICS = [
    "source_utility",
    "source_accuracy",
    "source_precision",
    "source_recall",
    "source_specificity",
    "source_real_fpr",
    "source_fake_miss_rate",
    "source_predicted_positive_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select reverse MS-COCOAI-to-Ishu fusion model families by source-side "
            "forensic utility, then summarize target performance without using target "
            "labels for selection."
        )
    )
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--family-dir", default="ms_cocoai_to_ishu_neural_fusion")
    parser.add_argument("--summary-dir", default="reports/assets")
    parser.add_argument(
        "--report-path",
        default="reports/ms_cocoai_to_ishu_model_utility_selection_2026_06_13.md",
    )
    parser.add_argument("--variant", default="ishu_test")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 29])
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help=(
            "Candidate run prefix without _seedN. Repeat to override the default "
            "reverse fusion candidate suite."
        ),
    )
    parser.add_argument(
        "--source-positive-rate-caps",
        default="none,0.50,0.48",
        help="Comma-separated source fake-call caps for selection; use 'none' for no cap.",
    )
    parser.add_argument("--fake-detection-weight", type=float, default=1.0)
    parser.add_argument("--real-clearance-weight", type=float, default=1.0)
    parser.add_argument("--real-fpr-penalty", type=float, default=4.0)
    parser.add_argument("--fake-miss-penalty", type=float, default=1.5)
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def _parse_cap_list(values: str) -> list[float | None]:
    caps: list[float | None] = []
    for value in values.split(","):
        cleaned = value.strip().lower()
        if not cleaned:
            continue
        if cleaned in {"none", "null", "na"}:
            caps.append(None)
        else:
            caps.append(float(cleaned))
    if not caps:
        raise ValueError("Expected at least one source positive-rate cap or 'none'")
    return caps


def _format_float_token(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def policy_label(cap: float | None) -> str:
    if cap is None:
        return "source_utility_unconstrained"
    return f"source_utility_cap_{_format_float_token(cap)}"


def _summary_row(summary_path: Path, variant: str) -> dict:
    frame = pd.read_csv(summary_path)
    matches = frame[frame["variant"] == variant]
    if matches.empty:
        raise ValueError(f"{summary_path} does not contain variant={variant!r}")
    return matches.iloc[0].to_dict()


def _threshold_from_row(row: dict) -> float:
    value = row.get("threshold", 0.5)
    if pd.isna(value):
        return 0.5
    return float(value)


def _float_or_na(row: dict, key: str):
    value = row.get(key, pd.NA)
    if pd.isna(value):
        return float("nan")
    return float(value)


def source_decision_metrics(
    predictions_path: Path,
    threshold: float,
    fake_detection_weight: float = 1.0,
    real_clearance_weight: float = 1.0,
    real_fpr_penalty: float = 4.0,
    fake_miss_penalty: float = 1.5,
) -> dict[str, float]:
    frame = pd.read_csv(predictions_path)
    required = {"y_true", "fake_score"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{predictions_path} is missing prediction columns: {sorted(missing)}")
    y_true = frame["y_true"].astype(int).to_numpy()
    scores = frame["fake_score"].astype(float).to_numpy()
    if len(y_true) == 0:
        raise ValueError(f"{predictions_path} has no prediction rows")
    if not np.isfinite(scores).all():
        raise ValueError(f"{predictions_path} contains non-finite scores")
    predicted = (scores >= threshold).astype(int)
    true_positive = float(((predicted == 1) & (y_true == 1)).sum())
    false_positive = float(((predicted == 1) & (y_true == 0)).sum())
    true_negative = float(((predicted == 0) & (y_true == 0)).sum())
    false_negative = float(((predicted == 0) & (y_true == 1)).sum())
    positive_total = max(true_positive + false_negative, 1.0)
    negative_total = max(true_negative + false_positive, 1.0)
    recall = true_positive / positive_total
    specificity = true_negative / negative_total
    precision = true_positive / max(true_positive + false_positive, 1.0)
    real_fpr = 1.0 - specificity
    fake_miss_rate = 1.0 - recall
    utility = (
        fake_detection_weight * recall
        + real_clearance_weight * specificity
        - real_fpr_penalty * real_fpr
        - fake_miss_penalty * fake_miss_rate
    )
    return {
        "source_utility": float(utility),
        "source_accuracy": float((predicted == y_true).mean()),
        "source_precision": float(precision),
        "source_recall": float(recall),
        "source_specificity": float(specificity),
        "source_real_fpr": float(real_fpr),
        "source_fake_miss_rate": float(fake_miss_rate),
        "source_positive_rate": float(y_true.mean()),
        "source_predicted_positive_rate": float(predicted.mean()),
    }


def collect_candidate_metrics(
    run_root: Path,
    family_dir: str,
    seeds: list[int],
    candidates: list[str],
    variant: str,
    fake_detection_weight: float = 1.0,
    real_clearance_weight: float = 1.0,
    real_fpr_penalty: float = 4.0,
    fake_miss_penalty: float = 1.5,
    allow_missing: bool = False,
) -> pd.DataFrame:
    rows = []
    family_root = run_root / family_dir
    for candidate in candidates:
        for seed in seeds:
            run_dir = family_root / f"{candidate}_seed{seed}"
            summary_path = run_dir / "summary.csv"
            predictions_path = run_dir / "train" / "predictions.csv"
            if not summary_path.exists() or not predictions_path.exists():
                if allow_missing:
                    continue
                missing = summary_path if not summary_path.exists() else predictions_path
                raise FileNotFoundError(missing)
            train_row = _summary_row(summary_path, "train")
            target_row = _summary_row(summary_path, variant)
            threshold = _threshold_from_row(train_row)
            source_metrics = source_decision_metrics(
                predictions_path,
                threshold,
                fake_detection_weight=fake_detection_weight,
                real_clearance_weight=real_clearance_weight,
                real_fpr_penalty=real_fpr_penalty,
                fake_miss_penalty=fake_miss_penalty,
            )
            rows.append(
                {
                    "candidate": candidate,
                    "seed": seed,
                    "run_dir": str(run_dir).replace("\\", "/"),
                    "threshold": threshold,
                    "threshold_strategy": train_row.get("threshold_strategy", "fixed"),
                    "threshold_source": train_row.get("threshold_source", "fixed"),
                    "threshold_max_positive_rate": _float_or_na(
                        train_row, "threshold_max_positive_rate"
                    ),
                    "summary_source_predicted_positive_rate": _float_or_na(
                        train_row, "predicted_positive_rate"
                    ),
                    **source_metrics,
                    "target_variant": variant,
                    "target_accuracy": _float_or_na(target_row, "accuracy"),
                    "target_precision": _float_or_na(target_row, "precision"),
                    "target_recall": _float_or_na(target_row, "recall"),
                    "target_f1": _float_or_na(target_row, "f1"),
                    "target_roc_auc": _float_or_na(target_row, "roc_auc"),
                    "target_brier_score": _float_or_na(target_row, "brier_score"),
                    "target_expected_calibration_error": _float_or_na(
                        target_row, "expected_calibration_error"
                    ),
                    "target_predicted_positive_rate": _float_or_na(
                        target_row, "predicted_positive_rate"
                    ),
                }
            )
    if not rows:
        raise ValueError("No candidate metrics were collected")
    return pd.DataFrame(rows)


def select_candidates(metrics: pd.DataFrame, cap: float | None) -> pd.DataFrame:
    selected = []
    for seed in sorted(metrics["seed"].unique()):
        seed_frame = metrics[metrics["seed"] == seed].copy()
        if cap is not None:
            seed_frame = seed_frame[
                seed_frame["source_predicted_positive_rate"] <= cap + 1e-12
            ].copy()
        if seed_frame.empty:
            raise ValueError(f"No candidate for seed={seed} satisfies source cap={cap}")
        seed_frame = seed_frame.sort_values(
            ["source_utility", "source_predicted_positive_rate", "candidate"],
            ascending=[False, True, True],
        )
        row = seed_frame.iloc[0].to_dict()
        row["selection_policy"] = policy_label(cap)
        row["selection_source_positive_rate_cap"] = cap
        selected.append(row)
    return pd.DataFrame(selected)


def summarize_candidates(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = SOURCE_METRICS + TARGET_METRICS + ["threshold"]
    summary = _aggregate(metrics, ["candidate"], columns)
    return summary.sort_values(
        ["target_accuracy_mean", "target_predicted_positive_rate_mean", "target_roc_auc_mean"],
        ascending=[False, True, False],
    )


def summarize_selection(selected: pd.DataFrame) -> pd.DataFrame:
    columns = SOURCE_METRICS + TARGET_METRICS + ["threshold"]
    summary = _aggregate(selected, ["selection_policy"], columns)
    choices = (
        selected.sort_values(["selection_policy", "seed"])
        .groupby("selection_policy")["candidate"]
        .apply(lambda values: "; ".join(values))
        .rename("selected_candidates")
        .reset_index()
    )
    summary = summary.merge(choices, on="selection_policy", how="left")
    return summary.sort_values(
        ["target_accuracy_mean", "target_predicted_positive_rate_mean", "target_roc_auc_mean"],
        ascending=[False, True, False],
    )


def concat_selection_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        raise ValueError("Expected at least one selection frame")
    cleaned = [frame.dropna(axis=1, how="all") for frame in frames]
    return pd.concat(cleaned, ignore_index=True)


def _aggregate(frame: pd.DataFrame, group_columns: list[str], value_columns: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(group_columns, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {column: key for column, key in zip(group_columns, keys, strict=True)}
        row["n_seeds"] = int(group["seed"].nunique())
        for column in value_columns:
            if column not in group.columns:
                continue
            values = pd.to_numeric(group[column], errors="coerce")
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_std"] = float(values.std(ddof=1)) if len(values.dropna()) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


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


def _report_sentence(selection_summary: pd.DataFrame, policy: str, metric: str) -> float | None:
    match = selection_summary[selection_summary["selection_policy"] == policy]
    if match.empty:
        return None
    return float(match.iloc[0][metric])


def write_report(
    candidate_summary: pd.DataFrame,
    selected_rows: pd.DataFrame,
    selection_summary: pd.DataFrame,
    out_path: Path,
) -> None:
    selection_columns = [
        "selection_policy",
        "n_seeds",
        "target_accuracy_mean",
        "target_roc_auc_mean",
        "target_brier_score_mean",
        "target_expected_calibration_error_mean",
        "target_predicted_positive_rate_mean",
        "source_utility_mean",
        "source_predicted_positive_rate_mean",
        "selected_candidates",
    ]
    candidate_columns = [
        "candidate",
        "n_seeds",
        "target_accuracy_mean",
        "target_roc_auc_mean",
        "target_brier_score_mean",
        "target_expected_calibration_error_mean",
        "target_predicted_positive_rate_mean",
        "source_utility_mean",
        "source_predicted_positive_rate_mean",
    ]
    selected_columns = [
        "selection_policy",
        "seed",
        "candidate",
        "source_utility",
        "source_predicted_positive_rate",
        "target_accuracy",
        "target_roc_auc",
        "target_predicted_positive_rate",
        "threshold",
    ]
    unconstrained_acc = _report_sentence(
        selection_summary, "source_utility_unconstrained", "target_accuracy_mean"
    )
    unconstrained_fake_rate = _report_sentence(
        selection_summary,
        "source_utility_unconstrained",
        "target_predicted_positive_rate_mean",
    )
    cap48_acc = _report_sentence(
        selection_summary, "source_utility_cap_0p48", "target_accuracy_mean"
    )
    cap48_fake_rate = _report_sentence(
        selection_summary, "source_utility_cap_0p48", "target_predicted_positive_rate_mean"
    )
    best_candidate = candidate_summary.iloc[0]
    lines = [
        "# MS COCOAI to Ishu Source-Utility Model Selection",
        "",
        (
            "This report tests the next SCP-Fusion question after threshold-only tuning: "
            "can source-side forensic utility choose among existing reverse fusion model families "
            "without looking at Ishu target labels?"
        ),
        "",
        "Selection uses the train/source predictions already saved inside each run directory. Missing threshold columns are treated as the original fixed 0.5 operating point.",
        "",
        "## Selection Policies",
        "",
        _markdown_table(selection_summary, selection_columns),
        "",
        "## Selected Runs",
        "",
        _markdown_table(selected_rows.sort_values(["selection_policy", "seed"]), selected_columns),
        "",
        "## Candidate Frontier",
        "",
        _markdown_table(candidate_summary.head(12), candidate_columns),
        "",
        "## Read",
        "",
    ]
    if unconstrained_acc is not None and unconstrained_fake_rate is not None:
        lines.append(
            (
                f"Unconstrained source-utility model selection reaches {unconstrained_acc:.4f} "
                f"mean target accuracy with a {unconstrained_fake_rate:.4f} target fake-call rate. "
                "It mostly selects the older high-recall fusion heads, which look excellent on source "
                "but over-call generated images after the MS-COCOAI-to-Ishu shift."
            )
        )
    if cap48_acc is not None and cap48_fake_rate is not None:
        lines.append(
            (
                f"Adding a source fake-call cap of 0.48 recovers a much cleaner operating point "
                f"at {cap48_acc:.4f} accuracy with a {cap48_fake_rate:.4f} target fake-call rate, "
                "but it still does not beat the fixed capped threshold family."
            )
        )
    lines.extend(
        [
            (
                f"The best target-labeled candidate in this suite is `{best_candidate['candidate']}` "
                f"at {best_candidate['target_accuracy_mean']:.4f} accuracy and "
                f"{best_candidate['target_roc_auc_mean']:.4f} AUC; that row is diagnostic only, "
                "because target labels are not allowed in model selection."
            ),
            (
                "The useful negative result is that source-train utility alone is not a reliable "
                "model selector under generator/domain shift. The source fake-rate constraint remains "
                "the active ingredient, so the next version should use source-heldout generator splits "
                "or train-time utility regularization rather than selecting on the full source train set."
            ),
            "",
            "## Rebuild",
            "",
            "```powershell",
            ".\\.venv\\Scripts\\python.exe scripts\\select_reverse_fusion_by_source_utility.py",
            "```",
            "",
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    for value in [
        args.fake_detection_weight,
        args.real_clearance_weight,
        args.real_fpr_penalty,
        args.fake_miss_penalty,
    ]:
        if value < 0.0:
            raise ValueError("Source utility weights and penalties must be non-negative")
    candidates = args.candidate or DEFAULT_CANDIDATES
    caps = _parse_cap_list(args.source_positive_rate_caps)
    metrics = collect_candidate_metrics(
        Path(args.run_root),
        args.family_dir,
        args.seeds,
        candidates,
        args.variant,
        fake_detection_weight=args.fake_detection_weight,
        real_clearance_weight=args.real_clearance_weight,
        real_fpr_penalty=args.real_fpr_penalty,
        fake_miss_penalty=args.fake_miss_penalty,
        allow_missing=args.allow_missing,
    )
    selected_frames = [select_candidates(metrics, cap) for cap in caps]
    selected = concat_selection_frames(selected_frames)
    candidate_summary = summarize_candidates(metrics)
    selection_summary = summarize_selection(selected)

    summary_dir = Path(args.summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = summary_dir / "ms_cocoai_to_ishu_model_utility_selection_candidates.csv"
    selected_path = summary_dir / "ms_cocoai_to_ishu_model_utility_selection_selected.csv"
    summary_path = summary_dir / "ms_cocoai_to_ishu_model_utility_selection_summary.csv"
    metrics.to_csv(candidate_path, index=False)
    selected.to_csv(selected_path, index=False)
    selection_summary.to_csv(summary_path, index=False)
    report_path = Path(args.report_path)
    write_report(candidate_summary, selected, selection_summary, report_path)
    print(candidate_path.resolve())
    print(selected_path.resolve())
    print(summary_path.resolve())
    print(report_path.resolve())


if __name__ == "__main__":
    main()
