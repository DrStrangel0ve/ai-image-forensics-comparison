from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from select_reverse_fusion_by_source_utility import (
    DEFAULT_CANDIDATES,
    TARGET_METRICS,
    _format_float_token,
    _float_or_na,
    _markdown_table,
    _summary_row,
    _threshold_from_row,
    concat_selection_frames,
)


ROOT = Path(__file__).resolve().parents[1]

SOURCE_HOLDOUT_METRICS = [
    "source_holdout_utility_mean",
    "source_holdout_utility_min",
    "source_holdout_accuracy_mean",
    "source_holdout_recall_mean",
    "source_holdout_specificity_mean",
    "source_holdout_predicted_positive_rate_mean",
    "source_utility",
    "source_predicted_positive_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select reverse MS-COCOAI-to-Ishu fusion models using leave-one-generator-out "
            "source utility from MS COCOAI metadata."
        )
    )
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--family-dir", default="ms_cocoai_to_ishu_neural_fusion")
    parser.add_argument(
        "--source-metadata",
        default="data/raw/ms_cocoai_2026_validation_source_balanced_100/metadata.csv",
    )
    parser.add_argument("--summary-dir", default="reports/assets")
    parser.add_argument(
        "--report-path",
        default="reports/ms_cocoai_to_ishu_source_holdout_model_selection_2026_06_13.md",
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
        help="Comma-separated full-source fake-call caps; use 'none' for no cap.",
    )
    parser.add_argument(
        "--selection-scores",
        default="mean,min",
        help="Comma-separated source-holdout selection scores: mean and/or min.",
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


def _parse_score_modes(values: str) -> list[str]:
    modes = [value.strip().lower() for value in values.split(",") if value.strip()]
    if not modes:
        raise ValueError("Expected at least one selection score")
    unsupported = sorted(set(modes) - {"mean", "min"})
    if unsupported:
        raise ValueError(f"Unsupported selection score modes: {unsupported}")
    return modes


def _path_key(path: str | Path) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return str(candidate.resolve()).replace("\\", "/").lower()


def source_decision_metrics_from_frame(
    frame: pd.DataFrame,
    threshold: float,
    fake_detection_weight: float = 1.0,
    real_clearance_weight: float = 1.0,
    real_fpr_penalty: float = 4.0,
    fake_miss_penalty: float = 1.5,
) -> dict[str, float]:
    required = {"y_true", "fake_score"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction frame is missing columns: {sorted(missing)}")
    y_true = frame["y_true"].astype(int).to_numpy()
    scores = frame["fake_score"].astype(float).to_numpy()
    if len(y_true) == 0:
        raise ValueError("Cannot score an empty source fold")
    if not np.isfinite(scores).all():
        raise ValueError("Source fold contains non-finite scores")
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
        "utility": float(utility),
        "accuracy": float((predicted == y_true).mean()),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "real_fpr": float(real_fpr),
        "fake_miss_rate": float(fake_miss_rate),
        "positive_rate": float(y_true.mean()),
        "predicted_positive_rate": float(predicted.mean()),
        "n_samples": int(len(frame)),
    }


def load_source_metadata(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"path", "label", "source_label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing metadata columns: {sorted(missing)}")
    metadata = frame[["path", "label", "source_label"]].copy()
    metadata["path_key"] = metadata["path"].map(_path_key)
    metadata["label"] = metadata["label"].astype(int)
    metadata["source_label"] = metadata["source_label"].astype(int)
    return metadata


def _joined_predictions(predictions_path: Path, metadata: pd.DataFrame) -> pd.DataFrame:
    predictions = pd.read_csv(predictions_path)
    predictions["path_key"] = predictions["path"].map(_path_key)
    joined = predictions.merge(
        metadata[["path_key", "source_label"]],
        on="path_key",
        how="left",
        validate="many_to_one",
    )
    missing = joined["source_label"].isna().sum()
    if missing:
        raise ValueError(f"{predictions_path} has {missing} rows without source metadata")
    joined["source_label"] = joined["source_label"].astype(int)
    return joined


def collect_source_holdout_metrics(
    run_root: Path,
    family_dir: str,
    source_metadata: Path,
    seeds: list[int],
    candidates: list[str],
    variant: str,
    fake_detection_weight: float = 1.0,
    real_clearance_weight: float = 1.0,
    real_fpr_penalty: float = 4.0,
    fake_miss_penalty: float = 1.5,
    allow_missing: bool = False,
) -> pd.DataFrame:
    metadata = load_source_metadata(source_metadata)
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
            joined = _joined_predictions(predictions_path, metadata)
            source_all = source_decision_metrics_from_frame(
                joined,
                threshold,
                fake_detection_weight=fake_detection_weight,
                real_clearance_weight=real_clearance_weight,
                real_fpr_penalty=real_fpr_penalty,
                fake_miss_penalty=fake_miss_penalty,
            )
            fake_sources = sorted(joined.loc[joined["y_true"].astype(int) == 1, "source_label"].unique())
            for heldout_source in fake_sources:
                fold = joined[
                    (joined["y_true"].astype(int) == 0)
                    | (
                        (joined["y_true"].astype(int) == 1)
                        & (joined["source_label"] == heldout_source)
                    )
                ].copy()
                fold_metrics = source_decision_metrics_from_frame(
                    fold,
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
                        "heldout_source_label": int(heldout_source),
                        "run_dir": str(run_dir).replace("\\", "/"),
                        "threshold": threshold,
                        "threshold_strategy": train_row.get("threshold_strategy", "fixed"),
                        "threshold_source": train_row.get("threshold_source", "fixed"),
                        "threshold_max_positive_rate": _float_or_na(
                            train_row, "threshold_max_positive_rate"
                        ),
                        "source_utility": source_all["utility"],
                        "source_accuracy": source_all["accuracy"],
                        "source_recall": source_all["recall"],
                        "source_specificity": source_all["specificity"],
                        "source_predicted_positive_rate": source_all[
                            "predicted_positive_rate"
                        ],
                        "source_holdout_utility": fold_metrics["utility"],
                        "source_holdout_accuracy": fold_metrics["accuracy"],
                        "source_holdout_precision": fold_metrics["precision"],
                        "source_holdout_recall": fold_metrics["recall"],
                        "source_holdout_specificity": fold_metrics["specificity"],
                        "source_holdout_real_fpr": fold_metrics["real_fpr"],
                        "source_holdout_fake_miss_rate": fold_metrics["fake_miss_rate"],
                        "source_holdout_predicted_positive_rate": fold_metrics[
                            "predicted_positive_rate"
                        ],
                        "source_holdout_n_samples": fold_metrics["n_samples"],
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
        raise ValueError("No source-holdout metrics were collected")
    return pd.DataFrame(rows)


def summarize_candidate_holdouts(folds: pd.DataFrame) -> pd.DataFrame:
    per_seed = _candidate_seed_summary(folds)
    summary = _aggregate(
        per_seed,
        ["candidate"],
        SOURCE_HOLDOUT_METRICS + TARGET_METRICS + ["threshold"],
    )
    return summary.sort_values(
        [
            "source_holdout_utility_mean_mean",
            "source_predicted_positive_rate_mean",
            "target_roc_auc_mean",
        ],
        ascending=[False, True, False],
    )


def _candidate_seed_summary(folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (candidate, seed), group in folds.groupby(["candidate", "seed"], sort=False):
        base = group.iloc[0].to_dict()
        rows.append(
            {
                "candidate": candidate,
                "seed": int(seed),
                "run_dir": base["run_dir"],
                "threshold": float(base["threshold"]),
                "source_utility": float(base["source_utility"]),
                "source_accuracy": float(base["source_accuracy"]),
                "source_recall": float(base["source_recall"]),
                "source_specificity": float(base["source_specificity"]),
                "source_predicted_positive_rate": float(
                    base["source_predicted_positive_rate"]
                ),
                "source_holdout_utility_mean": float(group["source_holdout_utility"].mean()),
                "source_holdout_utility_min": float(group["source_holdout_utility"].min()),
                "source_holdout_accuracy_mean": float(group["source_holdout_accuracy"].mean()),
                "source_holdout_recall_mean": float(group["source_holdout_recall"].mean()),
                "source_holdout_specificity_mean": float(
                    group["source_holdout_specificity"].mean()
                ),
                "source_holdout_predicted_positive_rate_mean": float(
                    group["source_holdout_predicted_positive_rate"].mean()
                ),
                "target_accuracy": float(base["target_accuracy"]),
                "target_precision": float(base["target_precision"]),
                "target_recall": float(base["target_recall"]),
                "target_f1": float(base["target_f1"]),
                "target_roc_auc": float(base["target_roc_auc"]),
                "target_brier_score": float(base["target_brier_score"]),
                "target_expected_calibration_error": float(
                    base["target_expected_calibration_error"]
                ),
                "target_predicted_positive_rate": float(
                    base["target_predicted_positive_rate"]
                ),
            }
        )
    return pd.DataFrame(rows)


def policy_label(score_mode: str, cap: float | None) -> str:
    cap_label = "unconstrained" if cap is None else f"cap_{_format_float_token(cap)}"
    return f"source_holdout_{score_mode}_utility_{cap_label}"


def select_candidates(
    folds: pd.DataFrame,
    score_mode: str,
    cap: float | None,
) -> pd.DataFrame:
    if score_mode not in {"mean", "min"}:
        raise ValueError(f"Unsupported score mode: {score_mode}")
    score_column = f"source_holdout_utility_{score_mode}"
    per_seed = _candidate_seed_summary(folds)
    selected = []
    for seed in sorted(per_seed["seed"].unique()):
        seed_frame = per_seed[per_seed["seed"] == seed].copy()
        if cap is not None:
            seed_frame = seed_frame[
                seed_frame["source_predicted_positive_rate"] <= cap + 1e-12
            ].copy()
        if seed_frame.empty:
            raise ValueError(
                f"No candidate for seed={seed} satisfies source positive-rate cap={cap}"
            )
        seed_frame = seed_frame.sort_values(
            [score_column, "source_predicted_positive_rate", "candidate"],
            ascending=[False, True, True],
        )
        row = seed_frame.iloc[0].to_dict()
        row["selection_policy"] = policy_label(score_mode, cap)
        row["selection_score_mode"] = score_mode
        row["selection_source_positive_rate_cap"] = cap
        row["selection_score"] = float(row[score_column])
        selected.append(row)
    return pd.DataFrame(selected)


def summarize_selection(selected: pd.DataFrame) -> pd.DataFrame:
    summary = _aggregate(
        selected,
        ["selection_policy"],
        SOURCE_HOLDOUT_METRICS + TARGET_METRICS + ["selection_score", "threshold"],
    )
    choices = (
        selected.sort_values(["selection_policy", "seed"])
        .groupby("selection_policy")["candidate"]
        .apply(lambda values: "; ".join(values))
        .rename("selected_candidates")
        .reset_index()
    )
    summary = summary.merge(choices, on="selection_policy", how="left")
    return summary.sort_values(
        [
            "target_accuracy_mean",
            "target_predicted_positive_rate_mean",
            "target_roc_auc_mean",
        ],
        ascending=[False, True, False],
    )


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


def _metric_from_policy(summary: pd.DataFrame, policy: str, metric: str) -> float | None:
    match = summary[summary["selection_policy"] == policy]
    if match.empty:
        return None
    return float(match.iloc[0][metric])


def write_report(
    candidate_summary: pd.DataFrame,
    selected: pd.DataFrame,
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
        "source_holdout_utility_mean_mean",
        "source_predicted_positive_rate_mean",
        "selected_candidates",
    ]
    selected_columns = [
        "selection_policy",
        "seed",
        "candidate",
        "selection_score",
        "source_holdout_utility_mean",
        "source_holdout_utility_min",
        "source_predicted_positive_rate",
        "target_accuracy",
        "target_roc_auc",
        "target_predicted_positive_rate",
    ]
    candidate_columns = [
        "candidate",
        "n_seeds",
        "source_holdout_utility_mean_mean",
        "source_holdout_utility_min_mean",
        "source_predicted_positive_rate_mean",
        "target_accuracy_mean",
        "target_roc_auc_mean",
        "target_predicted_positive_rate_mean",
    ]
    unconstrained_acc = _metric_from_policy(
        selection_summary,
        "source_holdout_mean_utility_unconstrained",
        "target_accuracy_mean",
    )
    unconstrained_fake_rate = _metric_from_policy(
        selection_summary,
        "source_holdout_mean_utility_unconstrained",
        "target_predicted_positive_rate_mean",
    )
    cap48_acc = _metric_from_policy(
        selection_summary,
        "source_holdout_mean_utility_cap_0p48",
        "target_accuracy_mean",
    )
    cap48_fake_rate = _metric_from_policy(
        selection_summary,
        "source_holdout_mean_utility_cap_0p48",
        "target_predicted_positive_rate_mean",
    )
    best_candidate = candidate_summary.sort_values(
        ["target_accuracy_mean", "target_predicted_positive_rate_mean"],
        ascending=[False, True],
    ).iloc[0]
    lines = [
        "# MS COCOAI to Ishu Source-Holdout Model Selection",
        "",
        (
            "This report repeats reverse SCP-Fusion model selection with a stricter "
            "source-side evaluator: each MS COCOAI generated source label is held out "
            "in turn, scored against all real images, and aggregated before choosing "
            "a fusion family."
        ),
        "",
        "The target Ishu labels are used only after source-side selection has chosen a run for each seed.",
        "",
        "## Selection Policies",
        "",
        _markdown_table(selection_summary, selection_columns),
        "",
        "## Selected Runs",
        "",
        _markdown_table(selected.sort_values(["selection_policy", "seed"]), selected_columns),
        "",
        "## Candidate Source-Holdout Frontier",
        "",
        _markdown_table(candidate_summary.head(12), candidate_columns),
        "",
        "## Read",
        "",
    ]
    if unconstrained_acc is not None and unconstrained_fake_rate is not None:
        lines.append(
            (
                f"Leave-one-generator-out source utility without a fake-rate cap still "
                f"selects over-firing fusion heads, reaching {unconstrained_acc:.4f} "
                f"target accuracy with a {unconstrained_fake_rate:.4f} target fake-call rate."
            )
        )
    if cap48_acc is not None and cap48_fake_rate is not None:
        lines.append(
            (
                f"Adding the 0.48 source fake-rate cap recovers {cap48_acc:.4f} "
                f"accuracy and lowers the target fake-call rate to {cap48_fake_rate:.4f}, "
                "but it still does not exceed the fixed capped source-threshold family."
            )
        )
    lines.extend(
        [
            (
                f"The best target-labeled candidate remains `{best_candidate['candidate']}` "
                f"at {best_candidate['target_accuracy_mean']:.4f} accuracy, but target "
                "labels are not allowed for model selection."
            ),
            (
                "The paper-facing conclusion is now sharper: source-heldout generator "
                "utility is necessary as a diagnostic, but not sufficient as a selector "
                "unless it also controls the source fake-call rate. SCP-Fusion v1 should "
                "therefore train or select against held-out-generator utility with an "
                "explicit real-image false-positive/fake-rate constraint."
            ),
            "",
            "## Rebuild",
            "",
            "```powershell",
            ".\\.venv\\Scripts\\python.exe scripts\\select_reverse_fusion_by_source_holdout.py",
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
    score_modes = _parse_score_modes(args.selection_scores)
    folds = collect_source_holdout_metrics(
        Path(args.run_root),
        args.family_dir,
        Path(args.source_metadata),
        args.seeds,
        candidates,
        args.variant,
        fake_detection_weight=args.fake_detection_weight,
        real_clearance_weight=args.real_clearance_weight,
        real_fpr_penalty=args.real_fpr_penalty,
        fake_miss_penalty=args.fake_miss_penalty,
        allow_missing=args.allow_missing,
    )
    selected_frames = [
        select_candidates(folds, score_mode, cap)
        for score_mode in score_modes
        for cap in caps
    ]
    selected = concat_selection_frames(selected_frames)
    candidate_summary = summarize_candidate_holdouts(folds)
    selection_summary = summarize_selection(selected)

    summary_dir = Path(args.summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    folds_path = summary_dir / "ms_cocoai_to_ishu_source_holdout_model_selection_folds.csv"
    selected_path = summary_dir / "ms_cocoai_to_ishu_source_holdout_model_selection_selected.csv"
    summary_path = summary_dir / "ms_cocoai_to_ishu_source_holdout_model_selection_summary.csv"
    candidate_path = summary_dir / "ms_cocoai_to_ishu_source_holdout_model_selection_candidates.csv"
    folds.to_csv(folds_path, index=False)
    selected.to_csv(selected_path, index=False)
    selection_summary.to_csv(summary_path, index=False)
    candidate_summary.to_csv(candidate_path, index=False)
    report_path = Path(args.report_path)
    write_report(candidate_summary, selected, selection_summary, report_path)
    print(folds_path.resolve())
    print(selected_path.resolve())
    print(summary_path.resolve())
    print(candidate_path.resolve())
    print(report_path.resolve())


if __name__ == "__main__":
    main()
