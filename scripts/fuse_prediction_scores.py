from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from forensic_compare.calibration import CALIBRATORS, fit_calibrator, predict_calibrated
from forensic_compare.datasets import stable_path_score
from forensic_compare.metrics import binary_metrics
from forensic_compare.utils import ensure_dir, write_json


THRESHOLD_STRATEGIES = {
    "fixed",
    "source_accuracy",
    "source_balanced_accuracy",
    "source_f1",
    "source_utility",
    "source_youden",
}
THRESHOLD_TIEBREAKERS = {
    "near_half",
    "higher",
    "lower",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train score-level fusion from aligned prediction CSVs."
    )
    parser.add_argument(
        "--train",
        action="append",
        required=True,
        help="Training prediction in METHOD=PATH form. Repeat for each base model.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        help="Evaluation prediction in VARIANT:METHOD=PATH form. Repeat for each method/variant.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Fixed decision threshold, or fallback value when --threshold-strategy=fixed.",
    )
    parser.add_argument(
        "--threshold-strategy",
        choices=sorted(THRESHOLD_STRATEGIES),
        default="fixed",
        help=(
            "Decision threshold policy. Source strategies tune the operating point on "
            "held-out source calibration rows when available, otherwise on source train rows."
        ),
    )
    parser.add_argument(
        "--threshold-tiebreak",
        choices=sorted(THRESHOLD_TIEBREAKERS),
        default="near_half",
        help=(
            "Tie-breaker when multiple source thresholds have equal utility. "
            "'higher' is the conservative fake-call option."
        ),
    )
    parser.add_argument(
        "--threshold-max-positive-rate",
        type=float,
        default=None,
        help=(
            "Optional source-side cap on the selected threshold's predicted fake rate. "
            "Applies only to source threshold strategies."
        ),
    )
    parser.add_argument(
        "--threshold-fake-detection-weight",
        type=float,
        default=0.5,
        help="Reward for source fake recall when --threshold-strategy=source_utility.",
    )
    parser.add_argument(
        "--threshold-real-clearance-weight",
        type=float,
        default=0.5,
        help="Reward for source real specificity when --threshold-strategy=source_utility.",
    )
    parser.add_argument(
        "--threshold-real-fpr-penalty",
        type=float,
        default=1.0,
        help="Penalty for source real images called fake when --threshold-strategy=source_utility.",
    )
    parser.add_argument(
        "--threshold-fake-miss-penalty",
        type=float,
        default=1.0,
        help="Penalty for source fake images called real when --threshold-strategy=source_utility.",
    )
    parser.add_argument(
        "--fusion-c",
        type=float,
        default=1.0,
        help="Inverse regularization strength for the logistic-regression fusion head.",
    )
    parser.add_argument(
        "--branch-dropout-rate",
        type=float,
        default=0.0,
        help="Probability of replacing a branch score with a neutral value in augmented training rows.",
    )
    parser.add_argument(
        "--branch-dropout-repeats",
        type=int,
        default=0,
        help="Number of branch-dropout augmented copies to add per original training row.",
    )
    parser.add_argument(
        "--branch-dropout-fill",
        choices=["neutral", "mean"],
        default="neutral",
        help="Value used for dropped branch scores: 0.5 neutral probability or the branch train mean.",
    )
    parser.add_argument(
        "--score-calibrator",
        choices=["none", *sorted(CALIBRATORS)],
        default="none",
        help="Optional post-hoc calibrator fitted on source-domain fused scores.",
    )
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=0.0,
        help=(
            "Deterministic class-balanced fraction of source rows reserved for score calibration "
            "and/or source-threshold selection."
        ),
    )
    return parser.parse_args()


def _parse_train(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Training predictions must be METHOD=PATH, got {value!r}")
    method, path = value.split("=", 1)
    if not method or not path:
        raise ValueError(f"Training predictions must be METHOD=PATH, got {value!r}")
    return method, Path(path)


def _parse_variant(value: str) -> tuple[str, str, Path]:
    if "=" not in value:
        raise ValueError(f"Variant predictions must be VARIANT:METHOD=PATH, got {value!r}")
    left, path = value.split("=", 1)
    if ":" not in left:
        raise ValueError(f"Variant predictions must be VARIANT:METHOD=PATH, got {value!r}")
    variant, method = left.split(":", 1)
    if not variant or not method or not path:
        raise ValueError(f"Variant predictions must be VARIANT:METHOD=PATH, got {value!r}")
    return variant, method, Path(path)


def _norm(path: str | Path) -> str:
    return str(Path(path).resolve()).replace("\\", "/").lower()


def _prediction_frame(method: str, path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"path", "y_true", "fake_score"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing prediction columns: {sorted(missing)}")
    return pd.DataFrame(
        {
            "path": frame["path"],
            "path_key": frame["path"].map(_norm),
            "y_true": frame["y_true"].astype(int),
            method: frame["fake_score"].astype(float),
        }
    )


def _aligned_matrix(named_paths: list[tuple[str, Path]]) -> tuple[pd.DataFrame, list[str]]:
    methods = [method for method, _path in named_paths]
    if len(set(methods)) != len(methods):
        raise ValueError(f"Duplicate method names are not allowed: {methods}")
    merged: pd.DataFrame | None = None
    for method, path in named_paths:
        frame = _prediction_frame(method, path)
        if merged is None:
            merged = frame
            continue
        merged = merged.merge(
            frame[["path_key", "y_true", method]],
            on="path_key",
            suffixes=("", "_next"),
        )
        mismatches = merged[merged["y_true"] != merged["y_true_next"]]
        if not mismatches.empty:
            raise ValueError(f"Label mismatch while joining {method}: {len(mismatches)} rows")
        merged = merged.drop(columns=["y_true_next"])
    if merged is None or merged.empty:
        raise ValueError("No prediction rows were aligned")
    return merged, methods


def _variant_groups(values: list[str]) -> dict[str, list[tuple[str, Path]]]:
    grouped: dict[str, list[tuple[str, Path]]] = {}
    for value in values:
        variant, method, path = _parse_variant(value)
        grouped.setdefault(variant, []).append((method, path))
    return grouped


def _classifier(seed: int, fusion_c: float):
    if fusion_c <= 0.0:
        raise ValueError("--fusion-c must be positive")
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    C=fusion_c,
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )


def _augment_branch_dropout(
    x_train: np.ndarray,
    y_train: np.ndarray,
    rate: float,
    repeats: int,
    seed: int,
    fill: str,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 <= rate < 1.0:
        raise ValueError("--branch-dropout-rate must be in [0, 1)")
    if repeats < 0:
        raise ValueError("--branch-dropout-repeats must be non-negative")
    if rate == 0.0 or repeats == 0:
        return x_train, y_train
    rng = np.random.default_rng(seed)
    if fill == "neutral":
        fill_values = np.full(x_train.shape[1], 0.5, dtype=float)
    elif fill == "mean":
        fill_values = x_train.mean(axis=0)
    else:
        raise ValueError(f"Unsupported branch dropout fill: {fill}")
    augmented = [x_train]
    labels = [y_train]
    for _repeat in range(repeats):
        masked = x_train.copy()
        mask = rng.random(masked.shape) < rate
        all_dropped = np.flatnonzero(mask.all(axis=1))
        if len(all_dropped):
            keep_columns = rng.integers(0, masked.shape[1], size=len(all_dropped))
            mask[all_dropped, keep_columns] = False
        masked[mask] = np.broadcast_to(fill_values, masked.shape)[mask]
        augmented.append(masked)
        labels.append(y_train)
    return np.vstack(augmented), np.concatenate(labels)


def _candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    values = np.unique(np.clip(scores.astype(float), 0.0, 1.0))
    if len(values) == 0:
        raise ValueError("Cannot select a threshold without scores")
    midpoints = (values[:-1] + values[1:]) / 2.0 if len(values) > 1 else np.array([])
    return np.unique(np.concatenate(([0.0, 0.5, 1.0], values, midpoints)))


def _threshold_utility(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    strategy: str,
    fake_detection_weight: float,
    real_clearance_weight: float,
    real_fpr_penalty: float,
    fake_miss_penalty: float,
) -> float:
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

    if strategy == "source_accuracy":
        return float((predicted == y_true).mean())
    if strategy == "source_balanced_accuracy":
        return float((recall + specificity) / 2.0)
    if strategy == "source_f1":
        return float(2.0 * precision * recall / max(precision + recall, 1e-12))
    if strategy == "source_utility":
        real_fpr = 1.0 - specificity
        fake_miss_rate = 1.0 - recall
        return float(
            fake_detection_weight * recall
            + real_clearance_weight * specificity
            - real_fpr_penalty * real_fpr
            - fake_miss_penalty * fake_miss_rate
        )
    if strategy == "source_youden":
        return float(recall + specificity - 1.0)
    raise ValueError(f"Unsupported threshold strategy: {strategy}")


def _threshold_tiebreak_value(threshold: float, tiebreak: str) -> float:
    if tiebreak == "near_half":
        return -abs(float(threshold) - 0.5)
    if tiebreak == "higher":
        return float(threshold)
    if tiebreak == "lower":
        return -float(threshold)
    raise ValueError(f"Unsupported threshold tie-breaker: {tiebreak}")


def _select_source_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    strategy: str,
    tiebreak: str,
    max_positive_rate: float | None,
    fake_detection_weight: float = 0.5,
    real_clearance_weight: float = 0.5,
    real_fpr_penalty: float = 1.0,
    fake_miss_penalty: float = 1.0,
) -> tuple[float, float, float]:
    if strategy == "fixed":
        raise ValueError("Use --threshold directly for the fixed threshold strategy")
    if len(np.unique(y_true)) != 2:
        raise ValueError("Source threshold selection requires both classes")
    if not np.isfinite(scores).all():
        raise ValueError("Source threshold selection requires finite scores")
    if max_positive_rate is not None and not 0.0 <= max_positive_rate <= 1.0:
        raise ValueError("--threshold-max-positive-rate must be in [0, 1]")
    candidates = _candidate_thresholds(scores)
    best_threshold = float(candidates[0])
    best_utility = -np.inf
    best_tiebreak = -np.inf
    best_positive_rate = np.inf
    found_candidate = False
    for threshold in candidates:
        positive_rate = float((scores >= float(threshold)).mean())
        if max_positive_rate is not None and positive_rate > max_positive_rate + 1e-12:
            continue
        utility = _threshold_utility(
            y_true,
            scores,
            float(threshold),
            strategy,
            fake_detection_weight,
            real_clearance_weight,
            real_fpr_penalty,
            fake_miss_penalty,
        )
        tiebreak_value = _threshold_tiebreak_value(float(threshold), tiebreak)
        if (utility, tiebreak_value) > (best_utility, best_tiebreak):
            best_threshold = float(threshold)
            best_utility = float(utility)
            best_tiebreak = float(tiebreak_value)
            best_positive_rate = positive_rate
            found_candidate = True
    if not found_candidate:
        raise ValueError("No source threshold satisfies --threshold-max-positive-rate")
    return best_threshold, best_utility, best_positive_rate


def _split_calibration_frame(
    frame: pd.DataFrame,
    fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 <= fraction < 1.0:
        raise ValueError("--calibration-fraction must be in [0, 1)")
    if fraction == 0.0:
        return frame, frame
    if frame["y_true"].nunique() != 2:
        raise ValueError("Calibration split requires both classes")
    fit_indices = []
    calibration_indices = []
    for _label, group in frame.groupby("y_true", sort=True):
        if len(group) < 2:
            raise ValueError("Each class needs at least two rows for held-out calibration")
        ordered = group.assign(
            split_score=group["path_key"].map(lambda path: stable_path_score(path, seed))
        ).sort_values("split_score")
        n_calibration = int(round(len(ordered) * fraction))
        n_calibration = min(max(n_calibration, 1), len(ordered) - 1)
        calibration_indices.extend(ordered.index[:n_calibration])
        fit_indices.extend(ordered.index[n_calibration:])
    fit_frame = frame.loc[fit_indices].sort_index().reset_index(drop=True)
    calibration_frame = frame.loc[calibration_indices].sort_index().reset_index(drop=True)
    if fit_frame["y_true"].nunique() != 2 or calibration_frame["y_true"].nunique() != 2:
        raise ValueError("Model-fit and calibration splits must both contain both classes")
    return fit_frame, calibration_frame


def _coefficient_rows(methods: list[str], classifier) -> list[dict]:
    scaler = classifier.named_steps["scale"]
    model = classifier.named_steps["logreg"]
    standardized = model.coef_[0].astype(float)
    scale = np.where(scaler.scale_.astype(float) < 1e-12, 1.0, scaler.scale_.astype(float))
    mean = scaler.mean_.astype(float)
    raw = standardized / scale
    raw_intercept = float(model.intercept_[0] - np.sum(standardized * mean / scale))
    rows = [
        {
            "method": method,
            "standardized_coefficient": float(std_coef),
            "raw_score_coefficient": float(raw_coef),
        }
        for method, std_coef, raw_coef in zip(methods, standardized, raw, strict=True)
    ]
    rows.append(
        {
            "method": "__intercept__",
            "standardized_coefficient": float(model.intercept_[0]),
            "raw_score_coefficient": raw_intercept,
        }
    )
    return rows


def _metrics_row(
    variant: str,
    methods: list[str],
    frame: pd.DataFrame,
    classifier,
    threshold: float,
    threshold_strategy: str,
    threshold_tiebreak: str,
    threshold_source: str,
    threshold_max_positive_rate: float | None,
    threshold_source_predicted_positive_rate: float | None,
    calibrator=None,
) -> tuple[dict, list[dict]]:
    x = frame[methods].to_numpy(dtype=float)
    y_true = frame["y_true"].to_numpy(dtype=int)
    raw_scores = classifier.predict_proba(x)[:, 1]
    scores = predict_calibrated(calibrator, raw_scores) if calibrator is not None else raw_scores
    metrics = binary_metrics(y_true, scores, threshold=threshold)
    metrics.update(
        {
            "variant": variant,
            "method": "score_fusion",
            "base_methods": methods,
            "score_calibrator": calibrator.name if calibrator is not None else "none",
            "n_samples": int(len(y_true)),
            "threshold": float(threshold),
            "threshold_strategy": threshold_strategy,
            "threshold_tiebreak": threshold_tiebreak,
            "threshold_source": threshold_source,
            "threshold_max_positive_rate": threshold_max_positive_rate,
            "threshold_source_predicted_positive_rate": threshold_source_predicted_positive_rate,
            "positive_rate": float(y_true.mean()) if len(y_true) else 0.0,
            "predicted_positive_rate": float((scores >= threshold).mean()) if len(scores) else 0.0,
            "score_mean": float(scores.mean()) if len(scores) else 0.0,
            "score_std": float(scores.std(ddof=0)) if len(scores) else 0.0,
            "raw_score_mean": float(raw_scores.mean()) if len(raw_scores) else 0.0,
        }
    )
    rows = [
        {
            "path": path,
            "y_true": int(truth),
            "fake_score": float(score),
            "raw_fake_score": float(raw_score),
        }
        for path, truth, score, raw_score in zip(frame["path"], y_true, scores, raw_scores)
    ]
    return metrics, rows


def main() -> None:
    args = parse_args()
    utility_values = [
        args.threshold_fake_detection_weight,
        args.threshold_real_clearance_weight,
        args.threshold_real_fpr_penalty,
        args.threshold_fake_miss_penalty,
    ]
    if any(value < 0.0 for value in utility_values):
        raise ValueError("Source utility threshold weights and penalties must be non-negative")
    out_dir = ensure_dir(args.out_dir)
    if (
        args.score_calibrator == "none"
        and args.threshold_strategy == "fixed"
        and args.calibration_fraction != 0.0
    ):
        raise ValueError(
            "--calibration-fraction requires --score-calibrator or a source threshold strategy"
        )
    if args.threshold_max_positive_rate is not None and args.threshold_strategy == "fixed":
        raise ValueError("--threshold-max-positive-rate requires a source threshold strategy")
    train_frame, methods = _aligned_matrix(list(map(_parse_train, args.train)))
    fit_frame, calibration_frame = _split_calibration_frame(
        train_frame,
        args.calibration_fraction,
        args.seed + 101,
    )
    x_train = fit_frame[methods].to_numpy(dtype=float)
    y_train = fit_frame["y_true"].to_numpy(dtype=int)
    x_fit, y_fit = _augment_branch_dropout(
        x_train,
        y_train,
        args.branch_dropout_rate,
        args.branch_dropout_repeats,
        args.seed,
        args.branch_dropout_fill,
    )
    classifier = _classifier(args.seed, args.fusion_c)
    classifier.fit(x_fit, y_fit)
    calibrator = None
    if args.score_calibrator != "none":
        calibration_scores = classifier.predict_proba(
            calibration_frame[methods].to_numpy(dtype=float)
        )[:, 1]
        calibration_y = calibration_frame["y_true"].to_numpy(dtype=int)
        calibrator = fit_calibrator(args.score_calibrator, calibration_y, calibration_scores)

    threshold = float(args.threshold)
    threshold_source = "fixed"
    threshold_selection_utility = None
    threshold_source_predicted_positive_rate = None
    if args.threshold_strategy != "fixed":
        threshold_frame = calibration_frame if args.calibration_fraction > 0.0 else train_frame
        threshold_source = (
            "source_calibration" if args.calibration_fraction > 0.0 else "source_train"
        )
        source_raw_scores = classifier.predict_proba(
            threshold_frame[methods].to_numpy(dtype=float)
        )[:, 1]
        source_scores = (
            predict_calibrated(calibrator, source_raw_scores)
            if calibrator is not None
            else source_raw_scores
        )
        (
            threshold,
            threshold_selection_utility,
            threshold_source_predicted_positive_rate,
        ) = _select_source_threshold(
            threshold_frame["y_true"].to_numpy(dtype=int),
            source_scores,
            args.threshold_strategy,
            args.threshold_tiebreak,
            args.threshold_max_positive_rate,
            args.threshold_fake_detection_weight,
            args.threshold_real_clearance_weight,
            args.threshold_real_fpr_penalty,
            args.threshold_fake_miss_penalty,
        )

    all_metrics = []
    train_metrics, train_rows = _metrics_row(
        "train",
        methods,
        train_frame,
        classifier,
        threshold,
        args.threshold_strategy,
        args.threshold_tiebreak,
        threshold_source,
        args.threshold_max_positive_rate,
        threshold_source_predicted_positive_rate,
        calibrator=calibrator,
    )
    all_metrics.append(train_metrics)
    train_dir = ensure_dir(out_dir / "train")
    with (train_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "y_true", "fake_score", "raw_fake_score"])
        writer.writeheader()
        writer.writerows(train_rows)

    variant_groups = _variant_groups(args.variant)
    for variant, named_paths in sorted(variant_groups.items()):
        variant_frame, variant_methods = _aligned_matrix(named_paths)
        if variant_methods != methods:
            raise ValueError(
                f"Variant {variant!r} has methods {variant_methods}, expected {methods}"
            )
        metrics, rows = _metrics_row(
            variant,
            methods,
            variant_frame,
            classifier,
            threshold,
            args.threshold_strategy,
            args.threshold_tiebreak,
            threshold_source,
            args.threshold_max_positive_rate,
            threshold_source_predicted_positive_rate,
            calibrator=calibrator,
        )
        all_metrics.append(metrics)
        variant_dir = ensure_dir(out_dir / variant)
        with (variant_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["path", "y_true", "fake_score", "raw_fake_score"])
            writer.writeheader()
            writer.writerows(rows)

    write_json(
        {
            "method": "score_fusion",
            "base_methods": methods,
            "seed": int(args.seed),
            "threshold": float(threshold),
            "requested_threshold": float(args.threshold),
            "threshold_strategy": args.threshold_strategy,
            "threshold_tiebreak": args.threshold_tiebreak,
            "threshold_source": threshold_source,
            "threshold_selection_utility": threshold_selection_utility,
            "threshold_fake_detection_weight": float(args.threshold_fake_detection_weight),
            "threshold_real_clearance_weight": float(args.threshold_real_clearance_weight),
            "threshold_real_fpr_penalty": float(args.threshold_real_fpr_penalty),
            "threshold_fake_miss_penalty": float(args.threshold_fake_miss_penalty),
            "threshold_max_positive_rate": args.threshold_max_positive_rate,
            "threshold_source_predicted_positive_rate": threshold_source_predicted_positive_rate,
            "n_train": int(len(train_frame)),
            "n_fusion_train": int(len(y_train)),
            "n_calibration": int(len(calibration_frame)),
            "n_fit": int(len(y_fit)),
            "fusion_c": float(args.fusion_c),
            "branch_dropout_rate": float(args.branch_dropout_rate),
            "branch_dropout_repeats": int(args.branch_dropout_repeats),
            "branch_dropout_fill": args.branch_dropout_fill,
            "score_calibrator": args.score_calibrator,
            "calibration_fraction": float(args.calibration_fraction),
            "calibrator_temperature": None
            if calibrator is None
            else calibrator.temperature,
            "coefficients": _coefficient_rows(methods, classifier),
            "metrics": all_metrics,
        },
        out_dir / "metrics.json",
    )
    joblib.dump(classifier, out_dir / "score_fusion_model.joblib")
    pd.DataFrame(_coefficient_rows(methods, classifier)).to_csv(
        out_dir / "score_fusion_coefficients.csv",
        index=False,
    )
    if calibrator is not None:
        joblib.dump(calibrator, out_dir / "score_calibrator.joblib")
    summary_rows = [
        {
            "variant": metrics["variant"],
            "score_calibrator": metrics["score_calibrator"],
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
            "brier_score": metrics["brier_score"],
            "expected_calibration_error": metrics["expected_calibration_error"],
            "maximum_calibration_error": metrics["maximum_calibration_error"],
            "threshold": metrics["threshold"],
            "threshold_strategy": metrics["threshold_strategy"],
            "threshold_tiebreak": metrics["threshold_tiebreak"],
            "threshold_source": metrics["threshold_source"],
            "threshold_fake_detection_weight": args.threshold_fake_detection_weight,
            "threshold_real_clearance_weight": args.threshold_real_clearance_weight,
            "threshold_real_fpr_penalty": args.threshold_real_fpr_penalty,
            "threshold_fake_miss_penalty": args.threshold_fake_miss_penalty,
            "threshold_max_positive_rate": metrics["threshold_max_positive_rate"],
            "threshold_source_predicted_positive_rate": metrics[
                "threshold_source_predicted_positive_rate"
            ],
            "predicted_positive_rate": metrics["predicted_positive_rate"],
            "score_mean": metrics["score_mean"],
            "raw_score_mean": metrics["raw_score_mean"],
            "n_samples": metrics["n_samples"],
        }
        for metrics in all_metrics
    ]
    pd.DataFrame(summary_rows).to_csv(out_dir / "summary.csv", index=False)
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"Wrote score-fusion results to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
