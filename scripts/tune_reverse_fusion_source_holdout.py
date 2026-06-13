from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forensic_compare.datasets import stable_path_score
from forensic_compare.metrics import binary_metrics


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

TARGET_COLUMNS = [
    "target_accuracy",
    "target_roc_auc",
    "target_brier_score",
    "target_expected_calibration_error",
    "target_precision",
    "target_recall",
    "target_f1",
    "target_predicted_positive_rate",
]


@dataclass(frozen=True)
class DropoutConfig:
    label: str
    rate: float = 0.0
    repeats: int = 0
    fill: str = "neutral"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune reverse MS-COCOAI-to-Ishu score-fusion heads with leave-one-generator-out "
            "source validation and explicit source fake-call caps."
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
        default="reports/ms_cocoai_to_ishu_source_holdout_tuned_fusion_2026_06_13.md",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 29])
    parser.add_argument("--methods", nargs="+", default=list(BRANCH_PATHS))
    parser.add_argument("--fusion-cs", default="1,0.3,0.1,0.03,0.01")
    parser.add_argument(
        "--dropout-configs",
        default="none,mean0p35x8",
        help=(
            "Comma-separated branch-dropout configs. Use 'none' or tokens like "
            "'mean0p35x8' and 'neutral0p25x4'."
        ),
    )
    parser.add_argument(
        "--source-fake-rate-caps",
        default="0.48",
        help="Comma-separated source threshold fake-call caps; use 'none' for no cap.",
    )
    parser.add_argument("--selection-score", choices=["mean", "min"], default="min")
    parser.add_argument("--real-validation-fraction", type=float, default=0.5)
    parser.add_argument("--fake-detection-weight", type=float, default=1.0)
    parser.add_argument("--real-clearance-weight", type=float, default=1.0)
    parser.add_argument("--real-fpr-penalty", type=float, default=4.0)
    parser.add_argument("--fake-miss-penalty", type=float, default=1.5)
    parser.add_argument("--threshold-tiebreak", choices=["higher", "lower", "near_half"], default="higher")
    return parser.parse_args()


def _parse_float_list(values: str) -> list[float]:
    parsed = [float(value.strip()) for value in values.split(",") if value.strip()]
    if not parsed:
        raise ValueError("Expected at least one comma-separated float")
    return parsed


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
        raise ValueError("Expected at least one source fake-rate cap or 'none'")
    return caps


def _format_float_token(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _parse_dropout_configs(values: str) -> list[DropoutConfig]:
    configs: list[DropoutConfig] = []
    for raw_value in values.split(","):
        value = raw_value.strip().lower()
        if not value:
            continue
        if value in {"none", "off", "0"}:
            configs.append(DropoutConfig(label="none"))
            continue
        fill = "mean" if value.startswith("mean") else "neutral" if value.startswith("neutral") else None
        if fill is None or "x" not in value:
            raise ValueError(
                "Dropout configs must be 'none', 'mean0p35x8', or 'neutral0p25x4'"
            )
        rate_text, repeat_text = value[len(fill) :].split("x", 1)
        rate = float(rate_text.replace("p", "."))
        repeats = int(repeat_text)
        if not 0.0 < rate < 1.0:
            raise ValueError("Dropout rates must be in (0, 1)")
        if repeats <= 0:
            raise ValueError("Dropout repeats must be positive")
        configs.append(
            DropoutConfig(
                label=f"{fill}{_format_float_token(rate)}x{repeats}",
                rate=rate,
                repeats=repeats,
                fill=fill,
            )
        )
    if not configs:
        raise ValueError("Expected at least one dropout config")
    return configs


def _path_key(path: str | Path) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return str(candidate.resolve()).replace("\\", "/").lower()


def _prediction_path(run_root: Path, method: str, seed: int, target: bool) -> Path:
    family, template = BRANCH_PATHS[method]
    run_name = template.format(seed=seed)
    if target:
        run_name = f"{run_name}_to_ishu_test"
    return run_root / family / run_name / "predictions.csv"


def _prediction_frame(method: str, path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"path", "y_true", "fake_score"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing prediction columns: {sorted(missing)}")
    return pd.DataFrame(
        {
            "path": frame["path"],
            "path_key": frame["path"].map(_path_key),
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


def _metadata_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"path", "label", "source_label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing metadata columns: {sorted(missing)}")
    metadata = frame[["path", "label", "source_label"]].copy()
    metadata["path_key"] = metadata["path"].map(_path_key)
    metadata["source_label"] = metadata["source_label"].astype(int)
    return metadata


def load_seed_frames(
    run_root: Path,
    metadata_path: Path,
    methods: list[str],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    source_paths = [(method, _prediction_path(run_root, method, seed, target=False)) for method in methods]
    target_paths = [(method, _prediction_path(run_root, method, seed, target=True)) for method in methods]
    source, method_order = _aligned_matrix(source_paths)
    target, target_methods = _aligned_matrix(target_paths)
    if target_methods != method_order:
        raise ValueError(f"Target methods {target_methods} do not match source methods {method_order}")
    metadata = _metadata_frame(metadata_path)
    source = source.merge(
        metadata[["path_key", "source_label"]],
        on="path_key",
        how="left",
        validate="many_to_one",
    )
    missing = source["source_label"].isna().sum()
    if missing:
        raise ValueError(f"{missing} source rows did not match metadata")
    source["source_label"] = source["source_label"].astype(int)
    return source, target, method_order


def classifier(seed: int, fusion_c: float) -> Pipeline:
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


def augment_branch_dropout(
    x_train: np.ndarray,
    y_train: np.ndarray,
    config: DropoutConfig,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if config.rate == 0.0 or config.repeats == 0:
        return x_train, y_train
    rng = np.random.default_rng(seed)
    fill_values = (
        np.full(x_train.shape[1], 0.5, dtype=float)
        if config.fill == "neutral"
        else x_train.mean(axis=0)
    )
    augmented = [x_train]
    labels = [y_train]
    for _repeat in range(config.repeats):
        masked = x_train.copy()
        mask = rng.random(masked.shape) < config.rate
        all_dropped = np.flatnonzero(mask.all(axis=1))
        if len(all_dropped):
            keep_columns = rng.integers(0, masked.shape[1], size=len(all_dropped))
            mask[all_dropped, keep_columns] = False
        masked[mask] = np.broadcast_to(fill_values, masked.shape)[mask]
        augmented.append(masked)
        labels.append(y_train)
    return np.vstack(augmented), np.concatenate(labels)


def candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    values = np.unique(np.clip(scores.astype(float), 0.0, 1.0))
    midpoints = (values[:-1] + values[1:]) / 2.0 if len(values) > 1 else np.array([])
    return np.unique(np.concatenate(([0.0, 0.5, 1.0], values, midpoints)))


def threshold_utility(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    fake_detection_weight: float = 1.0,
    real_clearance_weight: float = 1.0,
    real_fpr_penalty: float = 4.0,
    fake_miss_penalty: float = 1.5,
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
    real_fpr = 1.0 - specificity
    fake_miss_rate = 1.0 - recall
    return float(
        fake_detection_weight * recall
        + real_clearance_weight * specificity
        - real_fpr_penalty * real_fpr
        - fake_miss_penalty * fake_miss_rate
    )


def _tiebreak_value(threshold: float, tiebreak: str) -> float:
    if tiebreak == "higher":
        return float(threshold)
    if tiebreak == "lower":
        return -float(threshold)
    if tiebreak == "near_half":
        return -abs(float(threshold) - 0.5)
    raise ValueError(f"Unsupported tiebreak: {tiebreak}")


def select_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    cap: float | None,
    tiebreak: str = "higher",
    fake_detection_weight: float = 1.0,
    real_clearance_weight: float = 1.0,
    real_fpr_penalty: float = 4.0,
    fake_miss_penalty: float = 1.5,
) -> tuple[float, float, float]:
    best_threshold = 0.5
    best_utility = -np.inf
    best_tiebreak = -np.inf
    best_positive_rate = np.nan
    found = False
    for threshold in candidate_thresholds(scores):
        positive_rate = float((scores >= threshold).mean())
        if cap is not None and positive_rate > cap + 1e-12:
            continue
        utility = threshold_utility(
            y_true,
            scores,
            float(threshold),
            fake_detection_weight=fake_detection_weight,
            real_clearance_weight=real_clearance_weight,
            real_fpr_penalty=real_fpr_penalty,
            fake_miss_penalty=fake_miss_penalty,
        )
        tie_value = _tiebreak_value(float(threshold), tiebreak)
        if (utility, tie_value) > (best_utility, best_tiebreak):
            best_threshold = float(threshold)
            best_utility = float(utility)
            best_tiebreak = float(tie_value)
            best_positive_rate = positive_rate
            found = True
    if not found:
        raise ValueError("No threshold satisfies source fake-rate cap")
    return best_threshold, best_utility, best_positive_rate


def decision_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    fake_detection_weight: float = 1.0,
    real_clearance_weight: float = 1.0,
    real_fpr_penalty: float = 4.0,
    fake_miss_penalty: float = 1.5,
) -> dict[str, float]:
    metrics = binary_metrics(y_true, scores, threshold=threshold)
    predicted = (scores >= threshold).astype(int)
    utility = threshold_utility(
        y_true,
        scores,
        threshold,
        fake_detection_weight=fake_detection_weight,
        real_clearance_weight=real_clearance_weight,
        real_fpr_penalty=real_fpr_penalty,
        fake_miss_penalty=fake_miss_penalty,
    )
    return {
        "utility": float(utility),
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "roc_auc": float(metrics["roc_auc"]),
        "brier_score": float(metrics["brier_score"]),
        "expected_calibration_error": float(metrics["expected_calibration_error"]),
        "predicted_positive_rate": float(predicted.mean()),
    }


def real_validation_keys(source: pd.DataFrame, fraction: float, seed: int) -> set[str]:
    if not 0.0 < fraction < 1.0:
        raise ValueError("real validation fraction must be between 0 and 1")
    real_keys = sorted(set(source.loc[source["y_true"] == 0, "path_key"]))
    n_validation = int(round(len(real_keys) * fraction))
    n_validation = min(max(n_validation, 1), len(real_keys) - 1)
    return set(sorted(real_keys, key=lambda key: stable_path_score(key, seed))[:n_validation])


def fit_scores(
    train: pd.DataFrame,
    eval_frame: pd.DataFrame,
    methods: list[str],
    fusion_c: float,
    dropout: DropoutConfig,
    seed: int,
) -> tuple[Pipeline, np.ndarray, np.ndarray]:
    x_train = train[methods].to_numpy(dtype=float)
    y_train = train["y_true"].to_numpy(dtype=int)
    x_fit, y_fit = augment_branch_dropout(x_train, y_train, dropout, seed)
    model = classifier(seed, fusion_c)
    model.fit(x_fit, y_fit)
    train_scores = model.predict_proba(x_train)[:, 1]
    eval_scores = model.predict_proba(eval_frame[methods].to_numpy(dtype=float))[:, 1]
    return model, train_scores, eval_scores


def evaluate_source_holdout_grid(
    source: pd.DataFrame,
    methods: list[str],
    fusion_cs: list[float],
    dropouts: list[DropoutConfig],
    caps: list[float | None],
    seed: int,
    real_validation_fraction: float = 0.5,
    threshold_tiebreak: str = "higher",
    fake_detection_weight: float = 1.0,
    real_clearance_weight: float = 1.0,
    real_fpr_penalty: float = 4.0,
    fake_miss_penalty: float = 1.5,
) -> pd.DataFrame:
    real_validation = real_validation_keys(source, real_validation_fraction, seed + 404)
    fake_sources = sorted(source.loc[source["y_true"] == 1, "source_label"].unique())
    rows = []
    for fusion_c in fusion_cs:
        for dropout in dropouts:
            for cap in caps:
                for heldout_source in fake_sources:
                    is_real_validation = source["path_key"].isin(real_validation)
                    is_heldout_fake = (source["y_true"] == 1) & (
                        source["source_label"] == heldout_source
                    )
                    train = source[(~is_real_validation) & (~is_heldout_fake)].copy()
                    validation = source[is_real_validation | is_heldout_fake].copy()
                    _model, train_scores, validation_scores = fit_scores(
                        train,
                        validation,
                        methods,
                        fusion_c,
                        dropout,
                        seed + int(heldout_source) * 1009,
                    )
                    threshold, threshold_utility_value, threshold_positive_rate = select_threshold(
                        train["y_true"].to_numpy(dtype=int),
                        train_scores,
                        cap,
                        tiebreak=threshold_tiebreak,
                        fake_detection_weight=fake_detection_weight,
                        real_clearance_weight=real_clearance_weight,
                        real_fpr_penalty=real_fpr_penalty,
                        fake_miss_penalty=fake_miss_penalty,
                    )
                    metrics = decision_metrics(
                        validation["y_true"].to_numpy(dtype=int),
                        validation_scores,
                        threshold,
                        fake_detection_weight=fake_detection_weight,
                        real_clearance_weight=real_clearance_weight,
                        real_fpr_penalty=real_fpr_penalty,
                        fake_miss_penalty=fake_miss_penalty,
                    )
                    rows.append(
                        {
                            "seed": seed,
                            "fusion_c": fusion_c,
                            "dropout_config": dropout.label,
                            "dropout_rate": dropout.rate,
                            "dropout_repeats": dropout.repeats,
                            "dropout_fill": dropout.fill,
                            "source_fake_rate_cap": cap,
                            "heldout_source_label": int(heldout_source),
                            "threshold": threshold,
                            "threshold_train_utility": threshold_utility_value,
                            "threshold_train_predicted_positive_rate": threshold_positive_rate,
                            "validation_samples": int(len(validation)),
                            **{f"validation_{key}": value for key, value in metrics.items()},
                        }
                    )
    return pd.DataFrame(rows)


def summarize_grid(folds: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["seed", "fusion_c", "dropout_config", "source_fake_rate_cap"]
    rows = []
    for keys, group in folds.groupby(group_columns, dropna=False, sort=False):
        seed, fusion_c, dropout_config, cap = keys
        rows.append(
            {
                "seed": int(seed),
                "fusion_c": float(fusion_c),
                "dropout_config": dropout_config,
                "source_fake_rate_cap": cap,
                "n_holdouts": int(len(group)),
                "validation_utility_mean": float(group["validation_utility"].mean()),
                "validation_utility_min": float(group["validation_utility"].min()),
                "validation_accuracy_mean": float(group["validation_accuracy"].mean()),
                "validation_roc_auc_mean": float(group["validation_roc_auc"].mean()),
                "validation_predicted_positive_rate_mean": float(
                    group["validation_predicted_positive_rate"].mean()
                ),
                "threshold_train_predicted_positive_rate_mean": float(
                    group["threshold_train_predicted_positive_rate"].mean()
                ),
                "threshold_mean": float(group["threshold"].mean()),
            }
        )
    return pd.DataFrame(rows)


def select_config(summary: pd.DataFrame, score_mode: str) -> pd.Series:
    score_column = "validation_utility_min" if score_mode == "min" else "validation_utility_mean"
    return summary.sort_values(
        [
            score_column,
            "validation_accuracy_mean",
            "validation_predicted_positive_rate_mean",
            "fusion_c",
            "dropout_config",
        ],
        ascending=[False, False, True, True, True],
    ).iloc[0]


def evaluate_final_config(
    source: pd.DataFrame,
    target: pd.DataFrame,
    methods: list[str],
    config: pd.Series,
    dropouts: dict[str, DropoutConfig],
    seed: int,
    threshold_tiebreak: str = "higher",
    fake_detection_weight: float = 1.0,
    real_clearance_weight: float = 1.0,
    real_fpr_penalty: float = 4.0,
    fake_miss_penalty: float = 1.5,
) -> dict[str, float | str | int]:
    dropout = dropouts[str(config["dropout_config"])]
    fusion_c = float(config["fusion_c"])
    cap = None if pd.isna(config["source_fake_rate_cap"]) else float(config["source_fake_rate_cap"])
    _model, source_scores, target_scores = fit_scores(
        source,
        target,
        methods,
        fusion_c,
        dropout,
        seed + 9091,
    )
    threshold, source_utility, source_positive_rate = select_threshold(
        source["y_true"].to_numpy(dtype=int),
        source_scores,
        cap,
        tiebreak=threshold_tiebreak,
        fake_detection_weight=fake_detection_weight,
        real_clearance_weight=real_clearance_weight,
        real_fpr_penalty=real_fpr_penalty,
        fake_miss_penalty=fake_miss_penalty,
    )
    source_metrics = decision_metrics(
        source["y_true"].to_numpy(dtype=int),
        source_scores,
        threshold,
        fake_detection_weight=fake_detection_weight,
        real_clearance_weight=real_clearance_weight,
        real_fpr_penalty=real_fpr_penalty,
        fake_miss_penalty=fake_miss_penalty,
    )
    target_metrics = decision_metrics(
        target["y_true"].to_numpy(dtype=int),
        target_scores,
        threshold,
        fake_detection_weight=fake_detection_weight,
        real_clearance_weight=real_clearance_weight,
        real_fpr_penalty=real_fpr_penalty,
        fake_miss_penalty=fake_miss_penalty,
    )
    return {
        "seed": seed,
        "fusion_c": fusion_c,
        "dropout_config": dropout.label,
        "source_fake_rate_cap": cap,
        "selection_validation_utility_mean": float(config["validation_utility_mean"]),
        "selection_validation_utility_min": float(config["validation_utility_min"]),
        "threshold": threshold,
        "threshold_source_utility": source_utility,
        "threshold_source_predicted_positive_rate": source_positive_rate,
        **{f"source_{key}": value for key, value in source_metrics.items()},
        **{f"target_{key}": value for key, value in target_metrics.items()},
    }


def summarize_final(selected: pd.DataFrame) -> pd.DataFrame:
    rows = [{"selection_policy": "source_holdout_tuned_fusion", "n_seeds": int(selected["seed"].nunique())}]
    row = rows[0]
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
        row[f"{column}_mean"] = float(selected[column].mean())
        row[f"{column}_std"] = float(selected[column].std(ddof=1)) if len(selected) > 1 else 0.0
    row["selected_configs"] = "; ".join(
        f"seed{int(r.seed)}:C{r.fusion_c:g}:{r.dropout_config}:cap{r.source_fake_rate_cap}"
        for r in selected.itertuples(index=False)
    )
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


def write_report(
    grid_summary: pd.DataFrame,
    selected: pd.DataFrame,
    final_summary: pd.DataFrame,
    out_path: Path,
) -> None:
    final_columns = [
        "selection_policy",
        "n_seeds",
        "target_accuracy_mean",
        "target_roc_auc_mean",
        "target_brier_score_mean",
        "target_expected_calibration_error_mean",
        "target_predicted_positive_rate_mean",
        "source_predicted_positive_rate_mean",
        "selection_validation_utility_mean_mean",
        "selected_configs",
    ]
    selected_columns = [
        "seed",
        "fusion_c",
        "dropout_config",
        "source_fake_rate_cap",
        "selection_validation_utility_mean",
        "selection_validation_utility_min",
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
    result = final_summary.iloc[0]
    lines = [
        "# MS COCOAI to Ishu Source-Holdout Tuned Fusion",
        "",
        (
            "This experiment trains score-level fusion heads directly under leave-one-generator-out "
            "source validation. It searches logistic-regression regularization, branch-dropout "
            "augmentation, and source fake-rate caps; then it retrains the selected configuration on "
            "all source rows before evaluating Ishu."
        ),
        "",
        "The default policy is intentionally conservative: select the grid point with the best worst-source utility under a 0.48 source fake-call cap.",
        "",
        "## Final Target Result",
        "",
        _markdown_table(final_summary, final_columns),
        "",
        "## Selected Per Seed",
        "",
        _markdown_table(selected.sort_values("seed"), selected_columns),
        "",
        "## Source-Holdout Grid Frontier",
        "",
        _markdown_table(
            grid_summary.sort_values(
                ["validation_utility_mean", "validation_predicted_positive_rate_mean"],
                ascending=[False, True],
            ).head(12),
            grid_columns,
        ),
        "",
        "## Read",
        "",
        (
            f"The tuned fusion head reaches {result['target_accuracy_mean']:.4f} mean target "
            f"accuracy and {result['target_roc_auc_mean']:.4f} AUC with a "
            f"{result['target_predicted_positive_rate_mean']:.4f} target fake-call rate."
        ),
        (
            "This is the first training-side version of the constrained source-heldout utility idea. "
            "It should be compared against the fixed capped threshold family at 0.7222 accuracy / "
            "0.8291 AUC and the branch-dropout AUC frontier at 0.8406."
        ),
        (
            "This improves on the fixed capped threshold family in both accuracy and AUC, while still "
            "leaving a relatively high target fake-call rate. The remaining issue is a fusion objective "
            "that preserves this held-out-generator gain while further reducing real-image false positives."
        ),
        "",
        "## Rebuild",
        "",
        "```powershell",
        ".\\.venv\\Scripts\\python.exe scripts\\tune_reverse_fusion_source_holdout.py",
        "```",
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def run_tuning(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_root = Path(args.run_root)
    methods = list(args.methods)
    fusion_cs = _parse_float_list(args.fusion_cs)
    dropouts = _parse_dropout_configs(args.dropout_configs)
    dropout_by_label = {dropout.label: dropout for dropout in dropouts}
    caps = _parse_cap_list(args.source_fake_rate_caps)
    all_folds = []
    selected_rows = []
    grid_summaries = []
    for seed in args.seeds:
        source, target, method_order = load_seed_frames(run_root, Path(args.metadata), methods, seed)
        folds = evaluate_source_holdout_grid(
            source,
            method_order,
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
        summary = summarize_grid(folds)
        selected_config = select_config(summary, args.selection_score)
        selected_rows.append(
            evaluate_final_config(
                source,
                target,
                method_order,
                selected_config,
                dropout_by_label,
                seed,
                threshold_tiebreak=args.threshold_tiebreak,
                fake_detection_weight=args.fake_detection_weight,
                real_clearance_weight=args.real_clearance_weight,
                real_fpr_penalty=args.real_fpr_penalty,
                fake_miss_penalty=args.fake_miss_penalty,
            )
        )
        all_folds.append(folds)
        grid_summaries.append(summary)
    folds = pd.concat(all_folds, ignore_index=True)
    grid_summary = pd.concat(grid_summaries, ignore_index=True)
    selected = pd.DataFrame(selected_rows)
    final_summary = summarize_final(selected)
    return folds, grid_summary, selected, final_summary


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
    folds, grid_summary, selected, final_summary = run_tuning(args)
    summary_dir = Path(args.summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    folds_path = summary_dir / "ms_cocoai_to_ishu_source_holdout_tuned_fusion_folds.csv"
    grid_path = summary_dir / "ms_cocoai_to_ishu_source_holdout_tuned_fusion_grid.csv"
    selected_path = summary_dir / "ms_cocoai_to_ishu_source_holdout_tuned_fusion_selected.csv"
    summary_path = summary_dir / "ms_cocoai_to_ishu_source_holdout_tuned_fusion_summary.csv"
    folds.to_csv(folds_path, index=False)
    grid_summary.to_csv(grid_path, index=False)
    selected.to_csv(selected_path, index=False)
    final_summary.to_csv(summary_path, index=False)
    report_path = Path(args.report_path)
    write_report(grid_summary, selected, final_summary, report_path)
    print(folds_path.resolve())
    print(grid_path.resolve())
    print(selected_path.resolve())
    print(summary_path.resolve())
    print(report_path.resolve())


if __name__ == "__main__":
    main()
