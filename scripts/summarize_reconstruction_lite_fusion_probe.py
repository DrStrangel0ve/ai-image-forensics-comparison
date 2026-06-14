from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forensic_compare.metrics import binary_metrics  # noqa: E402
from summarize_reconstruction_lite_probe import (  # noqa: E402
    METRIC_COLUMNS,
    _confusion_rates,
    _format_metric,
    _markdown_table,
    _seed_from_path,
)


RUN_DATE = "2026-06-14"
CANDIDATE_COLUMNS = [
    "combined_v3_logreg",
    "reconstruction_lite_logreg",
    "mean_fusion",
    "source_logreg_fusion",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize a bounded reconstruction_lite + combined_v3 transfer "
            "score-fusion diagnostic."
        )
    )
    parser.add_argument("--source-reconstruction-root", default="runs/reconstruction_lite_probe")
    parser.add_argument("--source-combined-root", default="runs/combined_v3_probe_for_recon_lite")
    parser.add_argument(
        "--target-reconstruction-root",
        default="runs/reconstruction_lite_probe_to_ms",
    )
    parser.add_argument(
        "--target-combined-root",
        default="runs/combined_v3_probe_for_recon_lite_to_ms",
    )
    parser.add_argument(
        "--seed-out",
        default="reports/assets/reconstruction_lite_fusion_probe_seed_summary.csv",
    )
    parser.add_argument(
        "--mean-out",
        default="reports/assets/reconstruction_lite_fusion_probe_mean_summary.csv",
    )
    parser.add_argument(
        "--delta-out",
        default="reports/assets/reconstruction_lite_fusion_probe_delta_summary.csv",
    )
    parser.add_argument(
        "--coefficient-out",
        default="reports/assets/reconstruction_lite_fusion_probe_coefficients.csv",
    )
    parser.add_argument(
        "--report-out",
        default="reports/reconstruction_lite_fusion_probe_2026_06_14.md",
    )
    return parser.parse_args()


def _read_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"path", "y_true", "fake_score"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    frame = frame[["path", "y_true", "fake_score"]].copy()
    frame["path"] = frame["path"].astype(str)
    frame["y_true"] = frame["y_true"].astype(int)
    frame["fake_score"] = pd.to_numeric(frame["fake_score"], errors="raise").clip(0.0, 1.0)
    return frame


def _paired_predictions(combined_path: Path, reconstruction_path: Path) -> pd.DataFrame:
    combined = _read_predictions(combined_path)
    reconstruction = _read_predictions(reconstruction_path)
    paired = combined.merge(
        reconstruction,
        on=["path", "y_true"],
        suffixes=("_combined", "_reconstruction"),
        validate="one_to_one",
    )
    if paired.empty:
        raise ValueError(f"No paired predictions for {combined_path} and {reconstruction_path}")
    if len(paired) != min(len(combined), len(reconstruction)):
        raise ValueError(
            "Prediction files do not align one-to-one: "
            f"{combined_path} has {len(combined)}, {reconstruction_path} has "
            f"{len(reconstruction)}, paired {len(paired)}"
        )
    return paired


def _scores_to_row(
    *,
    seed: int,
    candidate: str,
    y_true: np.ndarray,
    scores: np.ndarray,
    score_source: str,
) -> dict[str, Any]:
    metrics = binary_metrics(y_true, scores)
    return {
        "seed": seed,
        "candidate": candidate,
        "score_source": score_source,
        "n_target": metrics["n_samples"],
        "accuracy": metrics["accuracy"],
        "roc_auc": metrics["roc_auc"],
        "brier_score": metrics["brier_score"],
        "expected_calibration_error": metrics["expected_calibration_error"],
        "maximum_calibration_error": metrics["maximum_calibration_error"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        **_confusion_rates(metrics),
    }


def _fit_source_logreg(source: pd.DataFrame, seed: int) -> tuple[LogisticRegression, dict[str, Any]]:
    x_source = source[["fake_score_combined", "fake_score_reconstruction"]].to_numpy(dtype=float)
    y_source = source["y_true"].to_numpy(dtype=int)
    model = LogisticRegression(class_weight="balanced", random_state=seed, max_iter=1000)
    model.fit(x_source, y_source)
    return model, {
        "seed": seed,
        "n_source": int(len(source)),
        "combined_v3_coef": float(model.coef_[0][0]),
        "reconstruction_lite_coef": float(model.coef_[0][1]),
        "intercept": float(model.intercept_[0]),
    }


def _seed_prediction_path(root: Path, seed: int) -> Path:
    path = root / f"seed{seed}" / "predictions.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_seed_summary(
    source_reconstruction_root: Path,
    source_combined_root: Path,
    target_reconstruction_root: Path,
    target_combined_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    target_paths = sorted(target_reconstruction_root.glob("seed*/predictions.csv"))
    if not target_paths:
        raise FileNotFoundError(f"No target predictions found under {target_reconstruction_root}")

    for target_reconstruction_path in target_paths:
        seed = _seed_from_path(target_reconstruction_path)
        target = _paired_predictions(
            _seed_prediction_path(target_combined_root, seed),
            target_reconstruction_path,
        )
        source = _paired_predictions(
            _seed_prediction_path(source_combined_root, seed),
            _seed_prediction_path(source_reconstruction_root, seed),
        )
        source_model, coefficient_row = _fit_source_logreg(source, seed)
        coefficient_rows.append(coefficient_row)

        y_target = target["y_true"].to_numpy(dtype=int)
        combined_scores = target["fake_score_combined"].to_numpy(dtype=float)
        reconstruction_scores = target["fake_score_reconstruction"].to_numpy(dtype=float)
        mean_scores = (combined_scores + reconstruction_scores) / 2.0
        source_logreg_scores = source_model.predict_proba(
            target[["fake_score_combined", "fake_score_reconstruction"]].to_numpy(dtype=float)
        )[:, 1]
        for candidate, scores, score_source in [
            ("combined_v3_logreg", combined_scores, "target combined_v3 predictions"),
            (
                "reconstruction_lite_logreg",
                reconstruction_scores,
                "target reconstruction_lite predictions",
            ),
            ("mean_fusion", mean_scores, "unweighted mean of target scores"),
            (
                "source_logreg_fusion",
                source_logreg_scores,
                "logistic fusion fitted on paired source predictions",
            ),
        ]:
            rows.append(
                _scores_to_row(
                    seed=seed,
                    candidate=candidate,
                    y_true=y_target,
                    scores=scores,
                    score_source=score_source,
                )
            )

    seed_summary = pd.DataFrame(rows)
    seed_summary["candidate"] = pd.Categorical(
        seed_summary["candidate"], categories=CANDIDATE_COLUMNS, ordered=True
    )
    seed_summary = seed_summary.sort_values(["seed", "candidate"]).reset_index(drop=True)
    seed_summary["candidate"] = seed_summary["candidate"].astype(str)
    coefficients = pd.DataFrame(coefficient_rows).sort_values("seed").reset_index(drop=True)
    return seed_summary, coefficients


def build_mean_summary(seed_summary: pd.DataFrame) -> pd.DataFrame:
    metric_cols = METRIC_COLUMNS + [
        "maximum_calibration_error",
        "fake_call_rate",
        "real_false_positive_rate",
        "fake_miss_rate",
    ]
    rows = []
    for candidate, group in seed_summary.groupby("candidate", sort=False):
        row: dict[str, Any] = {
            "candidate": candidate,
            "score_source": group["score_source"].iloc[0],
            "n_seeds": int(group["seed"].nunique()),
            "seeds": ",".join(str(seed) for seed in sorted(group["seed"].unique())),
            "n_target_mean": float(pd.to_numeric(group["n_target"], errors="coerce").mean()),
        }
        for column in metric_cols:
            values = pd.to_numeric(group[column], errors="coerce")
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_std"] = float(values.std(ddof=1)) if len(values.dropna()) > 1 else 0.0
            row[f"{column}_min"] = float(values.min())
            row[f"{column}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


def build_delta_summary(mean_summary: pd.DataFrame) -> pd.DataFrame:
    baseline = mean_summary[mean_summary["candidate"] == "combined_v3_logreg"]
    if baseline.empty:
        return pd.DataFrame()
    base = baseline.iloc[0]
    metric_cols = METRIC_COLUMNS + [
        "maximum_calibration_error",
        "fake_call_rate",
        "real_false_positive_rate",
        "fake_miss_rate",
    ]
    rows = []
    for candidate in [
        "reconstruction_lite_logreg",
        "mean_fusion",
        "source_logreg_fusion",
    ]:
        candidate_rows = mean_summary[mean_summary["candidate"] == candidate]
        if candidate_rows.empty:
            continue
        current = candidate_rows.iloc[0]
        for metric in metric_cols:
            rows.append(
                {
                    "candidate": candidate,
                    "baseline": "combined_v3_logreg",
                    "metric": metric,
                    "candidate_mean": current[f"{metric}_mean"],
                    "baseline_mean": base[f"{metric}_mean"],
                    "delta_mean": current[f"{metric}_mean"] - base[f"{metric}_mean"],
                }
            )
    return pd.DataFrame(rows)


def _display_table(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    table = frame[columns].copy()
    for column in columns:
        if column in {"candidate", "metric", "n_seeds", "n_source", "seed", "score_source"}:
            continue
        table[column] = table[column].map(_format_metric)
    return table


def _metric_value(frame: pd.DataFrame, candidate: str, metric: str) -> float:
    row = frame[frame["candidate"] == candidate].iloc[0]
    return float(row[f"{metric}_mean"])


def write_report(
    seed_summary: pd.DataFrame,
    mean_summary: pd.DataFrame,
    delta_summary: pd.DataFrame,
    coefficients: pd.DataFrame,
    report_out: Path,
) -> None:
    mean_columns = [
        "candidate",
        "n_seeds",
        "accuracy_mean",
        "roc_auc_mean",
        "brier_score_mean",
        "expected_calibration_error_mean",
        "fake_call_rate_mean",
    ]
    seed_columns = [
        "seed",
        "candidate",
        "accuracy",
        "roc_auc",
        "brier_score",
        "expected_calibration_error",
        "fake_call_rate",
    ]
    coefficient_columns = [
        "seed",
        "n_source",
        "combined_v3_coef",
        "reconstruction_lite_coef",
        "intercept",
    ]
    mean_table = _display_table(mean_summary, mean_columns)
    seed_table = _display_table(seed_summary, seed_columns)
    coefficient_table = _display_table(coefficients, coefficient_columns)

    best_auc_candidate = mean_summary.sort_values("roc_auc_mean", ascending=False).iloc[0]
    best_calibration_candidate = mean_summary.sort_values(
        "expected_calibration_error_mean", ascending=True
    ).iloc[0]
    mean_fusion_auc_delta = _metric_value(mean_summary, "mean_fusion", "roc_auc") - _metric_value(
        mean_summary, "combined_v3_logreg", "roc_auc"
    )
    source_fusion_auc_delta = _metric_value(
        mean_summary, "source_logreg_fusion", "roc_auc"
    ) - _metric_value(mean_summary, "combined_v3_logreg", "roc_auc")
    reconstruction_auc = _metric_value(mean_summary, "reconstruction_lite_logreg", "roc_auc")
    mean_fusion_auc = _metric_value(mean_summary, "mean_fusion", "roc_auc")

    lines = [
        "# reconstruction_lite Fusion Probe",
        "",
        f"Run date: {RUN_DATE}",
        "",
        "Generated by `scripts/summarize_reconstruction_lite_fusion_probe.py` from ignored local prediction runs.",
        "",
        "This bounded diagnostic asks whether the standalone resize-reconstruction branch adds value when fused with bounded `combined_v3` logistic-regression scores on the Ishu -> source-balanced MS COCOAI transfer slice. It is not an official benchmark result and not a replacement for the full SCP-Fusion grid.",
        "",
        "## Mean Transfer Metrics",
        "",
        _markdown_table(mean_table, mean_columns),
        "",
        "## Readout",
        "",
        f"- Best mean AUC in this probe is `{_format_metric(best_auc_candidate['roc_auc_mean'])}` from `{best_auc_candidate['candidate']}`.",
        f"- Best mean ECE is `{_format_metric(best_calibration_candidate['expected_calibration_error_mean'])}` from `{best_calibration_candidate['candidate']}`.",
        f"- Naive mean fusion improves over bounded `combined_v3` by `{_format_metric(mean_fusion_auc_delta)}` mean AUC, but it does not beat `reconstruction_lite` alone (`{_format_metric(mean_fusion_auc)}` vs `{_format_metric(reconstruction_auc)}`).",
        f"- Source-logistic fusion improves over bounded `combined_v3` by `{_format_metric(source_fusion_auc_delta)}` mean AUC, but the source coefficients still lean too hard on `combined_v3` for this transfer direction.",
        "- The useful conclusion is narrower than a win claim: resize-reconstruction residuals are transfer-informative, but the fusion rule needs source-aware validation or stronger branch regularization before it belongs in the main SCP-Fusion headline.",
        "",
        "## Source Fusion Coefficients",
        "",
        _markdown_table(coefficient_table, coefficient_columns),
        "",
        "## Seed Transfer Metrics",
        "",
        _markdown_table(seed_table, seed_columns),
        "",
        "## Delta Against bounded combined_v3",
        "",
        _markdown_table(
            _display_table(
                delta_summary,
                ["candidate", "metric", "candidate_mean", "baseline_mean", "delta_mean"],
            ),
            ["candidate", "metric", "candidate_mean", "baseline_mean", "delta_mean"],
        ),
        "",
    ]
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    seed_summary, coefficients = load_seed_summary(
        source_reconstruction_root=Path(args.source_reconstruction_root),
        source_combined_root=Path(args.source_combined_root),
        target_reconstruction_root=Path(args.target_reconstruction_root),
        target_combined_root=Path(args.target_combined_root),
    )
    mean_summary = build_mean_summary(seed_summary)
    delta_summary = build_delta_summary(mean_summary)

    for path, frame in [
        (Path(args.seed_out), seed_summary),
        (Path(args.mean_out), mean_summary),
        (Path(args.delta_out), delta_summary),
        (Path(args.coefficient_out), coefficients),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)

    write_report(
        seed_summary=seed_summary,
        mean_summary=mean_summary,
        delta_summary=delta_summary,
        coefficients=coefficients,
        report_out=Path(args.report_out),
    )
    print(Path(args.report_out).resolve())
    print(Path(args.mean_out).resolve())


if __name__ == "__main__":
    main()
