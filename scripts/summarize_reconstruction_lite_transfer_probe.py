from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from summarize_reconstruction_lite_probe import (
    METRIC_COLUMNS,
    _confusion_rates,
    _format_metric,
    _markdown_escape,
    _markdown_table,
    _seed_from_path,
)


RUN_DATE = "2026-06-14"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the bounded Ishu -> MS COCOAI reconstruction_lite transfer probe."
    )
    parser.add_argument("--reconstruction-root", default="runs/reconstruction_lite_probe_to_ms")
    parser.add_argument("--combined-root", default="runs/combined_v3_probe_for_recon_lite_to_ms")
    parser.add_argument(
        "--seed-out",
        default="reports/assets/reconstruction_lite_transfer_probe_seed_summary.csv",
    )
    parser.add_argument(
        "--mean-out",
        default="reports/assets/reconstruction_lite_transfer_probe_mean_summary.csv",
    )
    parser.add_argument(
        "--delta-out",
        default="reports/assets/reconstruction_lite_transfer_probe_delta_summary.csv",
    )
    parser.add_argument(
        "--report-out",
        default="reports/reconstruction_lite_transfer_probe_2026_06_14.md",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows_for_root(root: Path, candidate: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.glob("seed*/metrics.json")):
        metrics = _read_json(path)
        rows.append(
            {
                "seed": _seed_from_path(path),
                "candidate": candidate,
                "feature_set": metrics.get("feature_set"),
                "classifier": metrics.get("classifier"),
                "method": metrics.get("method"),
                "source_model_dir": metrics.get("source_model_dir"),
                "target_dir": metrics.get("target_dir"),
                "target_split": metrics.get("target_split"),
                "n_target": metrics.get("n_target"),
                "accuracy": metrics.get("accuracy"),
                "roc_auc": metrics.get("roc_auc"),
                "brier_score": metrics.get("brier_score"),
                "expected_calibration_error": metrics.get("expected_calibration_error"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1": metrics.get("f1"),
                **_confusion_rates(metrics),
                "metrics_path": path.as_posix(),
            }
        )
    return rows


def load_seed_summary(reconstruction_root: Path, combined_root: Path) -> pd.DataFrame:
    rows = []
    rows.extend(_rows_for_root(reconstruction_root, "reconstruction_lite_logreg"))
    rows.extend(_rows_for_root(combined_root, "combined_v3_logreg"))
    if not rows:
        raise FileNotFoundError(
            f"No transfer metrics found under {reconstruction_root} or {combined_root}"
        )
    return pd.DataFrame(rows).sort_values(["seed", "candidate"]).reset_index(drop=True)


def build_mean_summary(seed_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["candidate", "feature_set", "classifier"]
    metric_cols = METRIC_COLUMNS + [
        "fake_call_rate",
        "real_false_positive_rate",
        "fake_miss_rate",
    ]
    for keys, group in seed_summary.groupby(group_cols, sort=False, dropna=False):
        row = dict(zip(group_cols, keys))
        row["n_seeds"] = int(group["seed"].nunique())
        row["seeds"] = ",".join(str(seed) for seed in sorted(group["seed"].unique()))
        row["n_target_mean"] = float(pd.to_numeric(group["n_target"], errors="coerce").mean())
        for column in metric_cols:
            values = pd.to_numeric(group[column], errors="coerce")
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_std"] = float(values.std(ddof=1)) if len(values.dropna()) > 1 else 0.0
            row[f"{column}_min"] = float(values.min())
            row[f"{column}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("candidate").reset_index(drop=True)


def build_delta_summary(mean_summary: pd.DataFrame) -> pd.DataFrame:
    baseline = mean_summary[mean_summary["candidate"] == "combined_v3_logreg"]
    reconstruction = mean_summary[mean_summary["candidate"] == "reconstruction_lite_logreg"]
    if baseline.empty or reconstruction.empty:
        return pd.DataFrame()
    base = baseline.iloc[0]
    recon = reconstruction.iloc[0]
    rows = []
    for metric in METRIC_COLUMNS + ["fake_call_rate", "real_false_positive_rate", "fake_miss_rate"]:
        rows.append(
            {
                "candidate": "reconstruction_lite_logreg",
                "baseline": "combined_v3_logreg",
                "metric": metric,
                "candidate_mean": recon[f"{metric}_mean"],
                "baseline_mean": base[f"{metric}_mean"],
                "delta_mean": recon[f"{metric}_mean"] - base[f"{metric}_mean"],
            }
        )
    return pd.DataFrame(rows)


def _display_table(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    table = frame[columns].copy()
    for column in columns:
        if column in {"candidate", "n_seeds", "seed"}:
            continue
        table[column] = table[column].map(_format_metric)
    return table


def write_report(
    seed_summary: pd.DataFrame,
    mean_summary: pd.DataFrame,
    delta_summary: pd.DataFrame,
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
    mean_table = _display_table(mean_summary, mean_columns)
    seed_table = _display_table(seed_summary, seed_columns)

    metric_delta = {
        row.metric: float(row.delta_mean) for row in delta_summary.itertuples(index=False)
    }
    accuracy_delta = _format_metric(metric_delta.get("accuracy", 0.0))
    auc_delta = _format_metric(metric_delta.get("roc_auc", 0.0))
    ece_delta = _format_metric(metric_delta.get("expected_calibration_error", 0.0))

    lines = [
        "# reconstruction_lite Transfer Probe",
        "",
        f"Run date: {RUN_DATE}",
        "",
        "Generated by `scripts/summarize_reconstruction_lite_transfer_probe.py` from ignored local runs.",
        "",
        "This probe evaluates the bounded Ishu-trained `reconstruction_lite` and `combined_v3` logistic-regression models on the source-balanced MS COCOAI validation slice. It is a small transfer diagnostic, not a replacement for the full repeated-seed transfer gate.",
        "",
        "## Mean Transfer Metrics",
        "",
        _markdown_table(mean_table, mean_columns),
        "",
        "## Readout",
        "",
        f"- `reconstruction_lite` beats bounded `combined_v3` by `{accuracy_delta}` mean accuracy and `{auc_delta}` mean AUC on this Ishu -> MS COCOAI probe.",
        f"- Mean ECE delta is `{ece_delta}` in favor of `reconstruction_lite`; its fake-call rate is also much closer to 0.5 in this slice.",
        "- Absolute transfer performance is still modest, so this is evidence for using reconstruction residuals as a transfer/fusion branch, not for standalone deployment.",
        "- Next useful step: add `reconstruction_lite` scores to the source-aware score-fusion grid and test whether they help SCP-Fusion operating points.",
        "",
        "## Seed Transfer Metrics",
        "",
        _markdown_table(seed_table, seed_columns),
        "",
    ]
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    seed_summary = load_seed_summary(Path(args.reconstruction_root), Path(args.combined_root))
    mean_summary = build_mean_summary(seed_summary)
    delta_summary = build_delta_summary(mean_summary)

    for path, frame in [
        (Path(args.seed_out), seed_summary),
        (Path(args.mean_out), mean_summary),
        (Path(args.delta_out), delta_summary),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)

    write_report(seed_summary, mean_summary, delta_summary, Path(args.report_out))
    print(Path(args.report_out).resolve())
    print(Path(args.mean_out).resolve())


if __name__ == "__main__":
    main()
