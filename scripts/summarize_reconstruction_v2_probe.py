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
    _markdown_table,
    _seed_from_path,
)


RUN_DATE = "2026-06-14"
CANDIDATE_ORDER = [
    "combined_v3_logreg",
    "reconstruction_lite_logreg",
    "reconstruction_v2_logreg",
]
SETTING_ORDER = ["ishu_same_bounded", "ishu_to_ms_cocoai_bounded"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize bounded reconstruction_v2 same-domain and transfer probes."
    )
    parser.add_argument("--v2-root", default="runs/reconstruction_v2_probe")
    parser.add_argument("--lite-root", default="runs/reconstruction_lite_probe")
    parser.add_argument("--combined-root", default="runs/combined_v3_probe_for_recon_lite")
    parser.add_argument("--v2-transfer-root", default="runs/reconstruction_v2_probe_to_ms")
    parser.add_argument("--lite-transfer-root", default="runs/reconstruction_lite_probe_to_ms")
    parser.add_argument(
        "--combined-transfer-root",
        default="runs/combined_v3_probe_for_recon_lite_to_ms",
    )
    parser.add_argument(
        "--seed-out",
        default="reports/assets/reconstruction_v2_probe_seed_summary.csv",
    )
    parser.add_argument(
        "--mean-out",
        default="reports/assets/reconstruction_v2_probe_mean_summary.csv",
    )
    parser.add_argument(
        "--delta-out",
        default="reports/assets/reconstruction_v2_probe_delta_summary.csv",
    )
    parser.add_argument(
        "--report-out",
        default="reports/reconstruction_v2_probe_2026_06_14.md",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows_for_root(root: Path, candidate: str, setting: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.glob("seed*/metrics.json")):
        metrics = _read_json(path)
        rows.append(
            {
                "setting": setting,
                "seed": _seed_from_path(path),
                "candidate": candidate,
                "feature_set": metrics.get("feature_set"),
                "classifier": metrics.get("classifier"),
                "method": metrics.get("method"),
                "n_train": metrics.get("n_train"),
                "n_eval": metrics.get("n_target", metrics.get("n_test", metrics.get("n_samples"))),
                "accuracy": metrics.get("accuracy"),
                "roc_auc": metrics.get("roc_auc"),
                "brier_score": metrics.get("brier_score"),
                "expected_calibration_error": metrics.get("expected_calibration_error"),
                "maximum_calibration_error": metrics.get("maximum_calibration_error"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1": metrics.get("f1"),
                **_confusion_rates(metrics),
                "metrics_path": path.as_posix(),
            }
        )
    return rows


def load_seed_summary(
    v2_root: Path,
    lite_root: Path,
    combined_root: Path,
    v2_transfer_root: Path,
    lite_transfer_root: Path,
    combined_transfer_root: Path,
) -> pd.DataFrame:
    rows = []
    rows.extend(_rows_for_root(combined_root, "combined_v3_logreg", "ishu_same_bounded"))
    rows.extend(_rows_for_root(lite_root, "reconstruction_lite_logreg", "ishu_same_bounded"))
    rows.extend(_rows_for_root(v2_root, "reconstruction_v2_logreg", "ishu_same_bounded"))
    rows.extend(
        _rows_for_root(
            combined_transfer_root,
            "combined_v3_logreg",
            "ishu_to_ms_cocoai_bounded",
        )
    )
    rows.extend(
        _rows_for_root(lite_transfer_root, "reconstruction_lite_logreg", "ishu_to_ms_cocoai_bounded")
    )
    rows.extend(_rows_for_root(v2_transfer_root, "reconstruction_v2_logreg", "ishu_to_ms_cocoai_bounded"))
    if not rows:
        raise FileNotFoundError("No reconstruction_v2 probe metrics found")
    frame = pd.DataFrame(rows)
    frame["setting"] = pd.Categorical(frame["setting"], categories=SETTING_ORDER, ordered=True)
    frame["candidate"] = pd.Categorical(frame["candidate"], categories=CANDIDATE_ORDER, ordered=True)
    frame = frame.sort_values(["setting", "seed", "candidate"]).reset_index(drop=True)
    frame["setting"] = frame["setting"].astype(str)
    frame["candidate"] = frame["candidate"].astype(str)
    return frame


def build_mean_summary(seed_summary: pd.DataFrame) -> pd.DataFrame:
    metric_cols = METRIC_COLUMNS + [
        "maximum_calibration_error",
        "fake_call_rate",
        "real_false_positive_rate",
        "fake_miss_rate",
    ]
    rows = []
    group_cols = ["setting", "candidate", "feature_set", "classifier"]
    for keys, group in seed_summary.groupby(group_cols, sort=False, dropna=False):
        row = dict(zip(group_cols, keys))
        row["n_seeds"] = int(group["seed"].nunique())
        row["seeds"] = ",".join(str(seed) for seed in sorted(group["seed"].unique()))
        row["n_eval_mean"] = float(pd.to_numeric(group["n_eval"], errors="coerce").mean())
        for column in metric_cols:
            values = pd.to_numeric(group[column], errors="coerce")
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_std"] = float(values.std(ddof=1)) if len(values.dropna()) > 1 else 0.0
            row[f"{column}_min"] = float(values.min())
            row[f"{column}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


def build_delta_summary(mean_summary: pd.DataFrame) -> pd.DataFrame:
    metric_cols = METRIC_COLUMNS + [
        "maximum_calibration_error",
        "fake_call_rate",
        "real_false_positive_rate",
        "fake_miss_rate",
    ]
    rows = []
    for setting, setting_rows in mean_summary.groupby("setting", sort=False):
        v2_rows = setting_rows[setting_rows["candidate"] == "reconstruction_v2_logreg"]
        if v2_rows.empty:
            continue
        v2 = v2_rows.iloc[0]
        for baseline_name in ["reconstruction_lite_logreg", "combined_v3_logreg"]:
            baseline_rows = setting_rows[setting_rows["candidate"] == baseline_name]
            if baseline_rows.empty:
                continue
            baseline = baseline_rows.iloc[0]
            for metric in metric_cols:
                rows.append(
                    {
                        "setting": setting,
                        "candidate": "reconstruction_v2_logreg",
                        "baseline": baseline_name,
                        "metric": metric,
                        "candidate_mean": v2[f"{metric}_mean"],
                        "baseline_mean": baseline[f"{metric}_mean"],
                        "delta_mean": v2[f"{metric}_mean"] - baseline[f"{metric}_mean"],
                    }
                )
    return pd.DataFrame(rows)


def _display_table(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    table = frame[columns].copy()
    for column in columns:
        if column in {"setting", "candidate", "baseline", "metric", "n_seeds", "seed"}:
            continue
        table[column] = table[column].map(_format_metric)
    return table


def _delta(delta_summary: pd.DataFrame, setting: str, baseline: str, metric: str) -> float:
    row = delta_summary[
        (delta_summary["setting"] == setting)
        & (delta_summary["baseline"] == baseline)
        & (delta_summary["metric"] == metric)
    ]
    if row.empty:
        return 0.0
    return float(row.iloc[0]["delta_mean"])


def write_report(
    seed_summary: pd.DataFrame,
    mean_summary: pd.DataFrame,
    delta_summary: pd.DataFrame,
    report_out: Path,
) -> None:
    mean_columns = [
        "setting",
        "candidate",
        "n_seeds",
        "accuracy_mean",
        "roc_auc_mean",
        "brier_score_mean",
        "expected_calibration_error_mean",
        "fake_call_rate_mean",
    ]
    delta_columns = [
        "setting",
        "baseline",
        "metric",
        "candidate_mean",
        "baseline_mean",
        "delta_mean",
    ]
    seed_columns = [
        "setting",
        "seed",
        "candidate",
        "accuracy",
        "roc_auc",
        "brier_score",
        "expected_calibration_error",
    ]
    mean_table = _display_table(mean_summary, mean_columns)
    delta_table = _display_table(
        delta_summary[delta_summary["metric"].isin(["accuracy", "roc_auc", "brier_score", "expected_calibration_error"])],
        delta_columns,
    )
    seed_table = _display_table(seed_summary, seed_columns)

    same_auc_delta = _format_metric(
        _delta(delta_summary, "ishu_same_bounded", "reconstruction_lite_logreg", "roc_auc")
    )
    transfer_auc_delta = _format_metric(
        _delta(delta_summary, "ishu_to_ms_cocoai_bounded", "reconstruction_lite_logreg", "roc_auc")
    )
    transfer_combined_auc_delta = _format_metric(
        _delta(delta_summary, "ishu_to_ms_cocoai_bounded", "combined_v3_logreg", "roc_auc")
    )

    lines = [
        "# reconstruction_v2 Probe",
        "",
        f"Run date: {RUN_DATE}",
        "",
        "Generated by `scripts/summarize_reconstruction_v2_probe.py` from ignored local runs.",
        "",
        "This bounded diagnostic compares the extended deterministic reconstruction residual branch against `reconstruction_lite` and bounded `combined_v3` on the same Ishu holdout and Ishu -> source-balanced MS COCOAI transfer slice.",
        "",
        "## Mean Metrics",
        "",
        _markdown_table(mean_table, mean_columns),
        "",
        "## Readout",
        "",
        f"- On the bounded Ishu same-domain split, `reconstruction_v2` improves over `reconstruction_lite` by `{same_auc_delta}` mean AUC.",
        f"- On Ishu -> MS COCOAI transfer, `reconstruction_v2` trails `reconstruction_lite` by `{transfer_auc_delta}` mean AUC, but still beats bounded `combined_v3` by `{transfer_combined_auc_delta}` mean AUC.",
        "- Interpretation: richer deterministic reconstruction residuals add same-domain capacity but appear more source-sensitive than the simpler resize-only branch.",
        "- Keep `reconstruction_v2` as an appendix/feature-selection candidate; the next main-method step should be a cached pretrained autoencoder or diffusion reconstruction residual branch with source-aware validation.",
        "",
        "## reconstruction_v2 Deltas",
        "",
        _markdown_table(delta_table, delta_columns),
        "",
        "## Seed Metrics",
        "",
        _markdown_table(seed_table, seed_columns),
        "",
    ]
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    seed_summary = load_seed_summary(
        v2_root=Path(args.v2_root),
        lite_root=Path(args.lite_root),
        combined_root=Path(args.combined_root),
        v2_transfer_root=Path(args.v2_transfer_root),
        lite_transfer_root=Path(args.lite_transfer_root),
        combined_transfer_root=Path(args.combined_transfer_root),
    )
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
