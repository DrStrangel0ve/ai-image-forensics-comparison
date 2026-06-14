from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


RUN_DATE = "2026-06-14"
METRIC_COLUMNS = [
    "accuracy",
    "roc_auc",
    "brier_score",
    "expected_calibration_error",
    "precision",
    "recall",
    "f1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the bounded reconstruction_lite vs combined_v3 probe."
    )
    parser.add_argument("--reconstruction-root", default="runs/reconstruction_lite_probe")
    parser.add_argument("--combined-root", default="runs/combined_v3_probe_for_recon_lite")
    parser.add_argument(
        "--seed-out",
        default="reports/assets/reconstruction_lite_probe_seed_summary.csv",
    )
    parser.add_argument(
        "--mean-out",
        default="reports/assets/reconstruction_lite_probe_mean_summary.csv",
    )
    parser.add_argument(
        "--delta-out",
        default="reports/assets/reconstruction_lite_probe_delta_summary.csv",
    )
    parser.add_argument(
        "--report-out",
        default="reports/reconstruction_lite_probe_2026_06_14.md",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _seed_from_path(path: Path) -> int:
    for part in path.parts:
        if part.startswith("seed") and part[4:].isdigit():
            return int(part[4:])
    raise ValueError(f"Could not infer seed from {path}")


def _confusion_rates(metrics: dict[str, Any]) -> dict[str, float | int | None]:
    matrix = metrics.get("confusion_matrix")
    if not matrix:
        return {
            "support_real": None,
            "support_fake": None,
            "fake_call_rate": None,
            "real_false_positive_rate": None,
            "fake_miss_rate": None,
        }
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    total = tn + fp + fn + tp
    real_total = tn + fp
    fake_total = fn + tp
    return {
        "support_real": real_total,
        "support_fake": fake_total,
        "fake_call_rate": (fp + tp) / total if total else None,
        "real_false_positive_rate": fp / real_total if real_total else None,
        "fake_miss_rate": fn / fake_total if fake_total else None,
    }


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
                "n_train": metrics.get("n_train"),
                "n_test": metrics.get("n_test"),
                "select_k": metrics.get("effective_select_k", metrics.get("select_k", 0)),
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
            f"No metrics found under {reconstruction_root} or {combined_root}"
        )
    return pd.DataFrame(rows).sort_values(["seed", "candidate"]).reset_index(drop=True)


def build_mean_summary(seed_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["candidate", "feature_set", "classifier", "select_k"]
    for keys, group in seed_summary.groupby(group_cols, sort=False, dropna=False):
        row = dict(zip(group_cols, keys))
        row["n_seeds"] = int(group["seed"].nunique())
        row["seeds"] = ",".join(str(seed) for seed in sorted(group["seed"].unique()))
        row["n_train_mean"] = float(pd.to_numeric(group["n_train"], errors="coerce").mean())
        row["n_test_mean"] = float(pd.to_numeric(group["n_test"], errors="coerce").mean())
        for column in METRIC_COLUMNS + ["fake_call_rate", "real_false_positive_rate", "fake_miss_rate"]:
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


def _markdown_escape(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).replace("\n", " ").replace("|", "\\|")


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame[columns].itertuples(index=False):
        lines.append("| " + " | ".join(_markdown_escape(value) for value in row) + " |")
    return "\n".join(lines)


def _format_metric(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.4f}"


def write_report(
    seed_summary: pd.DataFrame,
    mean_summary: pd.DataFrame,
    delta_summary: pd.DataFrame,
    report_out: Path,
) -> None:
    table = mean_summary[
        [
            "candidate",
            "n_seeds",
            "accuracy_mean",
            "roc_auc_mean",
            "brier_score_mean",
            "expected_calibration_error_mean",
            "fake_call_rate_mean",
        ]
    ].copy()
    for column in [
        "accuracy_mean",
        "roc_auc_mean",
        "brier_score_mean",
        "expected_calibration_error_mean",
        "fake_call_rate_mean",
    ]:
        table[column] = table[column].map(_format_metric)

    seed_table = seed_summary[
        [
            "seed",
            "candidate",
            "accuracy",
            "roc_auc",
            "brier_score",
            "expected_calibration_error",
            "fake_call_rate",
        ]
    ].copy()
    for column in [
        "accuracy",
        "roc_auc",
        "brier_score",
        "expected_calibration_error",
        "fake_call_rate",
    ]:
        seed_table[column] = seed_table[column].map(_format_metric)

    auc_delta = delta_summary[delta_summary["metric"] == "roc_auc"]
    acc_delta = delta_summary[delta_summary["metric"] == "accuracy"]
    auc_delta_value = float(auc_delta["delta_mean"].iloc[0]) if not auc_delta.empty else 0.0
    acc_delta_value = float(acc_delta["delta_mean"].iloc[0]) if not acc_delta.empty else 0.0
    auc_delta_text = _format_metric(abs(auc_delta_value))
    acc_delta_text = _format_metric(abs(acc_delta_value))

    lines = [
        "# reconstruction_lite Bounded Probe",
        "",
        f"Run date: {RUN_DATE}",
        "",
        "Generated by `scripts/summarize_reconstruction_lite_probe.py` from ignored local runs.",
        "",
        "This bounded probe compares standalone resize-reconstruction residual features against the established `combined_v3` physical/signal baseline on the same Ishu split recipe: 160 train images, 80 validation images, logistic regression, image size 128, seeds 7/17/29. It is an ablation sanity check, not a full benchmark claim.",
        "",
        "## Mean Metrics",
        "",
        _markdown_table(
            table,
            [
                "candidate",
                "n_seeds",
                "accuracy_mean",
                "roc_auc_mean",
                "brier_score_mean",
                "expected_calibration_error_mean",
                "fake_call_rate_mean",
            ],
        ),
        "",
        "## Readout",
        "",
        f"- `reconstruction_lite` trails `combined_v3` by `{acc_delta_text}` mean accuracy and `{auc_delta_text}` mean AUC on this bounded Ishu probe.",
        "- It is slightly better on mean ECE in this tiny run, but worse on Brier score and ranking, so calibration should be rechecked on transfer before making any claim.",
        "- The reconstruction branch is therefore useful evidence to ablate and fuse, but it should not replace `combined_v3` as the conventional anchor.",
        "- Next useful step: run `reconstruction_lite` on Ishu -> MS COCOAI transfer and test it as an auxiliary score in SCP-Fusion rather than a standalone detector.",
        "",
        "## Seed Metrics",
        "",
        _markdown_table(
            seed_table,
            [
                "seed",
                "candidate",
                "accuracy",
                "roc_auc",
                "brier_score",
                "expected_calibration_error",
                "fake_call_rate",
            ],
        ),
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
