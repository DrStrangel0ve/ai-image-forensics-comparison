from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning

from forensic_compare.metrics import binary_metrics
from forensic_compare.utils import ensure_dir


RUNS = [
    "combined_v3_logreg",
    "combined_v4_logreg",
    "combined_v4_logreg_selectk60",
]

SOURCE_LABEL_NAMES = {
    0: "real",
    1: "sd21",
    2: "sdxl",
    3: "sd3",
    4: "dalle3",
    5: "midjourney6",
}

METRICS = [
    "accuracy",
    "roc_auc",
    "brier_score",
    "expected_calibration_error",
    "predicted_fake_rate",
    "real_false_positive_rate",
    "fake_detection_rate",
    "mean_fake_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze combined_v4 transfer-gate predictions by Ishu category and MS source."
    )
    parser.add_argument("--source-root", default="runs/combined_v4_full_transfer")
    parser.add_argument("--transfer-root", default="runs/combined_v4_full_transfer_to_ms")
    parser.add_argument(
        "--metadata",
        default="data/raw/ms_cocoai_2026_validation_source_balanced_100/metadata.csv",
    )
    parser.add_argument("--run", action="append", dest="runs", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out-dir", default="reports/assets")
    parser.add_argument(
        "--report-out",
        default="reports/combined_v4_source_slice_diagnostics_2026_06_13.md",
    )
    return parser.parse_args()


def _seed_from_path(path: Path) -> int:
    for part in path.parts:
        if part.startswith("seed") and part[4:].isdigit():
            return int(part[4:])
    raise ValueError(f"Could not infer seed from {path}")


def _norm(path: str | Path) -> str:
    return str(Path(path).resolve()).replace("\\", "/").lower()


def _path_key(path: str | Path) -> str:
    return Path(path).name.lower()


def _canonical_ishu_category(path: str | Path) -> str:
    for part in Path(path).parts:
        lower = part.lower()
        if lower.startswith("ai_") or lower.startswith("real_"):
            category = lower.split("_", maxsplit=1)[1]
            if category == "humans":
                category = "human"
            return category
    return "unknown"


def _source_predictions(source_root: Path, runs: list[str]) -> pd.DataFrame:
    frames = []
    for run in runs:
        for path in sorted(source_root.glob(f"seed*/{run}/predictions.csv")):
            frame = pd.read_csv(path)
            frame["seed"] = _seed_from_path(path)
            frame["run"] = run
            frame["phase"] = "ishu_holdout"
            frame["group_type"] = "ishu_category"
            frame["group_label"] = frame["path"].map(_canonical_ishu_category)
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No source prediction CSVs found under {source_root}")
    return pd.concat(frames, ignore_index=True)


def _metadata_frame(metadata: Path) -> pd.DataFrame:
    frame = pd.read_csv(metadata)
    frame["path_key"] = frame["path"].map(_path_key)
    frame["source_label"] = frame["source_label"].astype(int)
    frame["source_name"] = frame["source_label"].map(
        lambda value: SOURCE_LABEL_NAMES.get(int(value), f"source_{value}")
    )
    return frame[["path_key", "label", "source_label", "source_name"]]


def _transfer_predictions(transfer_root: Path, metadata: Path, runs: list[str]) -> pd.DataFrame:
    metadata_frame = _metadata_frame(metadata)
    frames = []
    for run in runs:
        for path in sorted(transfer_root.glob(f"seed*/{run}/predictions.csv")):
            frame = pd.read_csv(path)
            frame["path_key"] = frame["path"].map(_path_key)
            frame = frame.merge(metadata_frame, on="path_key", how="left")
            missing = frame["source_name"].isna().sum()
            if missing:
                raise ValueError(f"{path} has {missing} rows missing from {metadata}")
            mismatches = frame[frame["y_true"].astype(int) != frame["label"].astype(int)]
            if not mismatches.empty:
                raise ValueError(f"{path} has {len(mismatches)} label mismatches against metadata")
            frame["seed"] = _seed_from_path(path)
            frame["run"] = run
            frame["phase"] = "ishu_to_ms_cocoai"
            frame["group_type"] = "ms_source"
            frame["group_label"] = frame["source_name"]
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No transfer prediction CSVs found under {transfer_root}")
    return pd.concat(frames, ignore_index=True)


def _group_metrics(group: pd.DataFrame, threshold: float) -> dict[str, Any]:
    y_true = group["y_true"].to_numpy(dtype=int)
    scores = group["fake_score"].to_numpy(dtype=float)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        metrics = binary_metrics(y_true, scores, threshold=threshold)
    predictions = scores >= threshold
    real_mask = y_true == 0
    fake_mask = y_true == 1
    metrics.update(
        {
            "n": int(len(group)),
            "n_real": int(real_mask.sum()),
            "n_fake": int(fake_mask.sum()),
            "predicted_fake_rate": float(predictions.mean()),
            "real_false_positive_rate": (
                float(predictions[real_mask].mean()) if real_mask.any() else np.nan
            ),
            "fake_detection_rate": (
                float(predictions[fake_mask].mean()) if fake_mask.any() else np.nan
            ),
            "mean_fake_score": float(scores.mean()),
        }
    )
    return metrics


def build_seed_metrics(
    source_root: Path,
    transfer_root: Path,
    metadata: Path,
    runs: list[str],
    threshold: float,
) -> pd.DataFrame:
    predictions = pd.concat(
        [
            _source_predictions(source_root, runs),
            _transfer_predictions(transfer_root, metadata, runs),
        ],
        ignore_index=True,
    )
    rows = []
    for keys, group in predictions.groupby(
        ["phase", "group_type", "group_label", "seed", "run"],
        sort=True,
        dropna=False,
    ):
        row = dict(zip(["phase", "group_type", "group_label", "seed", "run"], keys))
        row.update(_group_metrics(group, threshold))
        rows.append(row)
    return pd.DataFrame(rows)


def _summary(values: pd.Series) -> dict[str, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {"mean": np.nan, "min": np.nan, "max": np.nan}
    return {
        "mean": float(numeric.mean()),
        "min": float(numeric.min()),
        "max": float(numeric.max()),
    }


def build_mean_metrics(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = ["phase", "group_type", "group_label", "run"]
    for keys, group in seed_metrics.groupby(group_columns, sort=True, dropna=False):
        row = dict(zip(group_columns, keys))
        row["n_seeds"] = int(group["seed"].nunique())
        row["n_mean"] = float(group["n"].mean())
        for metric in METRICS:
            stats = _summary(group[metric])
            row[f"{metric}_mean"] = stats["mean"]
            row[f"{metric}_min"] = stats["min"]
            row[f"{metric}_max"] = stats["max"]
        rows.append(row)
    return pd.DataFrame(rows)


def build_delta_metrics(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    baseline = "combined_v3_logreg"
    rows = []
    candidates = [run for run in sorted(seed_metrics["run"].unique()) if run != baseline]
    group_columns = ["phase", "group_type", "group_label"]
    for keys, group in seed_metrics.groupby(group_columns, sort=True, dropna=False):
        base = group[group["run"] == baseline].set_index("seed")
        if base.empty:
            continue
        for candidate in candidates:
            cand = group[group["run"] == candidate].set_index("seed")
            common = sorted(set(base.index).intersection(set(cand.index)))
            if not common:
                continue
            row = dict(zip(group_columns, keys))
            row["candidate"] = candidate
            row["baseline"] = baseline
            row["n_paired_seeds"] = len(common)
            for metric in METRICS:
                diff = cand.loc[common, metric].astype(float) - base.loc[common, metric].astype(float)
                stats = _summary(diff)
                row[f"{metric}_delta_mean"] = stats["mean"]
                row[f"{metric}_delta_min"] = stats["min"]
                row[f"{metric}_delta_max"] = stats["max"]
            rows.append(row)
    return pd.DataFrame(rows)


def _format(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _index, row in frame.iterrows():
        lines.append("| " + " | ".join(_format(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def _top_deltas(delta: pd.DataFrame, phase: str, metric: str, candidate: str) -> pd.DataFrame:
    frame = delta[(delta["phase"] == phase) & (delta["candidate"] == candidate)].copy()
    if frame.empty:
        return frame
    return frame.sort_values(f"{metric}_delta_mean", ascending=False)


def build_report(mean_metrics: pd.DataFrame, delta_metrics: pd.DataFrame) -> str:
    select_run = "combined_v4_logreg_selectk60"
    raw_run = "combined_v4_logreg"
    ms_select = _top_deltas(delta_metrics, "ishu_to_ms_cocoai", "fake_detection_rate", select_run)
    ishu_select = _top_deltas(delta_metrics, "ishu_holdout", "accuracy", select_run)
    ms_raw_acc = _top_deltas(delta_metrics, "ishu_to_ms_cocoai", "accuracy", raw_run)

    report = [
        "# combined_v4 Source-Slice Diagnostics",
        "",
        "Run date: 2026-06-13",
        "",
        (
            "This source-slice diagnostic decomposes the completed `combined_v4` transfer gate by "
            "Ishu content category and source-balanced MS COCOAI generator label. It uses the saved "
            "`predictions.csv` files, so it is a cheap paper-facing explanation layer rather than a "
            "new training run."
        ),
        "",
        "## Main Read",
        "",
        (
            "`combined_v4_selectk60` helps transfer calibration/ranking on average, but the source-slice "
            "view shows why it should stay an ablation: gains are uneven across MS generator sources, "
            "and the same model loses accuracy on several Ishu content categories."
        ),
        "",
        "## MS COCOAI Select-k60 Detection-Rate Delta",
        "",
        _markdown_table(
            ms_select,
            [
                "group_label",
                "n_paired_seeds",
                "fake_detection_rate_delta_mean",
                "accuracy_delta_mean",
                "roc_auc_delta_mean",
                "brier_score_delta_mean",
                "expected_calibration_error_delta_mean",
            ],
        ),
        "",
        "## Ishu Select-k60 Category Accuracy Delta",
        "",
        _markdown_table(
            ishu_select,
            [
                "group_label",
                "n_paired_seeds",
                "accuracy_delta_mean",
                "roc_auc_delta_mean",
                "brier_score_delta_mean",
                "expected_calibration_error_delta_mean",
                "predicted_fake_rate_delta_mean",
            ],
        ),
        "",
        "## Raw v4 MS COCOAI Accuracy Delta",
        "",
        _markdown_table(
            ms_raw_acc,
            [
                "group_label",
                "n_paired_seeds",
                "accuracy_delta_mean",
                "fake_detection_rate_delta_mean",
                "real_false_positive_rate_delta_mean",
                "mean_fake_score_delta_mean",
            ],
        ),
        "",
        "## Mean Metrics By Slice",
        "",
        _markdown_table(
            mean_metrics,
            [
                "phase",
                "group_label",
                "run",
                "n_seeds",
                "accuracy_mean",
                "roc_auc_mean",
                "brier_score_mean",
                "expected_calibration_error_mean",
                "predicted_fake_rate_mean",
                "fake_detection_rate_mean",
            ],
        ),
        "",
    ]
    return "\n".join(report)


def write_outputs(
    seed_metrics: pd.DataFrame,
    mean_metrics: pd.DataFrame,
    delta_metrics: pd.DataFrame,
    report: str,
    out_dir: Path,
    report_out: Path,
) -> None:
    out_dir = ensure_dir(out_dir)
    seed_metrics.to_csv(out_dir / "combined_v4_source_slice_seed_metrics.csv", index=False)
    mean_metrics.to_csv(out_dir / "combined_v4_source_slice_mean_metrics.csv", index=False)
    delta_metrics.to_csv(out_dir / "combined_v4_source_slice_delta_metrics.csv", index=False)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    runs = args.runs or RUNS
    seed_metrics = build_seed_metrics(
        Path(args.source_root),
        Path(args.transfer_root),
        Path(args.metadata),
        runs,
        args.threshold,
    )
    mean_metrics = build_mean_metrics(seed_metrics)
    delta_metrics = build_delta_metrics(seed_metrics)
    report = build_report(mean_metrics, delta_metrics)
    write_outputs(
        seed_metrics,
        mean_metrics,
        delta_metrics,
        report,
        Path(args.out_dir),
        Path(args.report_out),
    )
    print(Path(args.report_out).resolve())
    print((Path(args.out_dir) / "combined_v4_source_slice_mean_metrics.csv").resolve())


if __name__ == "__main__":
    main()
