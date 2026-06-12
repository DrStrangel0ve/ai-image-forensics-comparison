from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from forensic_compare.metrics import bootstrap_mean_ci
from forensic_compare.utils import ensure_dir
from scripts.summarize_source_holdout import (
    _metadata_frame,
    _prediction_frame,
    _real_split_keys,
    _source_key,
)
from scripts.summarize_source_holdout_triage import (
    SCORE_MODES,
    _markdown_table,
    _parse_prediction_arg,
    _thresholds,
    _transform_scores,
    _triage_metrics,
    _validate_fraction,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune source-heldout two-threshold triage operating points on calibration "
            "sources, then evaluate the selected point on a held-out generated source."
        )
    )
    parser.add_argument("--metadata", required=True, help="metadata.csv from export_hf_image_dataset.py.")
    parser.add_argument(
        "--predictions",
        action="append",
        required=True,
        help=(
            "Prediction in GROUP:METHOD=PATH form. GROUP is optional; METHOD=PATH "
            "is treated as group 'default'. Repeat for each seed/method."
        ),
    )
    parser.add_argument("--data-dir", default=None, help="Dataset root for reconstructing missing paths.")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--real-source-label", default="0")
    parser.add_argument("--real-test-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--max-real-fpr-grid",
        default="0.02,0.05,0.08,0.10,0.15",
        help="Comma-separated calibration budgets for likely-fake real-image FPR.",
    )
    parser.add_argument(
        "--max-fake-clearance-grid",
        default="0.02,0.05,0.08,0.10,0.15",
        help="Comma-separated calibration budgets for fake images cleared as likely real.",
    )
    parser.add_argument(
        "--score-mode",
        action="append",
        choices=sorted(SCORE_MODES),
        default=None,
        help="Score transform to include in tuning. Repeat to include several modes.",
    )
    parser.add_argument(
        "--real-clearance-weight",
        type=float,
        default=0.5,
        help="Utility reward for each class-balanced real clearance rate.",
    )
    parser.add_argument(
        "--fake-detection-weight",
        type=float,
        default=0.5,
        help="Utility reward for each class-balanced fake detection rate.",
    )
    parser.add_argument(
        "--real-fpr-penalty",
        type=float,
        default=1.0,
        help="Utility penalty for likely-fake calls on real images.",
    )
    parser.add_argument(
        "--fake-clearance-penalty",
        type=float,
        default=1.0,
        help="Utility penalty for clearing fake images as likely real.",
    )
    parser.add_argument("--ci-confidence", type=float, default=0.95)
    parser.add_argument("--ci-resamples", type=int, default=2000)
    parser.add_argument("--ci-seed", type=int, default=0)
    return parser.parse_args()


def _parse_grid(name: str, values: str) -> list[float]:
    parsed: list[float] = []
    for raw_value in values.split(","):
        raw_value = raw_value.strip()
        if not raw_value:
            continue
        value = float(raw_value)
        _validate_fraction(name, value)
        parsed.append(value)
    if not parsed:
        raise ValueError(f"{name} must contain at least one value")
    return sorted(set(parsed))


def _utility(metrics: dict, args: argparse.Namespace) -> float:
    return float(
        args.fake_detection_weight * metrics["fake_detection"]
        + args.real_clearance_weight * metrics["real_clearance"]
        - args.real_fpr_penalty * metrics["real_fpr"]
        - args.fake_clearance_penalty * metrics["fake_false_clearance"]
    )


def _candidate_rows(
    group: str,
    method: str,
    heldout_source: str,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    score_modes: list[str],
    max_real_fpr_grid: list[float],
    max_fake_clearance_grid: list[float],
    args: argparse.Namespace,
) -> list[dict]:
    calibration_y = calibration["y_true"].to_numpy(dtype=int)
    calibration_scores = calibration["fake_score"].to_numpy(dtype=float)
    test_y = test["y_true"].to_numpy(dtype=int)
    test_scores = test["fake_score"].to_numpy(dtype=float)
    rows: list[dict] = []

    for score_mode in score_modes:
        transformed_calibration_scores, transformed_test_scores, temperature = _transform_scores(
            score_mode,
            calibration_y,
            calibration_scores,
            test_scores,
        )
        for max_real_fpr in max_real_fpr_grid:
            for max_fake_clearance in max_fake_clearance_grid:
                real_threshold, fake_threshold = _thresholds(
                    calibration_y,
                    transformed_calibration_scores,
                    max_real_fpr,
                    max_fake_clearance,
                )
                calibration_metrics = _triage_metrics(
                    calibration_y,
                    transformed_calibration_scores,
                    real_threshold,
                    fake_threshold,
                )
                test_metrics = _triage_metrics(
                    test_y,
                    transformed_test_scores,
                    real_threshold,
                    fake_threshold,
                )
                row = {
                    "group": group,
                    "method": method,
                    "score_mode": score_mode,
                    "heldout_source": heldout_source,
                    "calibration_samples": int(len(calibration)),
                    "test_samples": int(len(test)),
                    "max_real_fpr": float(max_real_fpr),
                    "max_fake_clearance": float(max_fake_clearance),
                    "real_threshold": float(real_threshold),
                    "fake_threshold": float(fake_threshold),
                    "temperature": temperature,
                    "calibration_utility": _utility(calibration_metrics, args),
                    "test_utility": _utility(test_metrics, args),
                }
                row.update({f"calibration_{key}": value for key, value in calibration_metrics.items()})
                row.update({f"test_{key}": value for key, value in test_metrics.items()})
                rows.append(row)
    return rows


def _select_row(candidates: list[dict]) -> dict:
    frame = pd.DataFrame(candidates)
    frame = frame.sort_values(
        [
            "calibration_utility",
            "calibration_coverage",
            "calibration_triage_accuracy",
            "max_real_fpr",
            "max_fake_clearance",
        ],
        ascending=[False, False, False, True, True],
    )
    selected = frame.iloc[0].to_dict()
    selected["selected_rank"] = 1
    return selected


def _summary_frame(
    selected: pd.DataFrame,
    confidence: float,
    n_resamples: int,
    seed: int,
) -> pd.DataFrame:
    metric_map = {
        "test_utility": "mean_test_utility",
        "test_coverage": "mean_test_coverage",
        "test_triage_accuracy": "mean_test_triage_accuracy",
        "test_real_fpr": "mean_test_real_fpr",
        "test_fake_false_clearance": "mean_test_fake_false_clearance",
        "test_fake_detection": "mean_test_fake_detection",
        "test_real_clearance": "mean_test_real_clearance",
        "test_fake_precision": "mean_test_fake_precision",
        "test_real_precision": "mean_test_real_precision",
        "calibration_utility": "mean_calibration_utility",
        "max_real_fpr": "mean_selected_max_real_fpr",
        "max_fake_clearance": "mean_selected_max_fake_clearance",
    }
    rows = []
    for method, group in selected.groupby("method", sort=True):
        row = {
            "method": method,
            "n_holdouts": int(len(group)),
            "selected_score_modes": ";".join(
                f"{mode}:{count}" for mode, count in group["score_mode"].value_counts().sort_index().items()
            ),
        }
        for index, (metric, output_name) in enumerate(metric_map.items()):
            interval = bootstrap_mean_ci(
                group[metric].to_numpy(dtype=float),
                confidence=confidence,
                n_resamples=n_resamples,
                seed=seed + index,
            )
            row[output_name] = interval["mean"]
            row[f"{output_name}_ci_low"] = interval["ci_low"]
            row[f"{output_name}_ci_high"] = interval["ci_high"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["mean_test_utility", "mean_test_triage_accuracy", "mean_test_coverage"],
        ascending=[False, False, False],
    )


def main() -> None:
    args = parse_args()
    if args.real_clearance_weight < 0 or args.fake_detection_weight < 0:
        raise ValueError("Utility rewards must be non-negative")
    if args.real_fpr_penalty < 0 or args.fake_clearance_penalty < 0:
        raise ValueError("Utility penalties must be non-negative")
    max_real_fpr_grid = _parse_grid("--max-real-fpr-grid", args.max_real_fpr_grid)
    max_fake_clearance_grid = _parse_grid("--max-fake-clearance-grid", args.max_fake_clearance_grid)
    score_modes = args.score_mode or ["raw", "temperature_balanced"]
    out_dir = ensure_dir(args.out_dir)
    metadata = _metadata_frame(Path(args.metadata), args.split)

    frames = []
    for group, method, path in map(_parse_prediction_arg, args.predictions):
        frame = _prediction_frame(path, method, args.data_dir, args.split)
        frame["group"] = group
        frames.append(frame)
    predictions = pd.concat(frames, ignore_index=True)
    joined = predictions.merge(metadata, on="path_key", suffixes=("_prediction", "_metadata"))
    if len(joined) != len(predictions):
        raise ValueError(f"Matched {len(joined)} prediction rows but expected {len(predictions)}")
    label_mismatches = joined[joined["y_true"].astype(int) != joined["label"].astype(int)]
    if not label_mismatches.empty:
        raise ValueError(
            f"Prediction labels disagree with metadata labels for {len(label_mismatches)} rows"
        )

    real_key = _source_key(args.real_source_label)
    real_mask = joined["source_key"] == real_key
    fake_sources = sorted(
        joined.loc[~real_mask, ["source_key", "source_name"]]
        .drop_duplicates()
        .itertuples(index=False)
    )
    if not fake_sources:
        raise ValueError("No fake source labels were found in the joined predictions")
    real_test_keys = _real_split_keys(joined.loc[real_mask, "path_key"], args.real_test_fraction, args.seed)

    all_candidates = []
    selected_rows = []
    for (group, method), method_frame in joined.groupby(["group", "method"], sort=True):
        is_real = method_frame["source_key"] == real_key
        real_test = method_frame["path_key"].isin(real_test_keys)
        for source_key, source_name in fake_sources:
            heldout_fake = method_frame["source_key"] == source_key
            calibration = method_frame[(is_real & ~real_test) | (~is_real & ~heldout_fake)]
            test = method_frame[(is_real & real_test) | heldout_fake]
            if calibration["y_true"].nunique() < 2 or test["y_true"].nunique() < 2:
                continue
            candidates = _candidate_rows(
                group,
                method,
                source_name,
                calibration,
                test,
                score_modes,
                max_real_fpr_grid,
                max_fake_clearance_grid,
                args,
            )
            all_candidates.extend(candidates)
            selected_rows.append(_select_row(candidates))

    if not selected_rows:
        raise ValueError("No source-holdout triage tuning rows could be computed")

    candidates = pd.DataFrame(
        sorted(
            all_candidates,
            key=lambda row: (
                row["method"],
                row["group"],
                row["heldout_source"],
                row["score_mode"],
                row["max_real_fpr"],
                row["max_fake_clearance"],
            ),
        )
    )
    selected = pd.DataFrame(
        sorted(
            selected_rows,
            key=lambda row: (row["method"], row["group"], row["heldout_source"]),
        )
    )
    summary = _summary_frame(selected, args.ci_confidence, args.ci_resamples, args.ci_seed)

    candidates.to_csv(out_dir / "source_holdout_triage_grid.csv", index=False)
    selected.to_csv(out_dir / "source_holdout_triage_tuned.csv", index=False)
    summary.to_csv(out_dir / "source_holdout_triage_tuned_summary.csv", index=False)
    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["artifact", "path"])
        for artifact in [
            "source_holdout_triage_grid.csv",
            "source_holdout_triage_tuned.csv",
            "source_holdout_triage_tuned_summary.csv",
            "report.md",
        ]:
            writer.writerow([artifact, str((out_dir / artifact).resolve())])

    summary_columns = [
        "method",
        "n_holdouts",
        "selected_score_modes",
        "mean_test_utility",
        "mean_test_utility_ci_low",
        "mean_test_utility_ci_high",
        "mean_test_coverage",
        "mean_test_triage_accuracy",
        "mean_test_real_fpr",
        "mean_test_fake_false_clearance",
        "mean_test_fake_detection",
        "mean_test_real_clearance",
        "mean_selected_max_real_fpr",
        "mean_selected_max_fake_clearance",
    ]
    detail_columns = [
        "group",
        "method",
        "score_mode",
        "heldout_source",
        "max_real_fpr",
        "max_fake_clearance",
        "real_threshold",
        "fake_threshold",
        "calibration_utility",
        "test_utility",
        "test_coverage",
        "test_triage_accuracy",
        "test_real_fpr",
        "test_fake_false_clearance",
        "test_fake_detection",
        "test_real_clearance",
    ]
    report = [
        "# Source-Heldout Utility-Tuned Triage",
        "",
        f"Split: `{args.split}`",
        f"Real test fraction: `{args.real_test_fraction}`",
        f"Seed: `{args.seed}`",
        f"Score modes searched: `{', '.join(score_modes)}`",
        f"Max real-FPR grid: `{', '.join(str(value) for value in max_real_fpr_grid)}`",
        f"Max fake-clearance grid: `{', '.join(str(value) for value in max_fake_clearance_grid)}`",
        (
            "Utility: "
            f"`{args.fake_detection_weight} * fake_detection + "
            f"{args.real_clearance_weight} * real_clearance - "
            f"{args.real_fpr_penalty} * real_fpr - "
            f"{args.fake_clearance_penalty} * fake_false_clearance`"
        ),
        f"Mean confidence intervals: `{args.ci_confidence:.0%}` bootstrap over held-out sources/seeds, `{args.ci_resamples}` resamples",
        "",
        "For each held-out generated source, the script searches score transforms and asymmetric triage budgets on all non-heldout sources plus the real calibration split. It then evaluates the selected operating point on the held-out generated source plus the real test split.",
        "",
        "## Method Summary",
        "",
        _markdown_table(summary.to_dict("records"), summary_columns),
        "",
        "## Selected Held-Out Operating Points",
        "",
        _markdown_table(selected.to_dict("records"), detail_columns),
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(_markdown_table(summary.to_dict("records"), summary_columns))
    print(f"Wrote source-holdout triage tuning summary to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
