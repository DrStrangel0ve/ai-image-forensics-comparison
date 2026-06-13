from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


FOCUS_RUNS = [
    "combined_v3_logreg",
    "combined_v4_logreg",
    "combined_v4_logreg_selectk60",
]

TRANSFER_ROW_PREFIXES = [
    "ishu_to_ms_combined_v4",
    "ishu_to_ms_combined_v4_selectk60",
]

COMMAND_CONFIGS = [
    ("combined_v3_logreg", "combined_v3", "logistic_regression", 0),
    ("combined_v4_logreg", "combined_v4", "logistic_regression", 0),
    ("combined_v4_logreg_selectk60", "combined_v4", "logistic_regression", 60),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a combined_v4 transfer-readiness report from checked-in summaries."
    )
    parser.add_argument(
        "--medium-summary",
        default="reports/assets/combined_v4_medium_selectk_probe/feature_ablation_summary.csv",
    )
    parser.add_argument(
        "--small-summary",
        default="reports/assets/combined_v4_selectk_probe/feature_ablation_summary.csv",
    )
    parser.add_argument(
        "--core-results",
        default="reports/assets/publication_core_results.csv",
    )
    parser.add_argument(
        "--out-path",
        default="reports/combined_v4_transfer_readiness_2026_06_13.md",
    )
    parser.add_argument(
        "--table-out",
        default="reports/assets/combined_v4_transfer_readiness.csv",
    )
    parser.add_argument(
        "--commands-out",
        default="reports/assets/combined_v4_transfer_command_manifest.csv",
    )
    parser.add_argument("--source-data-dir", default="data/raw/ishu_ai_vs_real_2026")
    parser.add_argument(
        "--target-data-dir",
        default="data/raw/ms_cocoai_2026_validation_source_balanced_100",
    )
    parser.add_argument("--run-root", default="runs/combined_v4_full_transfer")
    parser.add_argument("--transfer-root", default="runs/combined_v4_full_transfer_to_ms")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 29])
    return parser.parse_args()


def _load_focus(path: Path, scale: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    rows = frame[frame["run"].isin(FOCUS_RUNS)].copy()
    if rows.empty:
        raise ValueError(f"No focus runs found in {path}")
    rows.insert(0, "scale", scale)
    return rows


def _row(frame: pd.DataFrame, run: str) -> pd.Series:
    matches = frame[frame["run"] == run]
    if matches.empty:
        raise ValueError(f"Missing run={run!r}")
    return matches.iloc[0]


def _overlaps(left_low: float, left_high: float, right_low: float, right_high: float) -> bool:
    return left_low <= right_high and right_low <= left_high


def _delta_rows(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scale, group in summary.groupby("scale", sort=False):
        baseline = _row(group, "combined_v3_logreg")
        for run in ["combined_v4_logreg", "combined_v4_logreg_selectk60"]:
            candidate = _row(group, run)
            rows.append(
                {
                    "scale": scale,
                    "candidate": run,
                    "accuracy_delta_vs_v3": candidate["accuracy_mean"] - baseline["accuracy_mean"],
                    "auc_delta_vs_v3": candidate["roc_auc_mean"] - baseline["roc_auc_mean"],
                    "brier_delta_vs_v3": candidate["brier_score_mean"] - baseline["brier_score_mean"],
                    "ece_delta_vs_v3": candidate["expected_calibration_error_mean"]
                    - baseline["expected_calibration_error_mean"],
                    "candidate_auc_ci": (
                        f"[{candidate['roc_auc_ci_low']:.4f}, {candidate['roc_auc_ci_high']:.4f}]"
                    ),
                    "v3_auc_ci": f"[{baseline['roc_auc_ci_low']:.4f}, {baseline['roc_auc_ci_high']:.4f}]",
                    "auc_ci_overlap": _overlaps(
                        candidate["roc_auc_ci_low"],
                        candidate["roc_auc_ci_high"],
                        baseline["roc_auc_ci_low"],
                        baseline["roc_auc_ci_high"],
                    ),
                }
            )
    return pd.DataFrame(rows)


def _format(value: object) -> str:
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
    for _index, row in frame.iterrows():
        lines.append("| " + " | ".join(_format(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def _transfer_status(core_results: Path) -> tuple[bool, pd.DataFrame]:
    core = pd.read_csv(core_results)
    found = core[
        core["finding_id"].astype(str).apply(
            lambda value: any(value.startswith(prefix) for prefix in TRANSFER_ROW_PREFIXES)
        )
    ].copy()
    return not found.empty, found


def _commands(
    seeds: list[int],
    source_data_dir: str,
    target_data_dir: str,
    run_root: str,
    transfer_root: str,
    image_size: int,
) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        for name, feature_set, classifier, select_k in COMMAND_CONFIGS:
            model_dir = f"{run_root}/seed{seed}/{name}"
            transfer_dir = f"{transfer_root}/seed{seed}/{name}"
            train_command = (
                "python scripts/run_feature_baseline.py "
                f"--data-dir {source_data_dir} "
                f"--output-dir {model_dir} "
                f"--feature-set {feature_set} "
                f"--classifier {classifier} "
                f"--select-k {select_k} "
                f"--image-size {image_size} "
                f"--seed {seed} "
                "--val-fraction 0.2 "
                "--skip-errors"
            )
            eval_command = (
                "python scripts/evaluate_feature_model.py "
                f"--model-dir {model_dir} "
                f"--target-dir {target_data_dir} "
                f"--output-dir {transfer_dir} "
                f"--image-size {image_size} "
                "--target-split all "
                f"--seed {seed} "
                "--skip-errors"
            )
            rows.append(
                {
                    "phase": "train",
                    "seed": seed,
                    "run": name,
                    "feature_set": feature_set,
                    "select_k": select_k,
                    "command": train_command,
                }
            )
            rows.append(
                {
                    "phase": "transfer_eval",
                    "seed": seed,
                    "run": name,
                    "feature_set": feature_set,
                    "select_k": select_k,
                    "command": eval_command,
                }
            )
    return pd.DataFrame(rows)


def build_report(
    medium_summary: Path,
    small_summary: Path,
    core_results: Path,
    source_data_dir: str,
    target_data_dir: str,
    run_root: str,
    transfer_root: str,
    image_size: int,
    seeds: list[int],
) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    summary = pd.concat(
        [
            _load_focus(small_summary, "small_selectk_probe"),
            _load_focus(medium_summary, "medium_240_train_probe"),
        ],
        ignore_index=True,
    )
    deltas = _delta_rows(summary)
    has_transfer, transfer_rows = _transfer_status(core_results)
    command_manifest = _commands(
        seeds,
        source_data_dir,
        target_data_dir,
        run_root,
        transfer_root,
        image_size,
    )

    medium_raw = deltas[
        (deltas["scale"] == "medium_240_train_probe")
        & (deltas["candidate"] == "combined_v4_logreg")
    ].iloc[0]
    medium_select = deltas[
        (deltas["scale"] == "medium_240_train_probe")
        & (deltas["candidate"] == "combined_v4_logreg_selectk60")
    ].iloc[0]
    intro_sentence = (
        "This report records the completed `combined_v4` transfer gate for WIFS/DFF."
        if has_transfer
        else (
            "This report turns the current `combined_v4` ablation evidence into a decision gate for "
            "WIFS/DFF. It does not promote `combined_v4` to the main method; it defines what has to be "
            "run before that claim is safe."
        )
    )
    transfer_sentence = (
        "The `combined_v4` transfer rows are present in the core table; the gate outcome is now checked in."
        if has_transfer
        else "No `combined_v4` transfer row is present in the core table yet."
    )
    command_next_steps = (
        [
            "The manifest remains checked in for reproduction or extension.",
            "",
            "To reproduce the gate:",
            "",
            "1. Run all `train` commands.",
            "2. Run all `transfer_eval` commands.",
            "3. Run `python scripts\\summarize_combined_v4_transfer.py` and rebuild publication tables.",
        ]
        if has_transfer
        else [
            "Recommended execution order:",
            "",
            "1. Run all `train` commands.",
            "2. Run all `transfer_eval` commands.",
            "3. Summarize the resulting `metrics.json` files and add the new transfer rows to the publication core table.",
        ]
    )
    decision_sentence = (
        "`combined_v4` should remain an ablation candidate for now. In the medium bounded Ishu "
        f"probe, raw v4 improves AUC over `combined_v3` by {medium_raw['auc_delta_vs_v3']:.4f} "
        f"and accuracy by {medium_raw['accuracy_delta_vs_v3']:.4f}, but the AUC intervals still "
        f"{'overlap' if medium_raw['auc_ci_overlap'] else 'do not overlap'}. Select-k60 is the "
        f"calibration-friendly variant, changing ECE by {medium_select['ece_delta_vs_v3']:.4f} "
        "relative to v3 on the same medium probe."
    )
    if has_transfer:
        decision_sentence += (
            " The completed full-transfer gate keeps `combined_v3` as the main conventional "
            "baseline and moves `combined_v4_selectk60` into the appendix-ablation bucket."
        )

    lines = [
        "# combined_v4 Transfer Readiness",
        "",
        "Run date: 2026-06-13",
        "",
        intro_sentence,
        "",
        "## Current Decision",
        "",
        decision_sentence,
        "",
        transfer_sentence,
        "",
        "## Delta Versus combined_v3",
        "",
        _markdown_table(
            deltas,
            [
                "scale",
                "candidate",
                "accuracy_delta_vs_v3",
                "auc_delta_vs_v3",
                "brier_delta_vs_v3",
                "ece_delta_vs_v3",
                "candidate_auc_ci",
                "v3_auc_ci",
                "auc_ci_overlap",
            ],
        ),
        "",
        "## Transfer Gate",
        "",
        "Promote raw `combined_v4` to the main conventional branch only if the full repeated-seed run shows:",
        "",
        "- Ishu same-domain AUC and accuracy stay at or above `combined_v3` without worse calibration.",
        "- Ishu -> MS COCOAI transfer AUC improves over the current `combined_v3` transfer baseline.",
        "- Source-heldout or source-balanced evaluation does not introduce a larger fake-call-rate bias.",
        "",
        "Keep select-k60 as a calibration ablation if it keeps lower Brier/ECE even when raw v4 has the better ranking.",
        "",
        "## Command Manifest",
        "",
        f"Commands are written to `reports/assets/combined_v4_transfer_command_manifest.csv` for seeds {', '.join(map(str, seeds))}.",
        "",
        *command_next_steps,
        "",
        "First command:",
        "",
        "```powershell",
        str(command_manifest.iloc[0]["command"]).replace("/", "\\"),
        "```",
        "",
        "First transfer command:",
        "",
        "```powershell",
        str(command_manifest[command_manifest["phase"] == "transfer_eval"].iloc[0]["command"]).replace("/", "\\"),
        "```",
        "",
    ]
    if has_transfer:
        lines.extend(
            [
                "## Existing Transfer Rows",
                "",
                _markdown_table(
                    transfer_rows,
                    ["finding_id", "setting", "method", "accuracy", "auc", "brier", "ece"],
                ),
                "",
            ]
        )
    return "\n".join(lines), deltas, command_manifest


def main() -> None:
    args = parse_args()
    report, deltas, command_manifest = build_report(
        Path(args.medium_summary),
        Path(args.small_summary),
        Path(args.core_results),
        args.source_data_dir,
        args.target_data_dir,
        args.run_root,
        args.transfer_root,
        args.image_size,
        args.seeds,
    )
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")

    table_out = Path(args.table_out)
    table_out.parent.mkdir(parents=True, exist_ok=True)
    deltas.to_csv(table_out, index=False)

    commands_out = Path(args.commands_out)
    commands_out.parent.mkdir(parents=True, exist_ok=True)
    command_manifest.to_csv(commands_out, index=False)

    print(out_path.resolve())
    print(table_out.resolve())
    print(commands_out.resolve())


if __name__ == "__main__":
    main()
