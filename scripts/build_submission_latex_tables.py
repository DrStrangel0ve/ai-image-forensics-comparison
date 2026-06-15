from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


TABLE_SPECS = [
    {
        "table_id": "same_domain_anchor",
        "caption": "Same-domain anchors on the Ishu benchmark.",
        "label": "tab:same-domain-anchor",
        "columns": [
            ("row_label", "Method", "l"),
            ("accuracy", "Acc.", "r"),
            ("auc", "AUC", "r"),
        ],
    },
    {
        "table_id": "transfer_frontier",
        "caption": "Ishu to source-balanced MS COCOAI transfer and triage frontier.",
        "label": "tab:transfer-frontier",
        "columns": [
            ("row_label", "Method", "l"),
            ("accuracy", "Acc.", "r"),
            ("auc", "AUC", "r"),
            ("brier", "Brier", "r"),
            ("ece", "ECE", "r"),
            ("fake_call_rate", "Fake", "r"),
            ("coverage", "Cov.", "r"),
            ("decided_accuracy", "Dec.", "r"),
        ],
    },
    {
        "table_id": "reverse_operating_points",
        "caption": "MS COCOAI to Ishu reverse-transfer operating points.",
        "label": "tab:reverse-operating-points",
        "columns": [
            ("row_label", "Method", "l"),
            ("accuracy", "Acc.", "r"),
            ("auc", "AUC", "r"),
            ("brier", "Brier", "r"),
            ("ece", "ECE", "r"),
            ("fake_call_rate", "Fake", "r"),
        ],
    },
    {
        "table_id": "robustness_stress",
        "caption": "Reverse tuned-fusion robustness deltas relative to the clean source-capped operating point.",
        "label": "tab:robustness-stress",
        "columns": [
            ("row_label", "Condition", "l"),
            ("accuracy", "Acc.", "r"),
            ("delta_accuracy_vs_clean", "$\\Delta$Acc.", "r"),
            ("auc", "AUC", "r"),
            ("delta_auc_vs_clean", "$\\Delta$AUC", "r"),
            ("fake_call_rate", "Fake", "r"),
        ],
    },
    {
        "table_id": "source_holdout_stress",
        "caption": "Held-out generator stress for the source-capped reverse tuned-fusion policy.",
        "label": "tab:source-holdout-stress",
        "columns": [
            ("row_label", "Held-out", "l"),
            ("recall", "Recall", "r"),
            ("fake_miss_rate", "Miss", "r"),
            ("predicted_fake_rate", "Fake", "r"),
            ("utility", "Utility", "r"),
        ],
    },
    {
        "table_id": "reconstruction_ablation",
        "caption": "Bounded reconstruction residual ablation.",
        "label": "tab:reconstruction-ablation",
        "columns": [
            ("setting_label", "Setting", "l"),
            ("row_label", "Branch", "l"),
            ("accuracy", "Acc.", "r"),
            ("auc", "AUC", "r"),
            ("delta_auc_vs_reconstruction_lite", "$\\Delta$AUC vs lite", "r"),
            ("brier", "Brier", "r"),
            ("ece", "ECE", "r"),
        ],
    },
    {
        "table_id": "method_family_comparison",
        "caption": "Method-family winners by local scenario and criterion.",
        "label": "tab:method-family-comparison",
        "columns": [
            ("scenario_short", "Scenario", "p{0.20\\linewidth}"),
            ("criterion_short", "Criterion", "p{0.18\\linewidth}"),
            ("winner_family", "Family", "p{0.19\\linewidth}"),
            ("winner_short", "Winner", "p{0.30\\linewidth}"),
        ],
    },
    {
        "table_id": "calibration_operating_modes",
        "caption": "Objective-specific operating modes; lower is better for Brier and ECE objectives.",
        "label": "tab:calibration-operating-modes",
        "columns": [
            ("objective_short", "Objective", "p{0.22\\linewidth}"),
            ("method_short", "Selected method", "p{0.27\\linewidth}"),
            ("mode_short", "Mode", "p{0.20\\linewidth}"),
            ("metric_short", "Metric", "p{0.16\\linewidth}"),
            ("metric_value", "Value", "r"),
        ],
    },
    {
        "table_id": "paired_seed_support",
        "caption": "Paired-seed deltas for conservative model-comparison wording; intervals bootstrap the mean seed delta.",
        "label": "tab:paired-seed-support",
        "columns": [
            ("comparison_short", "Comparison", "p{0.30\\linewidth}"),
            ("metric", "Metric", "l"),
            ("delta_ci", "$\\Delta$ (95\\% CI)", "p{0.22\\linewidth}"),
            ("seed_wins", "Seeds", "r"),
            ("support_short", "Support", "p{0.20\\linewidth}"),
        ],
    },
]

ROW_LABELS = {
    "ishu_same_combined_v3": "combined_v3",
    "ishu_same_resnet18": "ResNet-18",
    "ishu_same_physics_guided": "Physics-guided",
    "ishu_to_ms_combined_v3": "combined_v3",
    "ishu_to_ms_resnet18": "ResNet-18",
    "ishu_to_ms_physics_guided": "Physics-guided",
    "ishu_to_ms_convnext_tiny_frozen": "ConvNeXt",
    "ishu_to_ms_scp_fusion_v0": "SCP-Fusion v0",
    "ishu_to_ms_scp_fusion_dinov2": "SCP-Fusion + DINOv2",
    "ishu_to_ms_scp_fusion_all_foundation": "SCP-Fusion + CLIP + DINOv2",
    "ishu_to_ms_clip_standalone": "CLIP",
    "ishu_to_ms_triage5_clip_standalone": "CLIP triage",
    "ms_to_ishu_physics_guided": "Physics-guided",
    "ms_to_ishu_clip_vit_b_32": "CLIP",
    "ms_to_ishu_score_fusion_all6_temp_balanced": "All-branch fusion",
    "ms_to_ishu_branch_dropout_auc": "Branch-dropout fusion",
    "ms_to_ishu_source_cap_accuracy": "Source-threshold cap",
    "ms_to_ishu_source_holdout_tuned_fusion": "Source-heldout tuned",
    "ms_to_ishu_tuned_fusion_constraint_sweep_best": "Clean cap 0.40",
    "ms_to_ishu_tuned_fusion_native_tiling_best": "Native-tiled v3",
    "ms_to_ishu_tuned_fusion_jpeg70": "JPEG70",
    "ms_to_ishu_tuned_fusion_jpeg50": "JPEG50",
    "ms_to_ishu_tuned_fusion_jpeg30": "JPEG30",
    "ms_to_ishu_tuned_fusion_noise3": "Noise3",
    "ms_to_ishu_tuned_fusion_social_square": "Social square",
    "ms_to_ishu_tuned_fusion_social_720p": "Social 720p",
    "ms_to_ishu_tuned_fusion_crop85": "Crop85",
    "ms_to_ishu_tuned_fusion_screenshot": "Screenshot",
    "ms_to_ishu_tuned_fusion_blur1": "Blur1",
    "ms_to_ishu_tuned_fusion_resize_half": "Resize half",
}

SCENARIO_LABELS = {
    "same_domain_ishu": "Same-domain",
    "forward_transfer_ishu_to_ms": "Forward transfer",
    "forward_triage_ishu_to_ms": "Forward triage",
    "reverse_transfer_ms_to_ishu": "Reverse transfer",
}

CRITERION_LABELS = {
    "best_accuracy": "Accuracy",
    "best_auc": "AUC",
    "best_brier": "Brier",
    "best_ece": "ECE",
    "best_decided_accuracy": "Decided acc.",
    "best_coverage": "Coverage",
    "fake_rate_closest_0p5": "Fake gap",
}

OBJECTIVE_LABELS = {
    "threshold_accuracy": "Default threshold",
    "ranking_auc": "Transfer ranking",
    "probability_brier": "Probability error",
    "reliability_ece": "Reliability",
    "source_holdout_accuracy": "Source decision",
    "source_holdout_brier": "Source Brier",
    "source_holdout_ece": "Source ECE",
    "source_holdout_fake_detection": "Generated recall",
    "tiled_dino_accuracy": "Tiled-DINO decision",
    "tiled_dino_auc": "Tiled-DINO ranking",
    "tiled_dino_brier": "Tiled-DINO Brier",
    "tiled_dino_ece": "Tiled-DINO ECE",
}

METHOD_LABELS = {
    "Frozen CLIP ViT-B/32": "Frozen CLIP",
    "SCP-Fusion + CLIP": "SCP-Fusion + CLIP",
    "combined_v4 select-k60": "combined_v4 k60",
    "branch_dropout": "Branch dropout",
    "source_calibrated": "Source calibrated",
    "tiled DINOv2 reverse fusion": "Tiled DINOv2 fusion",
}

MODE_LABELS = {
    "default_score": "Default score",
    "temperature_balanced": "Temp. balanced",
    "tile_max": "tile_max",
    "tile_mean": "tile_mean",
}

METRIC_LABELS = {
    "accuracy": "Acc.",
    "auc": "AUC",
    "brier": "Brier",
    "ece": "ECE",
    "mean_calibrated_accuracy": "Acc.",
    "mean_calibrated_brier_score": "Brier",
    "mean_calibrated_ece": "ECE",
    "mean_calibrated_fake_detection": "Recall",
    "best_accuracy_delta": "dAcc.",
    "best_auc_delta": "dAUC",
    "best_brier_delta": "dBrier",
    "best_ece_delta": "dECE",
}

PAIRED_SEED_TABLE_ROWS = [
    ("ishu_physics_guided_vs_resnet18", "accuracy"),
    ("ishu_physics_guided_vs_resnet18", "AUC"),
    ("combined_v4_ishu_to_ms_selectk60_vs_v3", "AUC"),
    ("combined_v4_ishu_to_ms_selectk60_vs_v3", "ECE"),
    ("ishu_to_ms_scp_all_foundation_vs_clip", "AUC"),
    ("ishu_to_ms_source_calibrated_all_foundation_vs_clip", "Brier"),
    ("ishu_to_ms_source_calibrated_all_foundation_vs_clip", "ECE"),
    ("ms_to_ishu_physics_guided_vs_resnet18", "accuracy"),
    ("ms_to_ishu_physics_guided_vs_resnet18", "ECE"),
    ("ms_to_ishu_temp_balanced_fusion_vs_clip", "accuracy"),
    ("ms_to_ishu_temp_balanced_fusion_vs_clip", "ECE"),
]

PAIRED_COMPARISON_LABELS = {
    "ishu_physics_guided_vs_resnet18": "Ishu: physics-guided vs ResNet",
    "combined_v4_ishu_to_ms_selectk60_vs_v3": "Ishu->MS: v4 k60 vs v3",
    "ishu_to_ms_scp_all_foundation_vs_clip": "Ishu->MS: SCP all-found. vs CLIP",
    "ishu_to_ms_source_calibrated_all_foundation_vs_clip": "Ishu->MS: source-calib. all-found. vs CLIP",
    "ms_to_ishu_physics_guided_vs_resnet18": "MS->Ishu: physics-guided vs ResNet",
    "ms_to_ishu_temp_balanced_fusion_vs_clip": "MS->Ishu: fusion vs CLIP",
}

SUPPORT_LABELS = {
    "consistent_gain_ci_excludes_zero": "consistent gain",
    "all_seeds_favorable_ci_touches_zero": "all seeds favorable",
    "mixed_seed_mean_gain": "mixed mean gain",
    "candidate_trails": "candidate trails",
    "tie_or_no_mean_delta": "tie",
    "diagnostic_shift": "diagnostic shift",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build LaTeX table fragments from compact submission result CSVs."
    )
    parser.add_argument(
        "--table-manifest",
        default="reports/assets/submission_result_table_manifest.csv",
        help="Manifest generated by build_submission_result_tables.py.",
    )
    parser.add_argument("--out-dir", default="reports/assets/latex_tables")
    parser.add_argument(
        "--method-family-comparison",
        default="reports/assets/method_family_comparison.csv",
        help="Optional method-family comparison CSV generated by build_method_family_comparison.py.",
    )
    parser.add_argument(
        "--calibration-operating-modes",
        default="reports/assets/calibration_operating_modes.csv",
        help="Optional calibration operating-mode CSV generated by build_calibration_operating_modes.py.",
    )
    parser.add_argument(
        "--paired-seed-support",
        default="reports/assets/paired_seed_statistical_support.csv",
        help="Optional paired-seed support CSV generated by build_paired_seed_statistical_support.py.",
    )
    parser.add_argument(
        "--report-out",
        default="reports/submission_latex_tables_2026_06_14.md",
        help="Markdown report listing generated LaTeX fragments.",
    )
    return parser.parse_args()


def _latex_escape(value: object) -> str:
    text = "" if value is None or pd.isna(value) else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def _format_cell(value: object, *, signed: bool = False) -> str:
    if value is None or pd.isna(value):
        return "--"
    if isinstance(value, float):
        if signed:
            return f"{value:+.3f}"
        return f"{value:.3f}"
    return _latex_escape(value)


def _format_signed_metric(value: object) -> str:
    if value is None or pd.isna(value):
        return "--"
    return f"{float(value):+.4f}"


def _table_spec(table_id: str) -> dict[str, object]:
    for spec in TABLE_SPECS:
        if spec["table_id"] == table_id:
            return spec
    raise ValueError(f"No LaTeX table spec for table_id={table_id!r}")


def _row_label(row: pd.Series) -> str:
    if "finding_id" in row.index and not pd.isna(row["finding_id"]):
        finding_id = str(row["finding_id"])
        return ROW_LABELS.get(finding_id, str(row["method"]))
    if "heldout_source" in row.index and not pd.isna(row["heldout_source"]):
        return str(row["heldout_source"])
    return str(row.get("method", "row"))


def _winner_label(row: pd.Series) -> str:
    if "finding_id" in row.index and not pd.isna(row["finding_id"]):
        finding_id = str(row["finding_id"])
        if finding_id in ROW_LABELS:
            return ROW_LABELS[finding_id]
    method = str(row.get("method", "winner"))
    replacements = {
        "Physics-guided ResNet-18 + combined_v3": "Physics-guided",
        "Frozen CLIP ViT-B/32": "CLIP",
        "SCP-Fusion + CLIP + DINOv2": "SCP-Fusion all-foundation",
        "SCP-Fusion + CLIP": "SCP-Fusion + CLIP",
        "combined_v4 select-k60": "combined_v4 k60",
        "Reverse tuned fusion + native tiled combined_v3": "Tuned fusion + tiled v3",
        "Reverse tuned-fusion constraint sweep best": "Tuned fusion cap",
        "Reverse capped source-utility model selection": "Source-utility cap",
    }
    return replacements.get(method, method)


def _prepare_table_frame(frame: pd.DataFrame, spec: dict[str, object]) -> pd.DataFrame:
    table_frame = frame.copy()
    if spec["table_id"] == "method_family_comparison":
        table_frame["scenario_short"] = [
            SCENARIO_LABELS.get(str(value), str(row.scenario))
            for row, value in zip(table_frame.itertuples(index=False), table_frame["scenario_id"], strict=False)
        ]
        table_frame["criterion_short"] = [
            CRITERION_LABELS.get(str(value), str(row.criterion_label))
            for row, value in zip(table_frame.itertuples(index=False), table_frame["criterion"], strict=False)
        ]
        table_frame["winner_short"] = [_winner_label(row) for _index, row in table_frame.iterrows()]
    elif spec["table_id"] == "calibration_operating_modes":
        table_frame["objective_short"] = [
            OBJECTIVE_LABELS.get(str(value), str(value)) for value in table_frame["objective"]
        ]
        table_frame["method_short"] = [
            METHOD_LABELS.get(str(value), str(value)) for value in table_frame["selected_method"]
        ]
        table_frame["mode_short"] = [
            MODE_LABELS.get(str(value), str(value)) for value in table_frame["selected_mode"]
        ]
        table_frame["metric_short"] = [
            METRIC_LABELS.get(str(value), str(value)) for value in table_frame["metric"]
        ]
    elif spec["table_id"] == "paired_seed_support":
        table_frame["comparison_short"] = [
            PAIRED_COMPARISON_LABELS.get(str(value), str(value)) for value in table_frame["comparison_id"]
        ]
        table_frame["delta_ci"] = [
            (
                f"{_format_signed_metric(row.raw_delta_mean)} "
                f"[{_format_signed_metric(row.raw_delta_ci_low)}, "
                f"{_format_signed_metric(row.raw_delta_ci_high)}]"
            )
            for row in table_frame.itertuples(index=False)
        ]
        table_frame["seed_wins"] = [
            "--"
            if str(row.direction) == "diagnostic" or pd.isna(row.candidate_wins)
            else f"{int(row.candidate_wins)}/{int(row.n_paired_seeds)}"
            for row in table_frame.itertuples(index=False)
        ]
        table_frame["support_short"] = [
            SUPPORT_LABELS.get(str(value), str(value)) for value in table_frame["support_label"]
        ]
    else:
        table_frame["row_label"] = [_row_label(row) for _index, row in table_frame.iterrows()]
    return table_frame


def _paired_seed_table_frame(source_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(source_path)
    selected_frames = []
    for comparison_id, metric in PAIRED_SEED_TABLE_ROWS:
        subset = frame[(frame["comparison_id"] == comparison_id) & (frame["metric"] == metric)]
        if subset.empty:
            raise ValueError(f"Missing paired-seed row: {comparison_id}/{metric}")
        selected_frames.append(subset.iloc[[0]])
    return pd.concat(selected_frames, ignore_index=True)


def _tabular_spec(columns: list[tuple[str, str, str]]) -> str:
    return " ".join(spec for _column, _label, spec in columns)


def _latex_table(frame: pd.DataFrame, spec: dict[str, object]) -> str:
    columns = list(spec["columns"])
    table_frame = _prepare_table_frame(frame, spec)
    header = " & ".join(label for _column, label, _align in columns) + r" \\"
    body = []
    for row in table_frame.itertuples(index=False):
        row_data = row._asdict()
        cells = []
        for column, _label, _align in columns:
            signed = column.startswith("delta_")
            cells.append(_format_cell(row_data.get(column), signed=signed))
        body.append(" & ".join(cells) + r" \\")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{{_latex_escape(spec['caption'])}}}",
        rf"\label{{{spec['label']}}}",
        rf"\begin{{tabular}}{{{_tabular_spec(columns)}}}",
        r"\toprule",
        header,
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def _write_one_latex_table(
    rows: list[dict[str, object]],
    table_id: str,
    source_path: Path,
    out_dir: Path,
) -> None:
    spec = _table_spec(table_id)
    frame = pd.read_csv(source_path)
    tex = _latex_table(frame, spec)
    out_path = out_dir / f"{table_id}.tex"
    out_path.write_text(tex, encoding="utf-8")
    rows.append(
        {
            "table_id": table_id,
            "source_csv": source_path.as_posix(),
            "latex_path": out_path.as_posix(),
            "caption": spec["caption"],
            "n_rows": len(frame),
        }
    )


def _write_paired_seed_latex_table(
    rows: list[dict[str, object]],
    source_path: Path,
    out_dir: Path,
) -> None:
    table_id = "paired_seed_support"
    spec = _table_spec(table_id)
    frame = _paired_seed_table_frame(source_path)
    tex = _latex_table(frame, spec)
    out_path = out_dir / f"{table_id}.tex"
    out_path.write_text(tex, encoding="utf-8")
    rows.append(
        {
            "table_id": table_id,
            "source_csv": source_path.as_posix(),
            "latex_path": out_path.as_posix(),
            "caption": spec["caption"],
            "n_rows": len(frame),
        }
    )


def build_latex_tables(
    table_manifest: Path,
    out_dir: Path,
    method_family_comparison: Path | None = None,
    calibration_operating_modes: Path | None = None,
    paired_seed_support: Path | None = None,
) -> pd.DataFrame:
    manifest = pd.read_csv(table_manifest)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for manifest_row in manifest.itertuples(index=False):
        source_path = Path(manifest_row.path)
        _write_one_latex_table(rows, manifest_row.table_id, source_path, out_dir)
    if method_family_comparison and method_family_comparison.exists():
        _write_one_latex_table(rows, "method_family_comparison", method_family_comparison, out_dir)
    if calibration_operating_modes and calibration_operating_modes.exists():
        _write_one_latex_table(rows, "calibration_operating_modes", calibration_operating_modes, out_dir)
    if paired_seed_support and paired_seed_support.exists():
        _write_paired_seed_latex_table(rows, paired_seed_support, out_dir)
    latex_manifest = pd.DataFrame(rows)
    latex_manifest.to_csv(out_dir / "submission_latex_table_manifest.csv", index=False)
    return latex_manifest


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("\n", " ") for value in row) + " |")
    return "\n".join(lines)


def write_report(latex_manifest: pd.DataFrame, report_out: Path) -> None:
    lines = [
        "# Submission LaTeX Tables",
        "",
        "Run date: 2026-06-14",
        "",
        "Generated by `scripts/build_submission_latex_tables.py` from compact submission result tables.",
        "",
        "The fragments assume the paper template supports `booktabs` commands (`\\toprule`, `\\midrule`, `\\bottomrule`). They are meant as starting points for WIFS/DFF drafting, not final camera-ready layout.",
        "",
        _markdown_table(latex_manifest),
        "",
    ]
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    latex_manifest = build_latex_tables(
        Path(args.table_manifest),
        Path(args.out_dir),
        Path(args.method_family_comparison),
        Path(args.calibration_operating_modes),
        Path(args.paired_seed_support),
    )
    write_report(latex_manifest, Path(args.report_out))
    print(Path(args.report_out).resolve())
    print((Path(args.out_dir) / "submission_latex_table_manifest.csv").resolve())


if __name__ == "__main__":
    main()
