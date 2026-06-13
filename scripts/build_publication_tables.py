from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


CORE_COLUMNS = [
    "finding_id",
    "setting",
    "method",
    "accuracy",
    "auc",
    "brier",
    "ece",
    "predicted_fake_rate",
    "coverage",
    "decided_accuracy",
    "source",
    "interpretation",
]

METHOD_LABELS = {
    "combined_v3": "combined_v3",
    "resnet18": "ResNet-18",
    "physics_guided": "Physics-guided ResNet-18 + combined_v3",
    "convnext_tiny_frozen": "Frozen ConvNeXt-Tiny",
    "scp_fusion_v0": "SCP-Fusion v0",
    "scp_fusion_dinov2": "SCP-Fusion + DINOv2",
    "scp_fusion_clip": "SCP-Fusion + CLIP",
    "scp_fusion_all_foundation": "SCP-Fusion + CLIP + DINOv2",
    "clip_standalone": "Frozen CLIP ViT-B/32",
    "clip_vit_b_32": "Frozen CLIP ViT-B/32",
    "score_fusion_all6_dropout_mean_r35x8": "Reverse all-branch dropout fusion",
    "score_fusion_all6_temp_balanced": "Reverse all-branch fusion + balanced temperature",
    "cap_0p48": "Reverse source-threshold capped fusion",
    "source_utility_unconstrained": "Reverse source-utility model selection",
    "source_utility_cap_0p48": "Reverse capped source-utility model selection",
    "source_holdout_mean_utility_unconstrained": "Reverse source-heldout utility selection",
    "source_holdout_mean_utility_cap_0p48": "Reverse capped source-heldout utility selection",
    "source_holdout_tuned_fusion": "Reverse source-heldout tuned fusion",
    "tuned_fusion_constraint_sweep_best": "Reverse tuned-fusion constraint sweep best",
    "tuned_fusion_jpeg70": "Reverse tuned-fusion JPEG70 robustness",
    "tuned_fusion_transform": "Reverse tuned-fusion target-transform robustness",
}

SAME_DOMAIN_IDS = {
    "combined_v3": "ishu_same_combined_v3",
    "resnet18": "ishu_same_resnet18",
    "physics_guided": "ishu_same_physics_guided",
}

TRANSFER_METHODS = [
    "combined_v3",
    "resnet18",
    "physics_guided",
    "convnext_tiny_frozen",
    "scp_fusion_v0",
]

FOUNDATION_METHODS = [
    "scp_fusion_dinov2",
    "scp_fusion_clip",
    "scp_fusion_all_foundation",
    "clip_standalone",
]

TRIAGE_METHODS = [
    "scp_fusion_all_foundation",
    "clip_standalone",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build paper/poster result tables from checked-in benchmark summaries."
    )
    parser.add_argument(
        "--physics-guided-report",
        default="reports/physics_guided_vs_resnet_2026_06_12.md",
    )
    parser.add_argument(
        "--calibration-summary",
        default="reports/assets/calibration_summary_ishu_ms_cocoai_all4.csv",
    )
    parser.add_argument(
        "--clip-calibration-summary",
        default="reports/assets/score_fusion_clip_calibration_summary.csv",
    )
    parser.add_argument(
        "--clip-triage-5pct",
        default="reports/assets/score_fusion_clip_source_holdout_triage_5pct.csv",
    )
    parser.add_argument(
        "--reverse-all-methods",
        default="reports/assets/ms_cocoai_to_ishu_reverse_all_methods_mean_metrics.csv",
    )
    parser.add_argument(
        "--reverse-fusion-regularization",
        default="reports/assets/ms_cocoai_to_ishu_reverse_fusion_regularization_mean_metrics.csv",
    )
    parser.add_argument(
        "--reverse-threshold-cap",
        default="reports/assets/ms_cocoai_to_ishu_threshold_cap_mean_metrics.csv",
    )
    parser.add_argument(
        "--reverse-model-utility-selection",
        default="reports/assets/ms_cocoai_to_ishu_model_utility_selection_summary.csv",
    )
    parser.add_argument(
        "--reverse-source-holdout-selection",
        default="reports/assets/ms_cocoai_to_ishu_source_holdout_model_selection_summary.csv",
    )
    parser.add_argument(
        "--reverse-source-holdout-tuned-fusion",
        default="reports/assets/ms_cocoai_to_ishu_source_holdout_tuned_fusion_summary.csv",
    )
    parser.add_argument(
        "--reverse-tuned-fusion-constraint-sweep",
        default="reports/assets/ms_cocoai_to_ishu_tuned_fusion_constraint_sweep_summary.csv",
    )
    parser.add_argument(
        "--reverse-tuned-fusion-jpeg70-robustness",
        default="reports/assets/ms_cocoai_to_ishu_tuned_fusion_jpeg70_robustness_summary.csv",
    )
    parser.add_argument(
        "--reverse-tuned-fusion-extra-robustness",
        nargs="*",
        default=[
            "reports/assets/ms_cocoai_to_ishu_tuned_fusion_blur1_robustness_summary.csv",
            "reports/assets/ms_cocoai_to_ishu_tuned_fusion_resize_half_robustness_summary.csv",
            "reports/assets/ms_cocoai_to_ishu_tuned_fusion_crop85_robustness_summary.csv",
            "reports/assets/ms_cocoai_to_ishu_tuned_fusion_jpeg50_robustness_summary.csv",
            "reports/assets/ms_cocoai_to_ishu_tuned_fusion_jpeg30_robustness_summary.csv",
            "reports/assets/ms_cocoai_to_ishu_tuned_fusion_noise3_robustness_summary.csv",
            "reports/assets/ms_cocoai_to_ishu_tuned_fusion_screenshot_robustness_summary.csv",
            "reports/assets/ms_cocoai_to_ishu_tuned_fusion_social_square_robustness_summary.csv",
        ],
        help="Additional tuned-fusion target-transform robustness summary CSVs.",
    )
    parser.add_argument("--out-dir", default="reports/assets")
    return parser.parse_args()


def _blank_row(
    finding_id: str,
    setting: str,
    method: str,
    source: Path,
    interpretation: str,
    **metrics: float | str | None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "finding_id": finding_id,
        "setting": setting,
        "method": method,
        "accuracy": None,
        "auc": None,
        "brier": None,
        "ece": None,
        "predicted_fake_rate": None,
        "coverage": None,
        "decided_accuracy": None,
        "source": source.as_posix(),
        "interpretation": interpretation,
    }
    row.update(metrics)
    return row


def _single_row(frame: pd.DataFrame, column: str, value: str) -> pd.Series:
    rows = frame[frame[column] == value]
    if rows.empty:
        raise ValueError(f"No row found where {column}={value!r}")
    return rows.iloc[0]


def _markdown_table_after_heading(path: Path, heading: str) -> list[list[str]]:
    text = path.read_text(encoding="utf-8")
    if heading not in text:
        raise ValueError(f"Heading {heading!r} not found in {path}")
    section = text.split(heading, 1)[1]
    rows: list[list[str]] = []
    in_table = False
    for line in section.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            if in_table:
                break
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if not cells or cells[0].lower() == "method":
            in_table = True
            continue
        if all(set(cell) <= {"-", ":", " "} for cell in cells):
            in_table = True
            continue
        rows.append(cells)
        in_table = True
    if not rows:
        raise ValueError(f"No markdown table rows found after {heading!r} in {path}")
    return rows


def _same_domain_key(method: str) -> str:
    cleaned = method.replace("`", "").strip()
    if cleaned == "combined_v3":
        return "combined_v3"
    if cleaned == "ResNet-18":
        return "resnet18"
    if cleaned.startswith("physics-guided ResNet-18"):
        return "physics_guided"
    raise ValueError(f"Unexpected same-domain method label: {method!r}")


def _same_domain_rows(path: Path) -> list[dict[str, object]]:
    rows = []
    for cells in _markdown_table_after_heading(path, "## Where Physics Helps"):
        key = _same_domain_key(cells[0])
        accuracy = float(cells[1])
        auc = float(cells[2])
        rows.append(
            _blank_row(
                SAME_DOMAIN_IDS[key],
                "Ishu same-domain, 3 seeds",
                METHOD_LABELS[key],
                path,
                "Physics-guided fusion is the first Ishu same-domain model to beat both standalone branches."
                if key == "physics_guided"
                else "Same-domain comparator for the physics-guided result.",
                accuracy=accuracy,
                auc=auc,
            )
        )
    return rows


def _calibration_metric_rows(path: Path) -> list[dict[str, object]]:
    frame = pd.read_csv(path)
    rows = []
    for method in TRANSFER_METHODS:
        source_row = _single_row(frame, "method", method)
        rows.append(
            _blank_row(
                f"ishu_to_ms_{method}",
                "Ishu -> source-balanced MS COCOAI",
                METHOD_LABELS[method],
                path,
                "Cross-domain ranking and calibration baseline from the repeated-seed transfer suite.",
                accuracy=float(source_row["mean_accuracy"]),
                auc=float(source_row["mean_roc_auc"]),
                brier=float(source_row["mean_brier_score"]),
                ece=float(source_row["mean_expected_calibration_error"]),
                predicted_fake_rate=float(source_row["mean_predicted_positive_rate"]),
            )
        )
    return rows


def _foundation_metric_rows(path: Path) -> list[dict[str, object]]:
    frame = pd.read_csv(path)
    rows = []
    for method in FOUNDATION_METHODS:
        source_row = _single_row(frame, "method", method)
        interpretation = (
            "Current strongest standalone transfer-ranking branch; fusion still trails this ranker."
            if method == "clip_standalone"
            else "Foundation-encoder score-fusion comparator for transfer ranking and calibration."
        )
        rows.append(
            _blank_row(
                f"ishu_to_ms_{method}",
                "Ishu -> source-balanced MS COCOAI, foundation branches",
                METHOD_LABELS[method],
                path,
                interpretation,
                accuracy=float(source_row["mean_accuracy"]),
                auc=float(source_row["mean_roc_auc"]),
                brier=float(source_row["mean_brier_score"]),
                ece=float(source_row["mean_expected_calibration_error"]),
                predicted_fake_rate=float(source_row["mean_predicted_positive_rate"]),
            )
        )
    return rows


def _triage_rows(path: Path) -> list[dict[str, object]]:
    frame = pd.read_csv(path)
    frame = frame[frame["score_mode"] == "raw"].copy()
    rows = []
    for method in TRIAGE_METHODS:
        source_row = _single_row(frame, "method", method)
        rows.append(
            _blank_row(
                f"ishu_to_ms_triage5_{method}",
                "Ishu -> MS COCOAI source-heldout triage, 5% calibration budget",
                METHOD_LABELS[method],
                path,
                "High-confidence decided-case operating point under generator shift.",
                coverage=float(source_row["mean_test_coverage"]),
                decided_accuracy=float(source_row["mean_test_triage_accuracy"]),
            )
        )
    return rows


def _reverse_transfer_rows(all_methods_path: Path, regularization_path: Path) -> list[dict[str, object]]:
    all_methods = pd.read_csv(all_methods_path)
    all_methods = all_methods[all_methods["split"] == "ms_cocoai_to_ishu_test"].copy()
    rows = []
    for method in ["physics_guided_resnet18_combined_v3", "clip_vit_b_32", "score_fusion_all6_temp_balanced"]:
        source_row = _single_row(all_methods, "method", method)
        label_key = "physics_guided" if method == "physics_guided_resnet18_combined_v3" else method
        rows.append(
            _blank_row(
                f"ms_to_ishu_{label_key}",
                "MS COCOAI -> Ishu reverse transfer",
                METHOD_LABELS[label_key],
                all_methods_path,
                "Reverse-transfer comparator showing ranking, calibration, and fake-call bias.",
                accuracy=float(source_row["accuracy"]),
                auc=float(source_row["auc"]),
                brier=float(source_row["brier"]),
                ece=float(source_row["ece"]),
                predicted_fake_rate=float(source_row["predicted_fake_rate"]),
            )
        )

    regularized = pd.read_csv(regularization_path)
    regularized = regularized[regularized["split"] == "ms_cocoai_to_ishu_test"].copy()
    source_row = _single_row(regularized, "config", "score_fusion_all6_dropout_mean_r35x8")
    rows.append(
        _blank_row(
            "ms_to_ishu_branch_dropout_auc",
            "MS COCOAI -> Ishu reverse transfer",
            METHOD_LABELS["score_fusion_all6_dropout_mean_r35x8"],
            regularization_path,
            "Best reverse fusion AUC frontier so far, but still threshold-biased and poorly calibrated.",
            accuracy=float(source_row["accuracy"]),
            auc=float(source_row["auc"]),
            brier=float(source_row["brier"]),
            ece=float(source_row["ece"]),
            predicted_fake_rate=float(source_row["predicted_fake_rate"]),
        )
    )
    return rows


def _reverse_threshold_cap_row(path: Path) -> dict[str, object]:
    frame = pd.read_csv(path)
    frame = frame[frame["variant"] == "ishu_test"].copy()
    source_row = _single_row(frame, "config", "cap_0p48")
    return _blank_row(
        "ms_to_ishu_source_cap_accuracy",
        "MS COCOAI -> Ishu source-threshold operating point",
        METHOD_LABELS["cap_0p48"],
        path,
        "Best reverse-transfer decision operating point so far; fake-call rate is capped using source validation.",
        accuracy=float(source_row["accuracy"]),
        auc=float(source_row["auc"]),
        brier=float(source_row["brier"]),
        ece=float(source_row["ece"]),
        predicted_fake_rate=float(source_row["predicted_fake_rate"]),
    )


def _reverse_model_utility_rows(path: Path) -> list[dict[str, object]]:
    frame = pd.read_csv(path)
    rows = []
    for policy in ["source_utility_unconstrained", "source_utility_cap_0p48"]:
        source_row = _single_row(frame, "selection_policy", policy)
        rows.append(
            _blank_row(
                f"ms_to_ishu_{policy}",
                "MS COCOAI -> Ishu source-utility model selection",
                METHOD_LABELS[policy],
                path,
                (
                    "Source-train utility alone selects over-firing reverse fusion heads."
                    if policy == "source_utility_unconstrained"
                    else "Adding a source fake-rate cap improves the selected operating point but still trails the fixed capped threshold family."
                ),
                accuracy=float(source_row["target_accuracy_mean"]),
                auc=float(source_row["target_roc_auc_mean"]),
                brier=float(source_row["target_brier_score_mean"]),
                ece=float(source_row["target_expected_calibration_error_mean"]),
                predicted_fake_rate=float(source_row["target_predicted_positive_rate_mean"]),
            )
        )
    return rows


def _reverse_source_holdout_rows(path: Path) -> list[dict[str, object]]:
    frame = pd.read_csv(path)
    rows = []
    for policy in [
        "source_holdout_mean_utility_unconstrained",
        "source_holdout_mean_utility_cap_0p48",
    ]:
        source_row = _single_row(frame, "selection_policy", policy)
        rows.append(
            _blank_row(
                f"ms_to_ishu_{policy}",
                "MS COCOAI -> Ishu source-heldout model selection",
                METHOD_LABELS[policy],
                path,
                (
                    "Leave-one-generator-out source utility still selects over-firing reverse fusion heads."
                    if policy == "source_holdout_mean_utility_unconstrained"
                    else "Adding a source fake-rate cap recovers a cleaner heldout-generator utility operating point."
                ),
                accuracy=float(source_row["target_accuracy_mean"]),
                auc=float(source_row["target_roc_auc_mean"]),
                brier=float(source_row["target_brier_score_mean"]),
                ece=float(source_row["target_expected_calibration_error_mean"]),
                predicted_fake_rate=float(source_row["target_predicted_positive_rate_mean"]),
            )
        )
    return rows


def _reverse_source_holdout_tuned_fusion_row(path: Path) -> dict[str, object]:
    frame = pd.read_csv(path)
    source_row = _single_row(frame, "selection_policy", "source_holdout_tuned_fusion")
    return _blank_row(
        "ms_to_ishu_source_holdout_tuned_fusion",
        "MS COCOAI -> Ishu source-heldout tuned fusion",
        METHOD_LABELS["source_holdout_tuned_fusion"],
        path,
        "First training-side constrained source-heldout utility fusion result; improves reverse accuracy/AUC over fixed capped thresholding.",
        accuracy=float(source_row["target_accuracy_mean"]),
        auc=float(source_row["target_roc_auc_mean"]),
        brier=float(source_row["target_brier_score_mean"]),
        ece=float(source_row["target_expected_calibration_error_mean"]),
        predicted_fake_rate=float(source_row["target_predicted_positive_rate_mean"]),
    )


def _reverse_tuned_fusion_constraint_sweep_row(path: Path) -> dict[str, object]:
    frame = pd.read_csv(path)
    source_row = frame.sort_values(
        ["target_accuracy_mean", "target_predicted_positive_rate_mean"],
        ascending=[False, True],
    ).iloc[0]
    return _blank_row(
        "ms_to_ishu_tuned_fusion_constraint_sweep_best",
        "MS COCOAI -> Ishu tuned-fusion source fake-rate constraint sweep",
        METHOD_LABELS["tuned_fusion_constraint_sweep_best"],
        path,
        "Best reverse SCP-Fusion operating point so far; stricter source fake-rate cap reduces target fake-call bias while improving accuracy.",
        accuracy=float(source_row["target_accuracy_mean"]),
        auc=float(source_row["target_roc_auc_mean"]),
        brier=float(source_row["target_brier_score_mean"]),
        ece=float(source_row["target_expected_calibration_error_mean"]),
        predicted_fake_rate=float(source_row["target_predicted_positive_rate_mean"]),
    )


def _reverse_tuned_fusion_robustness_row(path: Path, *, expected_variant: str | None = None) -> dict[str, object]:
    frame = pd.read_csv(path)
    source_row = _single_row(frame, "variant", expected_variant) if expected_variant else frame.iloc[0]
    variant = str(source_row["variant"])
    if variant == "jpeg70":
        finding_id = "ms_to_ishu_tuned_fusion_jpeg70"
        method = METHOD_LABELS["tuned_fusion_jpeg70"]
        interpretation = (
            "Bounded robustness check for the best reverse tuned-fusion cap; "
            "source-selected policy survives JPEG recompression."
        )
    else:
        finding_id = f"ms_to_ishu_tuned_fusion_{variant}"
        method = f"{METHOD_LABELS['tuned_fusion_transform']} ({variant})"
        interpretation = (
            "Target-transform robustness stress test for the best reverse tuned-fusion cap; "
            "source-selected policy is evaluated without target tuning."
        )
    return _blank_row(
        finding_id,
        f"MS COCOAI -> Ishu tuned-fusion {variant} robustness",
        method,
        path,
        interpretation,
        accuracy=float(source_row["target_accuracy_mean"]),
        auc=float(source_row["target_roc_auc_mean"]),
        brier=float(source_row["target_brier_score_mean"]),
        ece=float(source_row["target_expected_calibration_error_mean"]),
        predicted_fake_rate=float(source_row["target_predicted_positive_rate_mean"]),
    )


def _reverse_tuned_fusion_jpeg70_row(path: Path) -> dict[str, object]:
    return _reverse_tuned_fusion_robustness_row(path, expected_variant="jpeg70")


def build_core_results_table(
    physics_guided_report: Path,
    calibration_summary: Path,
    clip_calibration_summary: Path,
    clip_triage_5pct: Path,
    reverse_all_methods: Path,
    reverse_fusion_regularization: Path,
    reverse_threshold_cap: Path,
    reverse_model_utility_selection: Path,
    reverse_source_holdout_selection: Path,
    reverse_source_holdout_tuned_fusion: Path,
    reverse_tuned_fusion_constraint_sweep: Path,
    reverse_tuned_fusion_jpeg70_robustness: Path,
    reverse_tuned_fusion_extra_robustness: list[Path] | None = None,
) -> pd.DataFrame:
    rows = []
    rows.extend(_same_domain_rows(physics_guided_report))
    rows.extend(_calibration_metric_rows(calibration_summary))
    rows.extend(_foundation_metric_rows(clip_calibration_summary))
    rows.extend(_triage_rows(clip_triage_5pct))
    rows.extend(_reverse_transfer_rows(reverse_all_methods, reverse_fusion_regularization))
    rows.append(_reverse_threshold_cap_row(reverse_threshold_cap))
    rows.extend(_reverse_model_utility_rows(reverse_model_utility_selection))
    rows.extend(_reverse_source_holdout_rows(reverse_source_holdout_selection))
    rows.append(_reverse_source_holdout_tuned_fusion_row(reverse_source_holdout_tuned_fusion))
    rows.append(_reverse_tuned_fusion_constraint_sweep_row(reverse_tuned_fusion_constraint_sweep))
    rows.append(_reverse_tuned_fusion_jpeg70_row(reverse_tuned_fusion_jpeg70_robustness))
    for robustness_path in reverse_tuned_fusion_extra_robustness or []:
        if robustness_path.exists():
            rows.append(_reverse_tuned_fusion_robustness_row(robustness_path))
    return pd.DataFrame(rows, columns=CORE_COLUMNS)


def _format_value(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown_table(frame: pd.DataFrame, path: Path) -> None:
    display = frame.copy()
    for column in ["accuracy", "auc", "brier", "ece", "predicted_fake_rate", "coverage", "decided_accuracy"]:
        display[column] = display[column].map(_format_value)
    headers = list(display.columns)
    lines = [
        "# Publication Core Results Table",
        "",
        "Generated by `scripts/build_publication_tables.py` from checked-in benchmark summaries.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in display.itertuples(index=False):
        values = [str(value).replace("\n", " ") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = build_core_results_table(
        Path(args.physics_guided_report),
        Path(args.calibration_summary),
        Path(args.clip_calibration_summary),
        Path(args.clip_triage_5pct),
        Path(args.reverse_all_methods),
        Path(args.reverse_fusion_regularization),
        Path(args.reverse_threshold_cap),
        Path(args.reverse_model_utility_selection),
        Path(args.reverse_source_holdout_selection),
        Path(args.reverse_source_holdout_tuned_fusion),
        Path(args.reverse_tuned_fusion_constraint_sweep),
        Path(args.reverse_tuned_fusion_jpeg70_robustness),
        [Path(path) for path in args.reverse_tuned_fusion_extra_robustness],
    )
    csv_path = out_dir / "publication_core_results.csv"
    markdown_path = out_dir / "publication_core_results.md"
    frame.to_csv(csv_path, index=False)
    write_markdown_table(frame, markdown_path)
    print(csv_path.resolve())
    print(markdown_path.resolve())


if __name__ == "__main__":
    main()
