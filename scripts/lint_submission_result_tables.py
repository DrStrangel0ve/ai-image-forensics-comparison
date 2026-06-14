from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import ModuleType

import pandas as pd


METRIC_MAP = {
    "accuracy": "accuracy",
    "auc": "auc",
    "brier": "brier",
    "ece": "ece",
    "fake_call_rate": "predicted_fake_rate",
    "coverage": "coverage",
    "decided_accuracy": "decided_accuracy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lint compact submission result tables against canonical result and claim files."
    )
    parser.add_argument("--repo-root", default=".", help="Repository root for resolving generated assets.")
    parser.add_argument(
        "--core-results",
        default="reports/assets/publication_core_results.csv",
        help="Canonical publication result table.",
    )
    parser.add_argument(
        "--claim-matrix",
        default="reports/assets/claim_evidence_matrix.csv",
        help="Claim/evidence matrix to validate against the canonical result table.",
    )
    parser.add_argument(
        "--manifest",
        default="reports/assets/submission_result_table_manifest.csv",
        help="Generated compact table manifest.",
    )
    parser.add_argument(
        "--report",
        default="reports/submission_result_tables_2026_06_14.md",
        help="Generated Markdown result-table report.",
    )
    parser.add_argument(
        "--out-path",
        default="reports/submission_result_tables_lint_2026_06_14.md",
        help="Markdown lint report to write.",
    )
    parser.add_argument(
        "--checks-out",
        default="reports/assets/submission_result_tables_lint.csv",
        help="Machine-readable lint checks to write.",
    )
    return parser.parse_args()


def _add_check(rows: list[dict[str, object]], check: str, passed: bool, detail: str) -> None:
    rows.append({"check": check, "passed": bool(passed), "detail": detail})


def _load_table_builder(repo_root: Path) -> ModuleType:
    builder_path = repo_root / "scripts" / "build_submission_result_tables.py"
    spec = importlib.util.spec_from_file_location("submission_result_table_builder", builder_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load table builder from {builder_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _values_match(left: object, right: object, tolerance: float = 1e-9) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    if pd.isna(left) or pd.isna(right):
        return False
    if isinstance(left, str) or isinstance(right, str):
        return str(left) == str(right)
    return abs(float(left) - float(right)) <= tolerance


def _split_evidence_ids(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _lint_source_stress_table(rows: list[dict[str, object]], table: pd.DataFrame) -> None:
    required_columns = {
        "selection_policy",
        "heldout_source",
        "utility",
        "recall",
        "fake_miss_rate",
        "predicted_fake_rate",
        "paper_use",
    }
    missing = required_columns - set(table.columns)
    _add_check(
        rows,
        "source_holdout_stress required columns present",
        not missing,
        "all source-stress columns present" if not missing else ", ".join(sorted(missing)),
    )
    if missing:
        return
    utility_sorted = table["utility"].tolist() == sorted(table["utility"].tolist())
    _add_check(
        rows,
        "source_holdout_stress sorted by utility",
        utility_sorted,
        "weakest source appears first" if utility_sorted else "utility order is not ascending",
    )
    miss_mismatches = []
    for row in table.itertuples(index=False):
        expected_miss = 1.0 - float(row.recall)
        if not _values_match(row.fake_miss_rate, expected_miss, tolerance=1e-6):
            miss_mismatches.append(str(row.heldout_source))
    _add_check(
        rows,
        "source_holdout_stress miss rate matches recall",
        not miss_mismatches,
        "all miss rates equal 1 - recall" if not miss_mismatches else ", ".join(miss_mismatches),
    )
    _add_check(
        rows,
        "source_holdout_stress has named sources",
        table["heldout_source"].astype(str).str.len().gt(0).all(),
        f"{len(table)} source rows",
    )


def _lint_reconstruction_ablation_table(rows: list[dict[str, object]], table: pd.DataFrame) -> None:
    required_columns = {
        "setting",
        "setting_label",
        "candidate",
        "method",
        "accuracy",
        "auc",
        "delta_auc_vs_reconstruction_lite",
        "brier",
        "ece",
        "paper_use",
    }
    missing = required_columns - set(table.columns)
    _add_check(
        rows,
        "reconstruction_ablation required columns present",
        not missing,
        "all reconstruction columns present" if not missing else ", ".join(sorted(missing)),
    )
    if missing:
        return
    expected_methods = [
        "combined_v3",
        "reconstruction_lite",
        "reconstruction_v2",
        "combined_v3",
        "reconstruction_lite",
        "reconstruction_v2",
    ]
    _add_check(
        rows,
        "reconstruction_ablation has expected methods",
        table["method"].tolist() == expected_methods,
        f"found {table['method'].tolist()}",
    )
    delta_mismatches = []
    for setting, group in table.groupby("setting", sort=False):
        lite = group[group["method"] == "reconstruction_lite"]
        if lite.empty:
            delta_mismatches.append(f"{setting}:missing_lite")
            continue
        lite_auc = float(lite.iloc[0]["auc"])
        for row in group.itertuples(index=False):
            expected_delta = float(row.auc) - lite_auc
            if not _values_match(row.delta_auc_vs_reconstruction_lite, expected_delta, tolerance=1e-9):
                delta_mismatches.append(f"{setting}:{row.method}")
    _add_check(
        rows,
        "reconstruction_ablation deltas recompute from reconstruction_lite",
        not delta_mismatches,
        "all reconstruction deltas match reconstruction_lite"
        if not delta_mismatches
        else ", ".join(delta_mismatches),
    )
    v2_rows = table[table["method"] == "reconstruction_v2"]
    sign_by_setting = {
        row.setting: float(row.delta_auc_vs_reconstruction_lite) for row in v2_rows.itertuples(index=False)
    }
    expected_signs = (
        sign_by_setting.get("ishu_same_bounded", 0.0) > 0
        and sign_by_setting.get("ishu_to_ms_cocoai_bounded", 0.0) < 0
    )
    _add_check(
        rows,
        "reconstruction_ablation captures source-sensitivity sign",
        expected_signs,
        f"v2 deltas by setting: {sign_by_setting}",
    )


def lint_submission_result_tables(
    repo_root: Path,
    core_results_path: Path,
    claim_matrix_path: Path,
    manifest_path: Path,
    report_path: Path,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    repo_root = repo_root.resolve()
    builder = _load_table_builder(repo_root)
    table_specs = list(builder.TABLES)
    core = pd.read_csv(core_results_path)
    claims = pd.read_csv(claim_matrix_path)
    manifest = pd.read_csv(manifest_path)
    report = report_path.read_text(encoding="utf-8")

    core_ids = set(core["finding_id"])
    expected_table_ids = {spec["table_id"] for spec in table_specs}
    actual_table_ids = set(manifest["table_id"])
    _add_check(
        rows,
        "manifest table ids match builder specs",
        actual_table_ids == expected_table_ids,
        f"expected {sorted(expected_table_ids)}, found {sorted(actual_table_ids)}",
    )

    claim_missing = []
    for claim in claims.itertuples(index=False):
        for finding_id in _split_evidence_ids(getattr(claim, "evidence_finding_ids")):
            if finding_id not in core_ids:
                claim_missing.append(f"{claim.claim_id}:{finding_id}")
    _add_check(
        rows,
        "claim evidence ids resolve to core results",
        not claim_missing,
        "all claim evidence IDs resolve" if not claim_missing else ", ".join(claim_missing),
    )

    for spec in table_specs:
        table_id = spec["table_id"]
        relative_table_path = f"reports/assets/{spec['filename']}"
        table_path = repo_root / relative_table_path
        exists = table_path.exists()
        _add_check(rows, f"{table_id} CSV exists", exists, relative_table_path)
        if not exists:
            continue

        table = pd.read_csv(table_path)

        manifest_row = manifest[manifest["table_id"] == table_id]
        manifest_ok = (
            len(manifest_row) == 1
            and manifest_row.iloc[0]["path"] == f"reports/assets/{spec['filename']}"
            and int(manifest_row.iloc[0]["n_rows"]) == len(table)
        )
        _add_check(
            rows,
            f"{table_id} manifest row is current",
            manifest_ok,
            "path and n_rows match generated CSV" if manifest_ok else "manifest path or n_rows is stale",
        )

        for title in [spec["title"], spec["purpose"]]:
            _add_check(
                rows,
                f"{table_id} report includes {title}",
                title in report,
                "present" if title in report else "missing",
            )

        if "finding_ids" not in spec:
            if table_id == "source_holdout_stress":
                _lint_source_stress_table(rows, table)
            elif table_id == "reconstruction_ablation":
                _lint_reconstruction_ablation_table(rows, table)
            continue

        expected_ids = list(spec["finding_ids"])
        actual_ids = table["finding_id"].tolist()
        _add_check(
            rows,
            f"{table_id} finding order matches builder",
            actual_ids == expected_ids,
            f"expected {expected_ids}, found {actual_ids}",
        )

        metric_mismatches = []
        text_mismatches = []
        for table_row in table.itertuples(index=False):
            matches = core[core["finding_id"] == table_row.finding_id]
            if matches.empty:
                metric_mismatches.append(f"{table_row.finding_id}: missing core row")
                continue
            core_row = matches.iloc[0]
            for table_column, core_column in METRIC_MAP.items():
                if table_column not in table.columns:
                    continue
                if not _values_match(getattr(table_row, table_column), core_row.get(core_column)):
                    metric_mismatches.append(f"{table_row.finding_id}:{table_column}")
            for table_column, core_column in [
                ("method", "method"),
                ("setting", "setting"),
                ("source", "source"),
                ("paper_use", "interpretation"),
            ]:
                if table_column in table.columns and getattr(table_row, table_column) != core_row.get(core_column):
                    text_mismatches.append(f"{table_row.finding_id}:{table_column}")

        _add_check(
            rows,
            f"{table_id} metrics match core results",
            not metric_mismatches,
            "all displayed metrics match core rows" if not metric_mismatches else ", ".join(metric_mismatches),
        )
        _add_check(
            rows,
            f"{table_id} labels match core results",
            not text_mismatches,
            "all labels and interpretations match core rows" if not text_mismatches else ", ".join(text_mismatches),
        )

        if table_id == "robustness_stress":
            clean = table[table["finding_id"] == "ms_to_ishu_tuned_fusion_constraint_sweep_best"].iloc[0]
            delta_mismatches = []
            for table_row in table.itertuples(index=False):
                expected_accuracy_delta = float(table_row.accuracy) - float(clean["accuracy"])
                expected_auc_delta = float(table_row.auc) - float(clean["auc"])
                if not _values_match(table_row.delta_accuracy_vs_clean, expected_accuracy_delta):
                    delta_mismatches.append(f"{table_row.finding_id}:delta_accuracy_vs_clean")
                if not _values_match(table_row.delta_auc_vs_clean, expected_auc_delta):
                    delta_mismatches.append(f"{table_row.finding_id}:delta_auc_vs_clean")
            _add_check(
                rows,
                "robustness deltas recompute from clean baseline",
                not delta_mismatches,
                "all deltas match clean baseline" if not delta_mismatches else ", ".join(delta_mismatches),
            )

    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("\n", " ") for value in row) + " |")
    return "\n".join(lines)


def write_report(frame: pd.DataFrame, out_path: Path) -> None:
    passed = int(frame["passed"].sum())
    total = len(frame)
    status = "PASS" if passed == total else "FAIL"
    lines = [
        "# Submission Result Tables Lint",
        "",
        "Run date: 2026-06-14",
        "",
        f"Status: **{status}** ({passed}/{total} checks passed).",
        "",
        "Generated by `scripts/lint_submission_result_tables.py` from compact result tables, the canonical result table, and the claim-evidence matrix.",
        "",
        _markdown_table(frame),
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    frame = lint_submission_result_tables(
        repo_root=Path(args.repo_root),
        core_results_path=Path(args.core_results),
        claim_matrix_path=Path(args.claim_matrix),
        manifest_path=Path(args.manifest),
        report_path=Path(args.report),
    )
    checks_path = Path(args.checks_out)
    checks_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(checks_path, index=False)
    write_report(frame, Path(args.out_path))
    if not bool(frame["passed"].all()):
        failed = frame[~frame["passed"]]
        raise SystemExit(
            "Submission result-table lint failed: "
            + "; ".join(f"{row.check} ({row.detail})" for row in failed.itertuples(index=False))
        )
    print(Path(args.out_path).resolve())
    print(checks_path.resolve())


if __name__ == "__main__":
    main()
