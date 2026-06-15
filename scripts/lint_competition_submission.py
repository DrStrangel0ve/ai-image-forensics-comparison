from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forensic_compare.utils import write_json


ID_COLUMNS = [
    "image_id",
    "id",
    "filename",
    "file_name",
    "image",
    "image_path",
    "file_path",
    "filepath",
    "path",
]
PATH_LIKE_COLUMNS = {"image_path", "file_path", "filepath", "path"}
REQUIRED_COLUMNS = ["image_id", "fake_score"]
OPTIONAL_COLUMNS = ["predicted_label", "predicted_label_name", "confidence", "triage_decision"]
LEAKAGE_COLUMNS = {
    "y_true",
    "true_label",
    "label",
    "labels",
    "target",
    "class",
    "class_name",
    "ground_truth",
    "groundtruth",
    "gt",
    "is_fake",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lint a competition-style AI-image detection submission CSV before upload."
    )
    parser.add_argument("--submission", required=True, help="Submission CSV to lint.")
    parser.add_argument(
        "--expected-ids",
        default=None,
        help="Optional CSV with the expected image ids or paths for the challenge split.",
    )
    parser.add_argument(
        "--expected-id-column",
        default=None,
        help="Column to read from --expected-ids. Auto-detected by default.",
    )
    parser.add_argument(
        "--id-from-path",
        choices=["stem", "name", "path"],
        default="stem",
        help="How to convert path-like expected ids into image_id values.",
    )
    parser.add_argument(
        "--manifest-out",
        default=None,
        help="JSON lint report. Defaults to <submission>.lint.json.",
    )
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument("--real-threshold", type=float, default=0.2)
    parser.add_argument("--fake-threshold", type=float, default=0.8)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    return parser.parse_args()


def _detect_column(frame: pd.DataFrame, requested: str | None, candidates: list[str], kind: str) -> str:
    if requested:
        if requested not in frame.columns:
            raise ValueError(f"Requested {kind} column {requested!r} is not present")
        return requested
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise ValueError(f"Could not detect a {kind} column; tried {candidates}")


def _image_id(value: object, column: str, id_from_path: str) -> str:
    raw = str(value)
    if column not in PATH_LIKE_COLUMNS or id_from_path == "path":
        return raw
    path = Path(raw)
    if id_from_path == "name":
        return path.name
    return path.stem


def _check(name: str, passed: bool, detail: str) -> dict[str, object]:
    return {"check": name, "passed": bool(passed), "detail": detail}


def _triage_decision(scores: pd.Series, real_threshold: float, fake_threshold: float) -> pd.Series:
    return pd.Series(
        np.where(scores <= real_threshold, "likely_real", np.where(scores >= fake_threshold, "likely_fake", "uncertain")),
        index=scores.index,
    )


def _load_expected_ids(path: Path, id_column: str | None, id_from_path: str) -> set[str]:
    frame = pd.read_csv(path)
    detected = _detect_column(frame, id_column, ID_COLUMNS, "expected image id")
    return {_image_id(value, detected, id_from_path) for value in frame[detected]}


def lint_submission(
    submission_path: Path,
    manifest_path: Path,
    expected_ids_path: Path | None = None,
    expected_id_column: str | None = None,
    id_from_path: str = "stem",
    decision_threshold: float = 0.5,
    real_threshold: float = 0.2,
    fake_threshold: float = 0.8,
    tolerance: float = 1e-6,
) -> tuple[bool, dict[str, object]]:
    if real_threshold >= fake_threshold:
        raise ValueError("--real-threshold must be lower than --fake-threshold")
    frame = pd.read_csv(submission_path)
    checks: list[dict[str, object]] = []

    missing_required = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    checks.append(
        _check(
            "required_columns",
            not missing_required,
            "all required columns present" if not missing_required else f"missing columns: {missing_required}",
        )
    )
    leakage = sorted(column for column in frame.columns if column.lower() in LEAKAGE_COLUMNS)
    checks.append(
        _check(
            "no_label_leakage_columns",
            not leakage,
            "no label-like leakage columns found" if not leakage else f"forbidden columns: {leakage}",
        )
    )

    if missing_required:
        report = {
            "status": "fail",
            "submission_path": str(submission_path),
            "n_rows": int(len(frame)),
            "checks": checks,
        }
        write_json(report, manifest_path)
        return False, report

    image_ids = frame["image_id"].astype(str)
    empty_ids = image_ids.str.strip().eq("")
    checks.append(
        _check(
            "image_id_nonempty",
            not empty_ids.any(),
            "all image_id values are non-empty" if not empty_ids.any() else f"{int(empty_ids.sum())} empty image_id values",
        )
    )
    duplicate_ids = sorted(image_ids[image_ids.duplicated()].unique().tolist())
    checks.append(
        _check(
            "image_id_unique",
            not duplicate_ids,
            "all image_id values are unique" if not duplicate_ids else f"duplicate ids: {duplicate_ids[:5]}",
        )
    )

    score_errors: str | None = None
    try:
        scores = pd.to_numeric(frame["fake_score"], errors="raise").astype(float)
        finite = np.isfinite(scores.to_numpy()).all()
    except Exception as exc:  # pragma: no cover - exercised through CLI failure in practice
        scores = pd.Series(dtype=float)
        finite = False
        score_errors = str(exc)
    in_range = finite and bool(((scores >= 0.0) & (scores <= 1.0)).all())
    checks.append(
        _check(
            "fake_score_probability",
            in_range,
            "all fake_score values are finite probabilities in [0, 1]"
            if in_range
            else f"invalid fake_score values: {score_errors or 'outside [0, 1] or non-finite'}",
        )
    )

    if in_range and "predicted_label" in frame.columns:
        labels = pd.to_numeric(frame["predicted_label"], errors="coerce")
        expected = (scores >= decision_threshold).astype(int)
        valid_label_values = labels.notna().all() and set(labels.astype(int).unique()).issubset({0, 1})
        consistent = valid_label_values and labels.astype(int).equals(expected)
        checks.append(
            _check(
                "predicted_label_consistent",
                consistent,
                f"predicted_label matches fake_score >= {decision_threshold}"
                if consistent
                else "predicted_label must be 0/1 and match the decision threshold",
            )
        )
    if in_range and "predicted_label_name" in frame.columns:
        expected_names = pd.Series(np.where(scores >= decision_threshold, "fake", "real"), index=frame.index)
        names = frame["predicted_label_name"].astype(str).str.lower()
        consistent = names.equals(expected_names)
        checks.append(
            _check(
                "predicted_label_name_consistent",
                consistent,
                "predicted_label_name matches the decision threshold"
                if consistent
                else "predicted_label_name must match real/fake threshold labels",
            )
        )
    if in_range and "confidence" in frame.columns:
        confidence = pd.to_numeric(frame["confidence"], errors="coerce")
        expected_confidence = (np.abs(scores - 0.5) * 2.0).to_numpy()
        consistent = confidence.notna().all() and np.allclose(confidence.to_numpy(), expected_confidence, atol=tolerance)
        checks.append(
            _check(
                "confidence_consistent",
                consistent,
                "confidence matches abs(fake_score - 0.5) * 2"
                if consistent
                else "confidence must match abs(fake_score - 0.5) * 2",
            )
        )
    if in_range and "triage_decision" in frame.columns:
        expected_triage = _triage_decision(scores, real_threshold, fake_threshold)
        triage = frame["triage_decision"].astype(str)
        consistent = triage.equals(expected_triage)
        checks.append(
            _check(
                "triage_decision_consistent",
                consistent,
                "triage_decision matches real/fake uncertainty thresholds"
                if consistent
                else "triage_decision must match likely_real/uncertain/likely_fake thresholds",
            )
        )

    missing_optional = [column for column in OPTIONAL_COLUMNS if column not in frame.columns]
    checks.append(
        _check(
            "optional_packager_columns",
            True,
            "all packager columns present" if not missing_optional else f"missing optional columns: {missing_optional}",
        )
    )

    expected_id_summary: dict[str, object] | None = None
    if expected_ids_path:
        expected_ids = _load_expected_ids(expected_ids_path, expected_id_column, id_from_path)
        actual_ids = set(image_ids)
        missing_ids = sorted(expected_ids - actual_ids)
        extra_ids = sorted(actual_ids - expected_ids)
        complete = not missing_ids and not extra_ids
        checks.append(
            _check(
                "expected_id_coverage",
                complete,
                "submission image_id values exactly match expected ids"
                if complete
                else f"missing={missing_ids[:5]}, extra={extra_ids[:5]}",
            )
        )
        expected_id_summary = {
            "expected_count": int(len(expected_ids)),
            "missing_count": int(len(missing_ids)),
            "extra_count": int(len(extra_ids)),
            "missing_examples": missing_ids[:10],
            "extra_examples": extra_ids[:10],
        }

    passed = all(bool(check["passed"]) for check in checks if check["check"] != "optional_packager_columns")
    report = {
        "status": "pass" if passed else "fail",
        "submission_path": str(submission_path),
        "expected_ids_path": str(expected_ids_path) if expected_ids_path else None,
        "n_rows": int(len(frame)),
        "columns": list(frame.columns),
        "score_min": float(scores.min()) if in_range and len(scores) else None,
        "score_max": float(scores.max()) if in_range and len(scores) else None,
        "decision_threshold": decision_threshold,
        "real_threshold": real_threshold,
        "fake_threshold": fake_threshold,
        "expected_id_summary": expected_id_summary,
        "checks": checks,
    }
    write_json(report, manifest_path)
    return passed, report


def main() -> None:
    args = parse_args()
    submission_path = Path(args.submission)
    manifest_path = Path(args.manifest_out) if args.manifest_out else submission_path.with_suffix(".lint.json")
    passed, _report = lint_submission(
        submission_path=submission_path,
        manifest_path=manifest_path,
        expected_ids_path=Path(args.expected_ids) if args.expected_ids else None,
        expected_id_column=args.expected_id_column,
        id_from_path=args.id_from_path,
        decision_threshold=args.decision_threshold,
        real_threshold=args.real_threshold,
        fake_threshold=args.fake_threshold,
        tolerance=args.tolerance,
    )
    print(manifest_path.resolve())
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
