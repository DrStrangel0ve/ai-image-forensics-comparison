from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forensic_compare.utils import write_json


REQUIRED_COLUMNS = ["id", "label"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lint a Kaggle FREUID submission before upload.")
    parser.add_argument("--sample-submission", required=True, help="Kaggle sample_submission.csv.")
    parser.add_argument("--submission", required=True, help="Submission CSV to lint.")
    parser.add_argument("--manifest-out", default=None, help="JSON lint report; defaults to <submission>.lint.json.")
    parser.add_argument(
        "--allow-reordered-ids",
        action="store_true",
        help="Allow an exact id set match even if row order differs from the sample submission.",
    )
    parser.add_argument(
        "--allow-score-labels",
        action="store_true",
        help="Allow numeric fraud scores in label instead of requiring binary 0/1 values.",
    )
    return parser.parse_args()


def _check(name: str, passed: bool, detail: str) -> dict[str, object]:
    return {"check": name, "passed": bool(passed), "detail": detail}


def _id_list(frame: pd.DataFrame) -> list[str]:
    return frame["id"].astype(str).tolist()


def lint_freuid_submission(
    sample_submission_path: Path,
    submission_path: Path,
    manifest_path: Path,
    allow_reordered_ids: bool = False,
    allow_score_labels: bool = False,
) -> tuple[bool, dict[str, object]]:
    sample = pd.read_csv(sample_submission_path)
    submission = pd.read_csv(submission_path)
    checks: list[dict[str, object]] = []

    checks.append(
        _check(
            "sample_columns",
            list(sample.columns) == REQUIRED_COLUMNS,
            "sample submission columns match id,label"
            if list(sample.columns) == REQUIRED_COLUMNS
            else f"sample columns are {list(sample.columns)}",
        )
    )
    checks.append(
        _check(
            "submission_columns",
            list(submission.columns) == REQUIRED_COLUMNS,
            "submission columns match id,label"
            if list(submission.columns) == REQUIRED_COLUMNS
            else f"submission columns are {list(submission.columns)}",
        )
    )

    if list(sample.columns) != REQUIRED_COLUMNS or list(submission.columns) != REQUIRED_COLUMNS:
        report = {
            "status": "fail",
            "sample_submission_path": str(sample_submission_path),
            "submission_path": str(submission_path),
            "n_sample_rows": int(len(sample)),
            "n_submission_rows": int(len(submission)),
            "checks": checks,
        }
        write_json(report, manifest_path)
        return False, report

    sample_ids = _id_list(sample)
    submission_ids = _id_list(submission)
    sample_set = set(sample_ids)
    submission_set = set(submission_ids)

    checks.append(
        _check(
            "row_count",
            len(sample_ids) == len(submission_ids),
            f"{len(submission_ids)} rows match sample" if len(sample_ids) == len(submission_ids) else f"sample={len(sample_ids)}, submission={len(submission_ids)}",
        )
    )
    empty_ids = submission["id"].astype(str).str.strip().eq("")
    checks.append(
        _check(
            "ids_nonempty",
            not empty_ids.any(),
            "all ids are non-empty" if not empty_ids.any() else f"{int(empty_ids.sum())} ids are empty",
        )
    )
    duplicate_ids = sorted(submission["id"].astype(str)[submission["id"].astype(str).duplicated()].unique().tolist())
    checks.append(
        _check(
            "ids_unique",
            not duplicate_ids,
            "all ids are unique" if not duplicate_ids else f"duplicate ids: {duplicate_ids[:5]}",
        )
    )
    missing_ids = sorted(sample_set - submission_set)
    extra_ids = sorted(submission_set - sample_set)
    checks.append(
        _check(
            "sample_id_set",
            not missing_ids and not extra_ids,
            "submission id set exactly matches sample"
            if not missing_ids and not extra_ids
            else f"missing={missing_ids[:5]}, extra={extra_ids[:5]}",
        )
    )
    order_ok = sample_ids == submission_ids
    checks.append(
        _check(
            "sample_id_order",
            order_ok or allow_reordered_ids,
            "submission row order matches sample"
            if order_ok
            else "submission id set matches but row order differs; rerun packager to preserve sample order",
        )
    )

    labels = pd.to_numeric(submission["label"], errors="coerce")
    finite_labels = labels.notna().all() and np.isfinite(labels.to_numpy(dtype=float)).all()
    if allow_score_labels:
        valid_labels = finite_labels and labels.between(0, 1).all()
        label_detail = "all labels are finite fraud scores in [0, 1]" if valid_labels else "labels must be finite scores in [0, 1]"
    else:
        valid_labels = finite_labels and labels.isin([0, 1]).all()
        label_detail = "all labels are binary 0/1" if valid_labels else "labels must be numeric binary values 0/1"
    checks.append(
        _check(
            "label_values",
            bool(valid_labels),
            label_detail,
        )
    )

    passed = all(bool(check["passed"]) for check in checks)
    report = {
        "status": "pass" if passed else "fail",
        "sample_submission_path": str(sample_submission_path),
        "submission_path": str(submission_path),
        "n_sample_rows": int(len(sample)),
        "n_submission_rows": int(len(submission)),
        "label_counts": {
            str(key): int(value)
            for key, value in labels.dropna().astype(int).value_counts().sort_index().items()
        }
        if valid_labels
        else None,
        "missing_id_count": int(len(missing_ids)),
        "extra_id_count": int(len(extra_ids)),
        "allow_reordered_ids": allow_reordered_ids,
        "allow_score_labels": allow_score_labels,
        "checks": checks,
    }
    write_json(report, manifest_path)
    return passed, report


def main() -> None:
    args = parse_args()
    submission_path = Path(args.submission)
    manifest_path = Path(args.manifest_out) if args.manifest_out else submission_path.with_suffix(".lint.json")
    passed, _report = lint_freuid_submission(
        sample_submission_path=Path(args.sample_submission),
        submission_path=submission_path,
        manifest_path=manifest_path,
        allow_reordered_ids=args.allow_reordered_ids,
        allow_score_labels=args.allow_score_labels,
    )
    print(manifest_path.resolve())
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
