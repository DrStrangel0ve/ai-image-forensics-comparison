from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path
from typing import Any

from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from forensic_compare.utils import write_json
from lint_freuid_submission import lint_freuid_submission


DEFAULT_COMPETITION = "the-freuid-challenge-2026-ijcai-ecai"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safely lint, submit, and poll a FREUID Kaggle candidate.")
    parser.add_argument("--competition", default=DEFAULT_COMPETITION)
    parser.add_argument("--submission", required=True, help="Submission CSV to upload.")
    parser.add_argument("--sample-submission", required=True, help="Official sample_submission.csv.")
    parser.add_argument("--message", required=True, help="Kaggle submission message.")
    parser.add_argument("--manifest-out", default=None, help="JSON run manifest; defaults to <submission>.submit.json.")
    parser.add_argument("--lint-manifest-out", default=None, help="JSON lint manifest; defaults to <submission>.lint.json.")
    parser.add_argument("--max-daily-submissions", type=int, default=5)
    parser.add_argument("--poll-seconds", type=int, default=180)
    parser.add_argument("--poll-interval", type=int, default=15)
    parser.add_argument("--dry-run", action="store_true", help="Run preflight checks without uploading.")
    parser.add_argument("--force", action="store_true", help="Submit even when the daily-count guard says no slots remain.")
    return parser.parse_args()


def _submission_row(submission: Any) -> dict[str, object]:
    public_score = getattr(submission, "public_score", None) or getattr(submission, "publicScore", None)
    private_score = getattr(submission, "private_score", None) or getattr(submission, "privateScore", None)
    error_description = getattr(submission, "error_description", None) or getattr(submission, "errorDescription", None)
    return {
        "ref": getattr(submission, "ref", None),
        "date": getattr(submission, "date", None).isoformat()
        if getattr(submission, "date", None) is not None
        else None,
        "description": getattr(submission, "description", None),
        "status": str(getattr(submission, "status", "")),
        "publicScore": public_score,
        "privateScore": private_score,
        "errorDescription": error_description,
    }


def _today_submission_count(submissions: list[Any], now_utc: dt.datetime) -> int:
    today = now_utc.date()
    count = 0
    for submission in submissions:
        submitted_at = getattr(submission, "date", None)
        if submitted_at is None:
            continue
        if submitted_at.tzinfo is not None:
            submitted_at = submitted_at.astimezone(dt.timezone.utc).replace(tzinfo=None)
        if submitted_at.date() == today:
            count += 1
    return count


def _find_new_submission(before_refs: set[int], submissions: list[Any], message: str) -> Any | None:
    for submission in submissions:
        ref = getattr(submission, "ref", None)
        description = getattr(submission, "description", "")
        if ref not in before_refs and description == message:
            return submission
    for submission in submissions:
        if getattr(submission, "ref", None) not in before_refs:
            return submission
    return None


def main() -> None:
    args = parse_args()
    submission_path = Path(args.submission)
    manifest_path = Path(args.manifest_out) if args.manifest_out else submission_path.with_suffix(".submit.json")
    lint_manifest_path = (
        Path(args.lint_manifest_out) if args.lint_manifest_out else submission_path.with_suffix(".lint.json")
    )

    lint_passed, lint_report = lint_freuid_submission(
        sample_submission_path=Path(args.sample_submission),
        submission_path=submission_path,
        manifest_path=lint_manifest_path,
        allow_reordered_ids=False,
        allow_score_labels=True,
    )
    if not lint_passed:
        manifest = {"status": "blocked", "reason": "lint_failed", "lint_report": lint_report}
        write_json(manifest, manifest_path)
        raise SystemExit("Submission lint failed")

    api = KaggleApi()
    api.authenticate()
    now_utc = dt.datetime.utcnow()
    before = api.competition_submissions(args.competition)
    before_refs = {int(getattr(submission, "ref")) for submission in before if getattr(submission, "ref", None)}
    today_count = _today_submission_count(before, now_utc)
    preflight = {
        "competition": args.competition,
        "submission_path": str(submission_path),
        "message": args.message,
        "now_utc": now_utc.isoformat(timespec="seconds") + "Z",
        "daily_submission_count_utc": today_count,
        "max_daily_submissions": int(args.max_daily_submissions),
        "remaining_daily_submissions_estimate": max(0, int(args.max_daily_submissions) - today_count),
        "recent_submissions": [_submission_row(submission) for submission in before],
        "lint_manifest_path": str(lint_manifest_path),
    }

    if args.dry_run:
        manifest = {"status": "dry_run_pass", **preflight}
        write_json(manifest, manifest_path)
        print(manifest_path.resolve())
        print("dry_run_pass")
        return

    if today_count >= args.max_daily_submissions and not args.force:
        manifest = {"status": "blocked", "reason": "daily_submission_limit_estimate", **preflight}
        write_json(manifest, manifest_path)
        print(manifest_path.resolve())
        raise SystemExit(
            f"Daily submission guard blocked upload: {today_count}/{args.max_daily_submissions} submissions today UTC"
        )

    submit_error = None
    try:
        response = api.competition_submit(
            file_name=str(submission_path),
            message=args.message,
            competition=args.competition,
            quiet=False,
        )
        response_message = getattr(response, "message", None)
    except Exception as exc:  # Kaggle raises generic API exceptions for quota/format failures.
        submit_error = repr(exc)
        response_message = None

    selected = None
    deadline = time.monotonic() + max(0, args.poll_seconds)
    latest = before
    while submit_error is None and time.monotonic() <= deadline:
        latest = api.competition_submissions(args.competition)
        selected = _find_new_submission(before_refs, latest, args.message)
        if selected is not None and str(getattr(selected, "status", "")).upper().endswith("COMPLETE"):
            break
        time.sleep(max(1, args.poll_interval))

    manifest = {
        "status": "submit_error" if submit_error else "submitted",
        "submit_error": submit_error,
        "submit_response_message": response_message,
        "selected_submission": _submission_row(selected) if selected is not None else None,
        "latest_submissions": [_submission_row(submission) for submission in latest],
        **preflight,
    }
    write_json(manifest, manifest_path)
    print(manifest_path.resolve())
    if submit_error:
        raise SystemExit(submit_error)
    if selected is not None:
        print(
            f"ref={getattr(selected, 'ref', None)} "
            f"status={getattr(selected, 'status', None)} "
            f"publicScore={getattr(selected, 'public_score', None) or getattr(selected, 'publicScore', None)}"
        )


if __name__ == "__main__":
    main()
