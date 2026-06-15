from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd


DEFAULT_RUN_DATE = date.today()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or dry-run the ordered publication artifact/control suite."
    )
    parser.add_argument("--repo-root", default=".", help="Repository root where commands should run.")
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional asset names to run, matching the packet regeneration names.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the plan without executing commands.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running later commands after a failure.",
    )
    parser.add_argument(
        "--out-path",
        default="reports/publication_control_suite_2026_06_14.md",
        help="Markdown run report to write.",
    )
    parser.add_argument(
        "--csv-out",
        default="reports/assets/publication_control_suite.csv",
        help="Machine-readable run report to write.",
    )
    parser.add_argument(
        "--run-date",
        default=DEFAULT_RUN_DATE.isoformat(),
        help="Date to stamp into the generated report, in YYYY-MM-DD format. Defaults to today's local date.",
    )
    return parser.parse_args()


def _load_regen_commands(repo_root: Path) -> list[tuple[str, str]]:
    module_path = repo_root / "scripts" / "build_submission_packet.py"
    spec = importlib.util.spec_from_file_location("build_submission_packet", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.REGEN_COMMANDS)


def _command_parts(command: str) -> list[str]:
    parts = command.split()
    if parts and parts[0] == "python":
        parts[0] = sys.executable
    return parts


def _selected_commands(commands: list[tuple[str, str]], only: list[str] | None) -> list[tuple[str, str]]:
    if not only:
        return commands
    wanted = set(only)
    selected = [(asset, command) for asset, command in commands if asset in wanted]
    missing = sorted(wanted - {asset for asset, _command in selected})
    if missing:
        raise ValueError(f"Unknown --only asset names: {', '.join(missing)}")
    return selected


def _command_stage(asset: str, command: str) -> str:
    text = f"{asset} {command}".lower()
    if "lint" in text:
        return "quality-control"
    if "manuscript" in text or "paper" in text or "poster" in text or "text drafts" in text:
        return "writing-assets"
    if "packet" in text or "scorecard" in text or "upload checklist" in text or "critical path" in text:
        return "submission-planning"
    if "sota" in text or "opportunity" in text or "benchmark readiness" in text:
        return "external-positioning"
    if "reconstruction" in text or "tiled" in text or "fusion" in text or "robustness" in text:
        return "experiments"
    if "table" in text or "figure" in text:
        return "publication-assets"
    return "reference-and-claims"


def _run_suite(
    repo_root: Path,
    commands: list[tuple[str, str]],
    dry_run: bool,
    continue_on_error: bool,
) -> pd.DataFrame:
    rows = []
    for index, (asset, command) in enumerate(commands, start=1):
        started = time.perf_counter()
        if dry_run:
            status = "planned"
            returncode = 0
        else:
            result = subprocess.run(_command_parts(command), cwd=repo_root)
            returncode = int(result.returncode)
            status = "passed" if returncode == 0 else "failed"
        elapsed = time.perf_counter() - started
        rows.append(
            {
                "order": index,
                "stage": _command_stage(asset, command),
                "asset": asset,
                "command": command,
                "status": status,
                "returncode": returncode,
                "elapsed_seconds": round(elapsed, 3),
            }
        )
        if returncode != 0 and not continue_on_error:
            break
    return pd.DataFrame(rows)


def _markdown_escape(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).replace("\n", " ").replace("|", "\\|")


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(_markdown_escape(value) for value in row) + " |")
    return "\n".join(lines)


def _stage_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["stage", "status"], sort=False)
        .size()
        .reset_index(name="commands")
    )


def _write_report(frame: pd.DataFrame, out_path: Path, dry_run: bool, run_date: date) -> None:
    status = "DRY-RUN" if dry_run else "PASS" if set(frame["status"]) <= {"passed"} else "FAIL"
    lines = [
        "# Publication Control Suite",
        "",
        f"Run date: {run_date.isoformat()}",
        "",
        f"Status: **{status}** ({len(frame)} commands listed).",
        "",
        "Generated by `scripts/run_publication_control_suite.py` from `scripts/build_submission_packet.py::REGEN_COMMANDS`.",
        "",
        "Use `--dry-run` to refresh this plan without changing derived artifacts; omit it to execute the commands in order.",
        "",
        "## Command Mix",
        "",
        _markdown_table(_stage_summary(frame)),
        "",
        "## Command Order",
        "",
        _markdown_table(frame),
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    commands = _selected_commands(_load_regen_commands(repo_root), args.only)
    frame = _run_suite(repo_root, commands, args.dry_run, args.continue_on_error)

    csv_path = Path(args.csv_out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    _write_report(frame, Path(args.out_path), args.dry_run, date.fromisoformat(args.run_date))

    print(Path(args.out_path).resolve())
    print(csv_path.resolve())
    if not args.dry_run and not frame["status"].eq("passed").all():
        raise SystemExit("Publication control suite failed")


if __name__ == "__main__":
    main()
