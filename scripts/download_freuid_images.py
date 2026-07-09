from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forensic_compare.datasets import stable_path_score
from forensic_compare.freuid import freuid_competition_path
from forensic_compare.utils import write_json


DEFAULT_COMPETITION = "the-freuid-challenge-2026-ijcai-ecai"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download FREUID images by metadata CSV with Kaggle's nested paths.")
    parser.add_argument("--metadata-csv", required=True, help="CSV with image_path or id values.")
    parser.add_argument("--out-dir", default="data/raw/freuid_2026/images")
    parser.add_argument("--manifest-out", default=None, help="JSON report; defaults to <out-dir>/download_manifest.json.")
    parser.add_argument("--competition", default=DEFAULT_COMPETITION)
    parser.add_argument("--path-column", default=None, help="Column with train/public_test relative image paths.")
    parser.add_argument("--id-column", default="id", help="Column with image ids when --path-column is omitted.")
    parser.add_argument("--split", choices=["train", "public_test"], default="public_test")
    parser.add_argument("--limit", type=int, default=0, help="Deterministic subset size; 0 means all rows.")
    parser.add_argument(
        "--balance-columns",
        nargs="+",
        default=[],
        help="Optional metadata columns used to allocate --limit evenly across strata, e.g. label type.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Write the manifest after this many completed rows; 0 only writes at the end.",
    )
    parser.add_argument(
        "--stop-after-failures",
        type=int,
        default=0,
        help="Stop early after this many consecutive failed downloads; 0 disables early stopping.",
    )
    return parser.parse_args()


def _limited_frame(frame: pd.DataFrame, limit: int, seed: int, balance_columns: list[str]) -> pd.DataFrame:
    if limit <= 0 or limit >= len(frame):
        return frame
    missing = [column for column in balance_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Balance columns are missing from metadata: {missing}")
    scored = frame.copy()
    scored["_download_score"] = scored.astype(str).agg("|".join, axis=1).map(lambda value: stable_path_score(value, seed))
    if not balance_columns:
        return scored.sort_values(["_download_score"]).head(limit).drop(columns=["_download_score"])

    groups = [group.sort_values(["_download_score"]) for _key, group in scored.groupby(balance_columns, sort=True)]
    selected_indices: list[int] = []
    cursor = 0
    while len(selected_indices) < limit:
        progressed = False
        for group in groups:
            if cursor < len(group):
                selected_indices.append(int(group.index[cursor]))
                progressed = True
                if len(selected_indices) >= limit:
                    break
        if not progressed:
            break
        cursor += 1
    return scored.loc[selected_indices].drop(columns=["_download_score"])


def build_download_plan(
    metadata_csv: Path,
    path_column: str | None = None,
    id_column: str = "id",
    split: str = "public_test",
    limit: int = 0,
    seed: int = 7,
    balance_columns: list[str] | None = None,
) -> list[str]:
    frame = pd.read_csv(metadata_csv)
    frame = _limited_frame(frame, limit=limit, seed=seed, balance_columns=list(balance_columns or []))
    if path_column is not None:
        if path_column not in frame.columns:
            raise ValueError(f"Requested path column {path_column!r} not found")
        values = frame[path_column].astype(str).tolist()
        paths = [freuid_competition_path(value) for value in values]
    else:
        if id_column not in frame.columns:
            raise ValueError(f"Requested id column {id_column!r} not found")
        values = frame[id_column].astype(str).tolist()
        paths = [freuid_competition_path(value, split=split) for value in values]
    paths = sorted(set(paths), key=lambda value: stable_path_score(value, seed))
    return paths


def _download_one(api, competition: str, competition_path: str, out_dir: Path, force: bool, retries: int) -> str:
    target = out_dir / competition_path
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0 and not force:
        return "skipped"
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            api.competition_download_file(
                competition,
                competition_path,
                path=str(target.parent),
                force=force,
                quiet=True,
            )
            if target.exists() and target.stat().st_size > 0:
                return "downloaded"
            basename_target = target.parent / Path(competition_path).name
            if basename_target.exists() and basename_target.stat().st_size > 0:
                return "downloaded"
            raise FileNotFoundError(f"Kaggle did not write expected file for {competition_path}")
        except Exception as exc:  # pragma: no cover - network behavior is integration-only
            last_error = exc
            if attempt < retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"Failed to download {competition_path}: {last_error}") from last_error


def _manifest_summary(args: argparse.Namespace, out_dir: Path, plan: list[str], rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "metadata_csv": str(args.metadata_csv),
        "out_dir": str(out_dir),
        "competition": args.competition,
        "dry_run": bool(args.dry_run),
        "n_planned": int(len(plan)),
        "n_completed": int(len(rows)),
        "limit": int(args.limit),
        "balance_columns": list(args.balance_columns),
        "status_counts": pd.Series([row["status"] for row in rows]).value_counts().sort_index().to_dict()
        if rows
        else {},
        "rows": rows[:1000],
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    manifest_out = Path(args.manifest_out) if args.manifest_out else out_dir / "download_manifest.json"
    plan = build_download_plan(
        metadata_csv=Path(args.metadata_csv),
        path_column=args.path_column,
        id_column=args.id_column,
        split=args.split,
        limit=args.limit,
        seed=args.seed,
        balance_columns=list(args.balance_columns),
    )

    rows: list[dict[str, object]] = []
    stopped_early = False
    if args.dry_run:
        rows = [{"competition_path": path, "status": "dry_run"} for path in plan]
    else:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        consecutive_failures = 0
        for index, competition_path in enumerate(plan, start=1):
            try:
                status = _download_one(api, args.competition, competition_path, out_dir, args.force, args.retries)
            except Exception as exc:  # pragma: no cover - network behavior is integration-only
                status = "failed"
                rows.append({"competition_path": competition_path, "status": status, "error": str(exc)})
                print(f"[{index}/{len(plan)}] failed {competition_path}: {exc}")
                consecutive_failures += 1
                if args.stop_after_failures > 0 and consecutive_failures >= args.stop_after_failures:
                    stopped_early = True
                    print(f"Stopping after {consecutive_failures} consecutive failures")
                    write_json(_manifest_summary(args, out_dir, plan, rows), manifest_out)
                    break
            else:
                consecutive_failures = 0
                rows.append({"competition_path": competition_path, "status": status})
                print(f"[{index}/{len(plan)}] {status} {competition_path}")
                if args.sleep_seconds > 0:
                    time.sleep(args.sleep_seconds)
            if args.checkpoint_every > 0 and len(rows) % args.checkpoint_every == 0:
                write_json(_manifest_summary(args, out_dir, plan, rows), manifest_out)

    summary = _manifest_summary(args, out_dir, plan, rows)
    summary["stopped_early"] = bool(stopped_early)
    write_json(summary, manifest_out)
    print(manifest_out.resolve())
    if any(row["status"] == "failed" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
