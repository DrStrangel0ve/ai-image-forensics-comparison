from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESERVED_FORWARDED_ARGS = {"--dataset-key", "--data-dir", "--out-dir", "--seed"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the benchmark wrapper over multiple deterministic split seeds."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset-key", default=None)
    source.add_argument("--data-dir", default=None)
    parser.add_argument("--out-dir", required=True, help="Parent directory for seed-specific runs.")
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument(
        "--summary-dir",
        default=None,
        help="Summary output directory. Defaults to OUT_DIR/summary.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not rerun seeds that already have comparison/comparison.csv.",
    )
    parser.add_argument(
        "benchmark_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to scripts/run_benchmark.py after a literal -- separator.",
    )
    return parser.parse_args()


def _clean_forwarded_args(values: list[str]) -> list[str]:
    if values and values[0] == "--":
        values = values[1:]
    option_names = {value.split("=", 1)[0] for value in values if value.startswith("--")}
    reserved = sorted(RESERVED_FORWARDED_ARGS & option_names)
    if reserved:
        raise SystemExit(
            "Do not pass these through the forwarded argument section: "
            + ", ".join(reserved)
        )
    return values


def _run(command: list[str]) -> None:
    print(" ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()
    benchmark_args = _clean_forwarded_args(args.benchmark_args)
    out_dir = Path(args.out_dir)
    summary_dir = Path(args.summary_dir) if args.summary_dir else out_dir / "summary"
    comparisons: list[tuple[str, Path]] = []

    for seed in args.seeds:
        run_name = f"seed{seed}"
        seed_out_dir = out_dir / run_name
        comparison_csv = seed_out_dir / "comparison" / "comparison.csv"
        if args.skip_existing and comparison_csv.exists():
            print(f"Skipping existing seed {seed}: {comparison_csv}")
        else:
            command = [
                sys.executable,
                "scripts/run_benchmark.py",
                "--out-dir",
                str(seed_out_dir),
                "--seed",
                str(seed),
            ]
            if args.dataset_key:
                command.extend(["--dataset-key", args.dataset_key])
            else:
                command.extend(["--data-dir", str(args.data_dir)])
            command.extend(benchmark_args)
            _run(command)
        comparisons.append((run_name, comparison_csv))

    summary_command = [
        sys.executable,
        "scripts/summarize_repeated_benchmarks.py",
        "--out-dir",
        str(summary_dir),
    ]
    for run_name, comparison_csv in comparisons:
        summary_command.extend(["--comparison", f"{run_name}={comparison_csv}"])
    _run(summary_command)


if __name__ == "__main__":
    main()
