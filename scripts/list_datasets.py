from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forensic_compare.catalog import load_catalog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List known real-vs-generated dataset candidates.")
    parser.add_argument("--source", choices=["kaggle", "huggingface", "external"], default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entries = load_catalog()
    print(f"{'key':30} {'source':12} {'updated':12} {'size_mb':>8} title")
    print("-" * 92)
    for entry in entries.values():
        if args.source and entry.source != args.source:
            continue
        size = "" if entry.size_mb is None else str(entry.size_mb)
        print(f"{entry.key:30} {entry.source:12} {entry.updated or '':12} {size:>8} {entry.title}")


if __name__ == "__main__":
    main()
