from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forensic_compare.catalog import load_catalog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a dataset from configs/datasets.json.")
    parser.add_argument("key", help="Dataset key from `python scripts/list_datasets.py`.")
    parser.add_argument("--out", default=None, help="Override output directory.")
    return parser.parse_args()


def _download_kaggle(ref: str, out_dir: Path) -> None:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise SystemExit("Install Kaggle first: python -m pip install kaggle") from exc
    out_dir.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(ref, path=out_dir, unzip=True, quiet=False)


def _download_huggingface(ref: str, out_dir: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "Install Hugging Face Hub support first: python -m pip install huggingface_hub"
        ) from exc
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=ref, repo_type="dataset", local_dir=out_dir)


def main() -> None:
    args = parse_args()
    catalog = load_catalog()
    if args.key not in catalog:
        choices = ", ".join(sorted(catalog))
        raise SystemExit(f"Unknown dataset key {args.key!r}. Choices: {choices}")
    entry = catalog[args.key]
    out_dir = Path(args.out) if args.out else entry.local_dir
    if entry.source == "kaggle":
        _download_kaggle(entry.ref, out_dir)
    elif entry.source == "huggingface":
        _download_huggingface(entry.ref, out_dir)
    else:
        raise SystemExit(
            f"{entry.key} is an external source. Follow this URL/instructions manually: {entry.ref}"
        )
    print(f"Downloaded {entry.title} to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
