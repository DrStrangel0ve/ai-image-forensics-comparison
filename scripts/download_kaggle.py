from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and unzip a Kaggle dataset.")
    parser.add_argument(
        "--dataset",
        default="birdy654/cifake-real-and-ai-generated-synthetic-images",
        help="Kaggle dataset reference, e.g. owner/dataset-slug.",
    )
    parser.add_argument("--out", default="data/raw/cifake", help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise SystemExit(
            "Kaggle is not installed. Run `python -m pip install -r requirements.txt` first."
        ) from exc

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(args.dataset, path=out_dir, unzip=True, quiet=False)
    print(f"Downloaded {args.dataset} to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
