from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a leave-one-document-type-out FREUID split.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--holdout-type", required=True)
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--val-out", required=True)
    return parser.parse_args()


def build_split(
    input_path: Path,
    holdout_type: str,
    train_out: Path,
    val_out: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(input_path)
    required = {"id", "label", "type"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")
    available = sorted(frame["type"].astype(str).unique().tolist())
    if holdout_type not in available:
        raise ValueError(f"Unknown holdout type {holdout_type!r}; available types: {available}")
    val = frame[frame["type"].astype(str) == holdout_type].sort_values("id").reset_index(drop=True)
    train = frame[frame["type"].astype(str) != holdout_type].sort_values("id").reset_index(drop=True)
    if train["label"].nunique() != 2 or val["label"].nunique() != 2:
        raise ValueError("Both train and validation splits must contain both labels")
    train_out.parent.mkdir(parents=True, exist_ok=True)
    val_out.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(train_out, index=False)
    val.to_csv(val_out, index=False)
    return train, val


def main() -> None:
    args = parse_args()
    train, val = build_split(
        Path(args.input_csv),
        args.holdout_type,
        Path(args.train_out),
        Path(args.val_out),
    )
    print(Path(args.train_out).resolve())
    print(Path(args.val_out).resolve())
    print(f"n_train={len(train)} n_val={len(val)} holdout_type={args.holdout_type}")


if __name__ == "__main__":
    main()
