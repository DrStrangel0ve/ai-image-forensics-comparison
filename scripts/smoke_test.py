from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a tiny dataset and run both baselines.")
    parser.add_argument("--data-dir", default="data/smoke")
    parser.add_argument("--runs-dir", default="runs/smoke")
    return parser.parse_args()


def _save(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(array * 255, 0, 255).astype(np.uint8)).save(path)


def make_smoke_dataset(root: Path, n_per_split_class: int = 12, size: int = 64) -> None:
    rng = np.random.default_rng(42)
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    x = x / max(1, size - 1)
    y = y / max(1, size - 1)
    for split in ["train", "test"]:
        for class_name in ["REAL", "FAKE"]:
            for idx in range(n_per_split_class):
                if class_name == "REAL":
                    light = 0.35 + 0.55 * (1.0 - np.sqrt((x - 0.25) ** 2 + (y - 0.2) ** 2))
                    color = np.stack([light * 0.9, light, light * 0.82], axis=2)
                    image = color + rng.normal(0, 0.015, size=(size, size, 3))
                else:
                    checker = ((np.floor(x * 8 + idx) + np.floor(y * 8)) % 2).astype(np.float32)
                    bands = 0.5 + 0.35 * np.sin((x * 19 + y * 7 + idx) * np.pi)
                    image = np.stack([checker, bands, 1.0 - checker * 0.7], axis=2)
                    image += rng.normal(0, 0.05, size=(size, size, 3))
                _save(root / split / class_name / f"{idx:03d}.png", image)


def _run(command: list[str]) -> None:
    print(" ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()
    data_dir = ROOT / args.data_dir
    runs_dir = ROOT / args.runs_dir
    make_smoke_dataset(data_dir)
    _run(
        [
            sys.executable,
            "scripts/run_photometric_baseline.py",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(runs_dir / "photometric"),
            "--image-size",
            "64",
        ]
    )
    _run(
        [
            sys.executable,
            "scripts/train_neural_net.py",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(runs_dir / "tiny_cnn"),
            "--model",
            "tiny_cnn",
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--image-size",
            "64",
            "--num-workers",
            "0",
        ]
    )
    _run(
        [
            sys.executable,
            "scripts/compare_methods.py",
            "--neural",
            str(runs_dir / "tiny_cnn" / "metrics.json"),
            "--photometric",
            str(runs_dir / "photometric" / "metrics.json"),
            "--out-dir",
            str(runs_dir / "comparison"),
        ]
    )


if __name__ == "__main__":
    main()
