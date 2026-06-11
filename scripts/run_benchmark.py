from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forensic_compare.catalog import load_catalog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run neural and conventional baselines on one dataset.")
    parser.add_argument("--dataset-key", default=None, help="Dataset key from configs/datasets.json.")
    parser.add_argument("--data-dir", default=None, help="Dataset folder. Overrides --dataset-key local_dir.")
    parser.add_argument("--out-dir", default=None, help="Output run directory.")
    parser.add_argument("--methods", nargs="+", default=["photometric", "noise", "combined", "neural"])
    parser.add_argument("--feature-classifier", default="logistic_regression")
    parser.add_argument("--feature-image-size", type=int, default=128)
    parser.add_argument("--neural-model", default="resnet18")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained neural weights when available.")
    parser.add_argument("--neural-image-size", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--skip-errors", action="store_true")
    return parser.parse_args()


def _run(command: list[str]) -> None:
    print(" ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()
    catalog = load_catalog()
    if args.data_dir:
        data_dir = Path(args.data_dir)
        run_name = data_dir.name
    elif args.dataset_key:
        entry = catalog[args.dataset_key]
        data_dir = entry.local_dir
        run_name = args.dataset_key
    else:
        raise SystemExit("Provide --dataset-key or --data-dir")
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "runs" / run_name
    metrics: list[tuple[str, Path]] = []

    for method in args.methods:
        if method in {"photometric", "noise", "noise_v2", "combined", "combined_v2"}:
            method_out = out_dir / f"feature_{method}_{args.feature_classifier}"
            command = [
                sys.executable,
                "scripts/run_feature_baseline.py",
                "--data-dir",
                str(data_dir),
                "--output-dir",
                str(method_out),
                "--feature-set",
                method,
                "--classifier",
                args.feature_classifier,
                "--image-size",
                str(args.feature_image_size),
                "--max-train-samples",
                str(args.max_train_samples),
                "--max-test-samples",
                str(args.max_test_samples),
            ]
            if args.skip_errors:
                command.append("--skip-errors")
            _run(command)
            metrics.append((f"feature_{method}", method_out / "metrics.json"))
        elif method == "neural":
            method_out = out_dir / args.neural_model
            command = [
                sys.executable,
                "scripts/train_neural_net.py",
                "--data-dir",
                str(data_dir),
                "--output-dir",
                str(method_out),
                "--model",
                args.neural_model,
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--image-size",
                str(args.neural_image_size),
                "--num-workers",
                str(args.num_workers),
                "--device",
                args.device,
                "--max-train-samples",
                str(args.max_train_samples),
                "--max-test-samples",
                str(args.max_test_samples),
            ]
            if args.pretrained:
                command.append("--pretrained")
            _run(command)
            metrics.append((f"neural_{args.neural_model}", method_out / "metrics.json"))
        else:
            raise ValueError(f"Unsupported method: {method}")

    compare_command = [sys.executable, "scripts/compare_methods.py", "--out-dir", str(out_dir / "comparison")]
    for name, path in metrics:
        compare_command.extend(["--metrics", f"{name}={path}"])
    _run(compare_command)


if __name__ == "__main__":
    main()
