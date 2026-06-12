from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.summarize_feature_ablation import summarize


DEFAULT_CONFIGS = [
    "combined_v3_logreg=combined_v3:logistic_regression:0",
    "combined_v4_logreg=combined_v4:logistic_regression:0",
    "combined_v4_logreg_selectk60=combined_v4:logistic_regression:60",
    "combined_v4_logreg_selectk80=combined_v4:logistic_regression:80",
]


@dataclass(frozen=True)
class FeatureConfig:
    name: str
    feature_set: str
    classifier: str
    select_k: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a grid of conventional feature ablations over repeated seeds."
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help=(
            "Config in NAME=FEATURE_SET:CLASSIFIER:SELECT_K form. "
            "Defaults compare combined_v3, raw combined_v4, and select-k v4."
        ),
    )
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--selection-score-func", choices=["f_classif", "mutual_info"], default="f_classif")
    parser.add_argument("--summary-dir", default=None)
    parser.add_argument("--extra-feature-base", default="combined_v3")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-errors", action="store_true")
    return parser.parse_args()


def _parse_config(value: str) -> FeatureConfig:
    if "=" not in value:
        raise ValueError(f"Config must be NAME=FEATURE_SET:CLASSIFIER:SELECT_K, got {value!r}")
    name, spec = value.split("=", 1)
    parts = spec.split(":")
    if len(parts) != 3 or not name or any(not part for part in parts):
        raise ValueError(f"Config must be NAME=FEATURE_SET:CLASSIFIER:SELECT_K, got {value!r}")
    feature_set, classifier, select_k_text = parts
    try:
        select_k = int(select_k_text)
    except ValueError as exc:
        raise ValueError(f"SELECT_K must be an integer in config {value!r}") from exc
    if select_k < 0:
        raise ValueError(f"SELECT_K must be non-negative in config {value!r}")
    return FeatureConfig(name=name, feature_set=feature_set, classifier=classifier, select_k=select_k)


def _configs(values: list[str]) -> list[FeatureConfig]:
    return [_parse_config(value) for value in (values or DEFAULT_CONFIGS)]


def _command(args: argparse.Namespace, config: FeatureConfig, seed: int, output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        "scripts/run_feature_baseline.py",
        "--data-dir",
        args.data_dir,
        "--output-dir",
        str(output_dir),
        "--feature-set",
        config.feature_set,
        "--classifier",
        config.classifier,
        "--select-k",
        str(config.select_k),
        "--selection-score-func",
        args.selection_score_func,
        "--image-size",
        str(args.image_size),
        "--seed",
        str(seed),
        "--val-fraction",
        str(args.val_fraction),
        "--max-train-samples",
        str(args.max_train_samples),
        "--max-test-samples",
        str(args.max_test_samples),
    ]
    if args.skip_errors:
        command.append("--skip-errors")
    return command


def _run(command: list[str]) -> None:
    print(" ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def run_grid(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    for seed in args.seeds:
        for config in _configs(args.config):
            output_dir = out_dir / f"seed{seed}" / config.name
            if args.skip_existing and (output_dir / "metrics.json").exists():
                print(f"Skipping existing {output_dir}")
                continue
            _run(_command(args, config, seed, output_dir))

    summary_dir = Path(args.summary_dir) if args.summary_dir else out_dir / "summary"
    summarize(out_dir, summary_dir, args.extra_feature_base)


def main() -> None:
    run_grid(parse_args())


if __name__ == "__main__":
    main()
