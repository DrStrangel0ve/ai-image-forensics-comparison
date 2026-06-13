from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class MethodConfig:
    name: str
    script: str
    model_dir_template: str
    output_name: str
    extra_args: tuple[str, ...] = ()


METHODS = {
    "combined_v3": MethodConfig(
        name="combined_v3",
        script="evaluate_feature_model.py",
        model_dir_template="{neural_root}/combined_v3_seed{seed}",
        output_name="combined_v3",
        extra_args=("--skip-errors",),
    ),
    "resnet18": MethodConfig(
        name="resnet18",
        script="evaluate_neural_net.py",
        model_dir_template="{neural_root}/resnet18_seed{seed}",
        output_name="resnet18",
    ),
    "physics_guided": MethodConfig(
        name="physics_guided",
        script="evaluate_physics_guided_net.py",
        model_dir_template="{neural_root}/physics_guided_resnet18_combined_v3_seed{seed}",
        output_name="physics_guided",
        extra_args=("--skip-errors",),
    ),
    "convnext_tiny": MethodConfig(
        name="convnext_tiny",
        script="evaluate_frozen_encoder_model.py",
        model_dir_template="{foundation_root}/convnext_tiny_seed{seed}",
        output_name="convnext_tiny",
    ),
    "clip_vit_b_32": MethodConfig(
        name="clip_vit_b_32",
        script="evaluate_frozen_encoder_model.py",
        model_dir_template="{foundation_root}/clip_vit_b_32_seed{seed}",
        output_name="clip_vit_b_32",
    ),
    "dinov2_vits14": MethodConfig(
        name="dinov2_vits14",
        script="evaluate_frozen_encoder_model.py",
        model_dir_template="{foundation_root}/dinov2_vits14_seed{seed}",
        output_name="dinov2_vits14",
    ),
}


@dataclass(frozen=True)
class EvalTask:
    variant: str
    seed: int
    method: MethodConfig
    target_dir: Path
    model_dir: Path
    output_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate saved MS-COCOAI-trained branch models on seed-specific transformed Ishu targets."
        )
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 29])
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["jpeg70", "blur1", "resize_half", "crop85"],
    )
    parser.add_argument("--methods", nargs="+", choices=sorted(METHODS), default=list(METHODS))
    parser.add_argument(
        "--target-template",
        default="data/raw/ishu_ai_vs_real_2026_seed{seed}_test_robustness_variants/{variant}",
    )
    parser.add_argument(
        "--output-template",
        default="runs/ms_cocoai_to_ishu_{variant}_robustness/seed{seed}",
    )
    parser.add_argument(
        "--neural-root",
        default="runs/ms_cocoai_to_ishu_neural_fusion",
    )
    parser.add_argument(
        "--foundation-root",
        default="runs/ms_cocoai_to_ishu_foundation",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _format_path(template: str, args: argparse.Namespace, seed: int, variant: str) -> Path:
    return Path(
        template.format(
            seed=seed,
            variant=variant,
            neural_root=args.neural_root,
            foundation_root=args.foundation_root,
        )
    )


def build_tasks(args: argparse.Namespace) -> list[EvalTask]:
    tasks = []
    for variant in args.variants:
        for seed in args.seeds:
            target_dir = _format_path(args.target_template, args, seed, variant)
            output_root = _format_path(args.output_template, args, seed, variant)
            for method_name in args.methods:
                method = METHODS[method_name]
                model_dir = _format_path(method.model_dir_template, args, seed, variant)
                tasks.append(
                    EvalTask(
                        variant=variant,
                        seed=seed,
                        method=method,
                        target_dir=target_dir,
                        model_dir=model_dir,
                        output_dir=output_root / method.output_name,
                    )
                )
    return tasks


def command_for_task(task: EvalTask, args: argparse.Namespace) -> list[str]:
    command = [
        args.python,
        str(ROOT / "scripts" / task.method.script),
        "--model-dir",
        str(task.model_dir),
        "--target-dir",
        str(task.target_dir),
        "--output-dir",
        str(task.output_dir),
        "--target-split",
        "all",
    ]
    if task.method.script in {"evaluate_neural_net.py", "evaluate_physics_guided_net.py"}:
        command.extend(
            [
                "--device",
                args.device,
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
            ]
        )
    if task.method.script == "evaluate_frozen_encoder_model.py":
        command.extend(
            [
                "--device",
                args.device,
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
            ]
        )
    command.extend(task.method.extra_args)
    return command


def _manifest_path(args: argparse.Namespace, variant: str) -> Path:
    root = _format_path(args.output_template, args, args.seeds[0], variant).parent
    return root / "manifest.csv"


def _validate_task(task: EvalTask) -> None:
    if not task.target_dir.exists():
        raise FileNotFoundError(f"Missing target dir: {task.target_dir}")
    if not (task.model_dir / "metrics.json").exists():
        raise FileNotFoundError(f"Missing model metrics: {task.model_dir / 'metrics.json'}")


def run_tasks(args: argparse.Namespace) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for task in build_tasks(args):
        _validate_task(task)
        predictions = task.output_dir / "predictions.csv"
        command = command_for_task(task, args)
        status = "dry_run"
        if predictions.exists() and not args.overwrite:
            status = "skipped_existing"
        elif not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)
            status = "ran"
        rows.append(
            {
                "variant": task.variant,
                "seed": task.seed,
                "method": task.method.name,
                "model_dir": str(task.model_dir),
                "target_dir": str(task.target_dir),
                "output_dir": str(task.output_dir),
                "status": status,
                "command": " ".join(command),
            }
        )
    return rows


def write_manifests(args: argparse.Namespace, rows: list[dict[str, str | int]]) -> None:
    by_variant: dict[str, list[dict[str, str | int]]] = {}
    for row in rows:
        by_variant.setdefault(str(row["variant"]), []).append(row)
    for variant, variant_rows in by_variant.items():
        path = _manifest_path(args, variant)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "variant",
                    "seed",
                    "method",
                    "model_dir",
                    "target_dir",
                    "output_dir",
                    "status",
                    "command",
                ],
            )
            writer.writeheader()
            writer.writerows(variant_rows)
        print(path.resolve())


def main() -> None:
    args = parse_args()
    rows = run_tasks(args)
    write_manifests(args, rows)


if __name__ == "__main__":
    main()
