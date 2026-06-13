from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from run_reverse_branch_robustness import build_tasks, command_for_task, run_tasks  # noqa: E402


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        seeds=[7],
        variants=["blur1"],
        methods=["combined_v3", "resnet18"],
        target_template=str(tmp_path / "seed{seed}" / "{variant}"),
        output_template=str(tmp_path / "runs" / "{variant}" / "seed{seed}"),
        neural_root=str(tmp_path / "models" / "neural"),
        foundation_root=str(tmp_path / "models" / "foundation"),
        python=sys.executable,
        device="cpu",
        batch_size=8,
        num_workers=0,
        overwrite=False,
        dry_run=True,
    )


def test_build_tasks_expands_seed_variant_and_methods(tmp_path: Path) -> None:
    args = _args(tmp_path)

    tasks = build_tasks(args)

    assert [task.method.name for task in tasks] == ["combined_v3", "resnet18"]
    assert tasks[0].target_dir == tmp_path / "seed7" / "blur1"
    assert tasks[0].model_dir == tmp_path / "models" / "neural" / "combined_v3_seed7"
    assert tasks[0].output_dir == tmp_path / "runs" / "blur1" / "seed7" / "combined_v3"


def test_command_for_task_uses_method_specific_flags(tmp_path: Path) -> None:
    args = _args(tmp_path)
    feature_task, neural_task = build_tasks(args)

    feature_command = command_for_task(feature_task, args)
    neural_command = command_for_task(neural_task, args)

    assert "evaluate_feature_model.py" in " ".join(feature_command)
    assert "--skip-errors" in feature_command
    assert "evaluate_neural_net.py" in " ".join(neural_command)
    assert "--device" in neural_command
    assert "--batch-size" in neural_command


def test_dry_run_validates_inputs_and_skips_subprocess(tmp_path: Path) -> None:
    args = _args(tmp_path)
    (tmp_path / "seed7" / "blur1").mkdir(parents=True)
    for model_name in ["combined_v3_seed7", "resnet18_seed7"]:
        model_dir = tmp_path / "models" / "neural" / model_name
        model_dir.mkdir(parents=True)
        (model_dir / "metrics.json").write_text("{}", encoding="utf-8")

    rows = run_tasks(args)

    assert [row["status"] for row in rows] == ["dry_run", "dry_run"]
    assert all(row["variant"] == "blur1" for row in rows)
