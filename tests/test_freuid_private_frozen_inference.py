from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
KERNEL_PATH = (
    ROOT
    / "kaggle"
    / "freuid_private_frozen_inference"
    / "freuid_private_frozen_inference.py"
)
SPEC = importlib.util.spec_from_file_location("freuid_private_frozen_inference", KERNEL_PATH)
assert SPEC is not None and SPEC.loader is not None
KERNEL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(KERNEL)


def test_find_private_image_root_prefers_populated_leaf(tmp_path: Path) -> None:
    private_root = tmp_path / "dataset" / "private_test" / "private_test"
    private_root.mkdir(parents=True)
    for image_id in ("a", "b", "c"):
        (private_root / f"{image_id}.jpeg").write_bytes(b"image")
    train_root = tmp_path / "dataset" / "train"
    train_root.mkdir()
    for index in range(10):
        (train_root / f"train_{index}.png").write_bytes(b"other")

    root, count = KERNEL.find_private_image_root(tmp_path)

    assert root == private_root
    assert count == 3


def test_find_private_image_root_rejects_non_private_images(tmp_path: Path) -> None:
    train_root = tmp_path / "dataset" / "train"
    train_root.mkdir(parents=True)
    (train_root / "train.png").write_bytes(b"other")

    try:
        KERNEL.find_private_image_root(tmp_path)
    except FileNotFoundError as exc:
        assert "private-test" in str(exc)
    else:
        raise AssertionError("Expected private-test discovery to fail")


def test_validate_submission_checks_ids_and_probabilities(tmp_path: Path) -> None:
    path = tmp_path / "submission.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "label"])
        writer.writeheader()
        writer.writerow({"id": "a", "label": "0.1"})
        writer.writerow({"id": "b", "label": "0.9"})

    result = KERNEL.validate_submission(path, {"a", "b"})

    assert result["rows"] == 2
    assert result["unique_scores"] == 2
    assert len(str(result["sha256"])) == 64


def test_member_commands_preserve_frozen_checkpoints(tmp_path: Path) -> None:
    commands = KERNEL.member_commands(
        tmp_path / "repo",
        tmp_path / "images",
        tmp_path / "template_convnext224.pt",
        tmp_path / "forensic_efficientnet384.pt",
        tmp_path / "working",
    )

    public = commands["public_member"]
    forensic = commands["forensic_member"]
    assert public[public.index("--checkpoint") + 1].endswith("template_convnext224.pt")
    assert forensic[forensic.index("--checkpoint") + 1].endswith(
        "forensic_efficientnet384.pt"
    )


def test_fuse_ranked_scores_preserves_frozen_weights() -> None:
    public = np.asarray([0.1, 0.5, 0.9])
    forensic = np.asarray([0.9, 0.5, 0.1])

    def ranker(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values)
        ranks = np.empty(values.size, dtype=float)
        ranks[order] = np.arange(1, values.size + 1) / values.size
        return ranks

    fused = KERNEL.fuse_ranked_scores(public, forensic, ranker)

    expected = 0.85 * np.asarray([1 / 3, 2 / 3, 1.0]) + 0.15 * np.asarray(
        [1.0, 2 / 3, 1 / 3]
    )
    np.testing.assert_allclose(fused, expected)


def test_execution_batches_run_in_parallel_with_two_gpus() -> None:
    batches = KERNEL.execution_batches(["public_specialist", "ood_rank"], 2)

    assert batches == [[("public_specialist", 0), ("ood_rank", 1)]]


def test_execution_batches_run_sequentially_with_one_gpu() -> None:
    batches = KERNEL.execution_batches(["public_specialist", "ood_rank"], 1)

    assert batches == [[("public_specialist", 0)], [("ood_rank", 0)]]
