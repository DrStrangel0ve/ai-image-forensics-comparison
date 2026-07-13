from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


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
    (tmp_path / "other.png").write_bytes(b"other")

    root, count = KERNEL.find_private_image_root(tmp_path)

    assert root == private_root
    assert count == 3


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


def test_inference_commands_preserve_frozen_variants(tmp_path: Path) -> None:
    commands = KERNEL.inference_commands(
        tmp_path / "repo",
        tmp_path / "images",
        tmp_path / "template_convnext224.pt",
        tmp_path / "forensic_efficientnet384.pt",
        tmp_path / "working",
    )

    public = commands["public_specialist"]
    ood = commands["ood_rank"]
    assert "infer_freuid_finetune.py" in public[1]
    assert "infer_freuid_checkpoint_ensemble.py" in ood[1]
    assert ood[ood.index("--normalization") + 1] == "rank"
    weights = [ood[index + 1] for index, value in enumerate(ood) if value == "--weight"]
    assert weights == ["0.85", "0.15"]
