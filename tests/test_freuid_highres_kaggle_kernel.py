from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
KERNEL_PATH = (
    ROOT
    / "kaggle"
    / "freuid_post_freeze_highres_research"
    / "freuid_post_freeze_highres_research.py"
)
SPEC = importlib.util.spec_from_file_location("freuid_highres_kaggle_kernel", KERNEL_PATH)
assert SPEC is not None and SPEC.loader is not None
KERNEL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(KERNEL)


def make_competition_root(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "train").mkdir()
    (root / "train_labels.csv").write_text(
        "id,image_path,label,is_digital,type\n",
        encoding="utf-8",
    )
    return root


def test_locate_competition_root_prefers_attached_input(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    competition_root = make_competition_root(input_root / KERNEL.COMPETITION_SLUG)

    root, labels, source = KERNEL.locate_competition_root(input_root)

    assert root == competition_root
    assert labels == competition_root / "train_labels.csv"
    assert source == "attached_input"


def test_locate_competition_root_uses_kagglehub_fallback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_root = tmp_path / "empty_input"
    attached_root = make_competition_root(tmp_path / "shared_cache" / "freuid")
    calls: list[tuple[str, str]] = []

    def fake_download(slug: str, *, output_dir: str) -> str:
        calls.append((slug, output_dir))
        return str(attached_root)

    fake_kagglehub = SimpleNamespace(competition_download=fake_download)
    monkeypatch.setitem(sys.modules, "kagglehub", fake_kagglehub)

    root, labels, source = KERNEL.locate_competition_root(input_root)

    assert calls == [
        (KERNEL.COMPETITION_SLUG, "/kaggle/temp/freuid-competition"),
    ]
    assert root == attached_root
    assert labels == attached_root / "train_labels.csv"
    assert source == "kagglehub_http_download"


def test_locate_competition_root_extracts_private_subset(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    dataset_root = input_root / "private-subset"
    dataset_root.mkdir(parents=True)
    archive = dataset_root / KERNEL.SUBSET_ARCHIVE_NAME
    with ZipFile(archive, "w") as handle:
        handle.writestr("train_labels.csv", "id,image_path,label,is_digital,type\n")
        handle.writestr("train/.keep", "")

    root, labels, source = KERNEL.locate_competition_root(input_root, tmp_path / "working")

    assert root == tmp_path / "working" / "freuid-highres-subset"
    assert labels == root / "train_labels.csv"
    assert source == "private_subset_archive"


def test_private_subset_rejects_zip_traversal(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    dataset_root = input_root / "private-subset"
    dataset_root.mkdir(parents=True)
    archive = dataset_root / KERNEL.SUBSET_ARCHIVE_NAME
    with ZipFile(archive, "w") as handle:
        handle.writestr("../outside.txt", "unsafe")

    try:
        KERNEL.locate_competition_root(input_root, tmp_path / "working")
    except ValueError as exc:
        assert "Unsafe path" in str(exc)
    else:
        raise AssertionError("Expected unsafe archive path to be rejected")
