from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "merge_freuid_private_predictions.py"
SPEC = importlib.util.spec_from_file_location("merge_freuid_private_predictions", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MERGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MERGE)


def write_scores(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["id", "label"])
        writer.writerows(rows)


def test_merge_replaces_only_private_rows(tmp_path: Path) -> None:
    frozen = tmp_path / "frozen.csv"
    private = tmp_path / "private.csv"
    output = tmp_path / "output.csv"
    manifest_path = tmp_path / "manifest.json"
    write_scores(frozen, [("public", "0.25"), ("private_a", "0.1"), ("private_b", "0.2")])
    write_scores(private, [("private_b", "0.8"), ("private_a", "0.9")])

    manifest = MERGE.merge_private_predictions(
        frozen_submission_path=frozen,
        private_predictions_path=private,
        output_path=output,
        manifest_path=manifest_path,
        expected_frozen_sha256=MERGE.sha256(frozen),
        expected_private_id_list_sha256=MERGE.sorted_id_list_sha256(
            {"private_a", "private_b"}
        ),
        expected_private_rows=2,
    )

    assert MERGE.read_score_rows(output) == [
        {"id": "public", "label": "0.25"},
        {"id": "private_a", "label": "0.9"},
        {"id": "private_b", "label": "0.8"},
    ]
    assert manifest["replaced_private_rows"] == 2
    assert manifest["preserved_frozen_rows"] == 1


def test_merge_rejects_private_ids_outside_frozen_submission(tmp_path: Path) -> None:
    frozen = tmp_path / "frozen.csv"
    private = tmp_path / "private.csv"
    write_scores(frozen, [("public", "0.25")])
    write_scores(private, [("unknown", "0.9")])

    with pytest.raises(ValueError, match="outside frozen submission"):
        MERGE.merge_private_predictions(
            frozen_submission_path=frozen,
            private_predictions_path=private,
            output_path=tmp_path / "output.csv",
            manifest_path=tmp_path / "manifest.json",
            expected_frozen_sha256=MERGE.sha256(frozen),
            expected_private_id_list_sha256=MERGE.sorted_id_list_sha256({"unknown"}),
            expected_private_rows=1,
        )
