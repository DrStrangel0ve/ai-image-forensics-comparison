from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


REQUIRED_COLUMNS = ["id", "label"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sorted_id_list_sha256(ids: set[str]) -> str:
    payload = ("\n".join(sorted(ids)) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_score_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != REQUIRED_COLUMNS:
            raise ValueError(f"{path} must contain exactly id,label columns")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} is empty")
    ids = [row["id"] for row in rows]
    if any(not image_id for image_id in ids):
        raise ValueError(f"{path} contains empty ids")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path} contains duplicate ids")
    for row in rows:
        score = float(row["label"])
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"{path} contains an invalid score for id={row['id']}")
    return rows


def merge_private_predictions(
    frozen_submission_path: Path,
    private_predictions_path: Path,
    output_path: Path,
    manifest_path: Path,
    expected_frozen_sha256: str,
    expected_private_id_list_sha256: str,
    expected_private_rows: int,
) -> dict[str, object]:
    frozen_hash = sha256(frozen_submission_path)
    if frozen_hash != expected_frozen_sha256.lower():
        raise ValueError(
            f"Frozen submission hash mismatch: expected={expected_frozen_sha256} actual={frozen_hash}"
        )

    frozen_rows = read_score_rows(frozen_submission_path)
    private_rows = read_score_rows(private_predictions_path)
    if len(private_rows) != expected_private_rows:
        raise ValueError(
            f"Private row count mismatch: expected={expected_private_rows} actual={len(private_rows)}"
        )

    frozen_ids = {row["id"] for row in frozen_rows}
    private_ids = {row["id"] for row in private_rows}
    private_id_hash = sorted_id_list_sha256(private_ids)
    if private_id_hash != expected_private_id_list_sha256.lower():
        raise ValueError(
            "Private id-list hash mismatch: "
            f"expected={expected_private_id_list_sha256} actual={private_id_hash}"
        )
    extra_private_ids = sorted(private_ids - frozen_ids)
    if extra_private_ids:
        raise ValueError(f"Private predictions contain ids outside frozen submission: {extra_private_ids[:5]}")

    private_scores = {row["id"]: row["label"] for row in private_rows}
    merged_rows = [
        {
            "id": row["id"],
            "label": private_scores.get(row["id"], row["label"]),
        }
        for row in frozen_rows
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(merged_rows)

    output_rows = read_score_rows(output_path)
    if [row["id"] for row in output_rows] != [row["id"] for row in frozen_rows]:
        raise RuntimeError("Merged output changed frozen submission id order")
    untouched_ids = frozen_ids - private_ids
    frozen_by_id = {row["id"]: row["label"] for row in frozen_rows}
    output_by_id = {row["id"]: row["label"] for row in output_rows}
    if any(output_by_id[image_id] != frozen_by_id[image_id] for image_id in untouched_ids):
        raise RuntimeError("Merged output changed a non-private frozen score")

    manifest = {
        "status": "pass",
        "frozen_submission_path": str(frozen_submission_path),
        "frozen_submission_sha256": frozen_hash,
        "private_predictions_path": str(private_predictions_path),
        "private_predictions_sha256": sha256(private_predictions_path),
        "private_id_list_sha256": private_id_hash,
        "output_path": str(output_path),
        "output_sha256": sha256(output_path),
        "total_rows": len(output_rows),
        "replaced_private_rows": len(private_rows),
        "preserved_frozen_rows": len(untouched_ids),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay frozen FREUID private scores without changing public-row predictions."
    )
    parser.add_argument("--frozen-submission", required=True)
    parser.add_argument("--private-predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument("--expected-frozen-sha256", required=True)
    parser.add_argument("--expected-private-id-list-sha256", required=True)
    parser.add_argument("--expected-private-rows", required=True, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = merge_private_predictions(
        frozen_submission_path=Path(args.frozen_submission),
        private_predictions_path=Path(args.private_predictions),
        output_path=Path(args.output),
        manifest_path=Path(args.manifest_out),
        expected_frozen_sha256=args.expected_frozen_sha256,
        expected_private_id_list_sha256=args.expected_private_id_list_sha256,
        expected_private_rows=args.expected_private_rows,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
