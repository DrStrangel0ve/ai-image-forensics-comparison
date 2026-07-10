from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy the current FREUID frozen-stack runtime assets into artifacts/freuid_2026."
    )
    parser.add_argument(
        "--combined-v4-model",
        default="runs/freuid_public12k_seed37_combined_v4_hgb/classifier.joblib",
    )
    parser.add_argument(
        "--convnext-model",
        default="runs/freuid_public12k_seed37_type_label38_convnext_tiny_logreg/classifier.joblib",
    )
    parser.add_argument(
        "--convnext-checkpoint",
        default=str(Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "convnext_tiny-983f1562.pth"),
    )
    parser.add_argument(
        "--fusion-summary",
        default="runs/freuid_public12k_seed37_fusion_v3_convnext_photometric_v4/fusion_summary.json",
    )
    parser.add_argument("--output-dir", default="artifacts/freuid_2026")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy(source: Path, destination: Path) -> dict[str, object]:
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {
        "path": str(destination),
        "source_name": source.name,
        "bytes": int(destination.stat().st_size),
        "sha256": _sha256(destination),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    files = [
        _copy(Path(args.combined_v4_model), output_dir / "combined_v4_hgb.joblib"),
        _copy(Path(args.convnext_model), output_dir / "convnext_tiny_logreg.joblib"),
        _copy(
            Path(args.convnext_checkpoint),
            output_dir / "torch" / "hub" / "checkpoints" / "convnext_tiny-983f1562.pth",
        ),
        _copy(Path(args.fusion_summary), output_dir / "fusion_summary.json"),
    ]
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stack": "0.7*combined_v4_hgb + 0.3*convnext_tiny_logreg",
        "best_kaggle_submission_ref": "54511333",
        "best_public_score": 0.37009,
        "files": files,
    }
    manifest_path = output_dir / "freeze_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(manifest_path.resolve())


if __name__ == "__main__":
    main()
