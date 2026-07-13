from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy the current FREUID checkpoint ensemble into artifacts/freuid_2026."
    )
    parser.add_argument(
        "--template-checkpoint",
        default="runs/freuid_v2_convnext_full_seed43/model.pt",
    )
    parser.add_argument(
        "--forensic-checkpoint",
        default="runs/freuid_forensic384_alltype_16k_seed71/model.pt",
    )
    parser.add_argument(
        "--fusion-summary",
        default="runs/freuid_loto_egypt_joint_fusion_clean/fusion_summary.json",
    )
    parser.add_argument(
        "--capture-fusion-summary",
        default="runs/freuid_loto_egypt_joint_fusion_screenshot/fusion_summary.json",
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
    for legacy_path in [
        output_dir / "combined_v4_hgb.joblib",
        output_dir / "convnext_tiny_logreg.joblib",
        output_dir / "torch" / "hub" / "checkpoints" / "convnext_tiny-983f1562.pth",
    ]:
        legacy_path.unlink(missing_ok=True)
    files = [
        _copy(
            Path(args.template_checkpoint),
            output_dir / "checkpoints" / "template_convnext224.pt",
        ),
        _copy(
            Path(args.forensic_checkpoint),
            output_dir / "checkpoints" / "forensic_efficientnet384.pt",
        ),
        _copy(Path(args.fusion_summary), output_dir / "loto_egypt_clean_fusion_summary.json"),
        _copy(
            Path(args.capture_fusion_summary),
            output_dir / "loto_egypt_screenshot_fusion_summary.json",
        ),
    ]
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime": "sequential_checkpoint_ensemble",
        "stack": "rank(0.85*template_convnext224 + 0.15*forensic_efficientnet384)",
        "best_public_submission_ref": "54624136",
        "best_public_score": 0.25470,
        "ood_candidate_submission_ref": "54626233",
        "ood_candidate_public_score": 0.27166,
        "selection_basis": {
            "protocol": "leave-EGYPT-document-type-out with paired capture transforms",
            "clean_global_audet": 0.171010,
            "clean_ensemble_audet": 0.167956,
            "screenshot_global_audet": 0.171834,
            "screenshot_ensemble_audet": 0.164086,
        },
        "files": files,
    }
    manifest_path = output_dir / "freeze_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(manifest_path.resolve())


if __name__ == "__main__":
    main()
