from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_freuid_docker_entrypoint_maps_both_selected_variants() -> None:
    entrypoint = (ROOT / "docker" / "freuid" / "entrypoint.sh").read_text(encoding="utf-8")
    dockerfile = (ROOT / "docker" / "freuid" / "Dockerfile").read_text(encoding="utf-8")

    assert "FREUID_VARIANT=ood_rank" in dockerfile
    assert "ood_rank)" in entrypoint
    assert "public_specialist)" in entrypoint
    assert "template_convnext224.pt" in entrypoint
    assert "forensic_efficientnet384.pt" in entrypoint
    assert "--normalization rank" in entrypoint
    assert "--network" not in entrypoint


def test_freuid_runtime_supports_all_organizer_image_extensions() -> None:
    inference = (ROOT / "scripts" / "infer_freuid_finetune.py").read_text(encoding="utf-8")
    for extension in [".jpeg", ".jpg", ".png", ".webp", ".bmp", ".tif", ".tiff"]:
        assert f'"{extension}"' in inference
