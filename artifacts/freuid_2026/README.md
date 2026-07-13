# FREUID Frozen Runtime Artifacts

This directory is the Docker/runtime artifact staging area for the current FREUID stack:

- `checkpoints/template_convnext224.pt`
- `checkpoints/forensic_efficientnet384.pt`
- `loto_egypt_clean_fusion_summary.json`
- `loto_egypt_screenshot_fusion_summary.json`
- `freeze_manifest.json`

The first network is the strongest known-layout/public detector. The second is a 384-pixel residual model trained with document-type adversarial learning and capture augmentation. The runtime loads them sequentially, converts each score vector to percentage ranks, and applies the validation-selected conservative `0.85/0.15` blend.

The binary checkpoints are intentionally ignored by Git because they are generated artifacts and the ConvNeXt checkpoint exceeds GitHub's regular file limit. Recreate the exact local artifact set and hash manifest before building the Docker image:

```bash
python scripts/freeze_freuid_submission_artifacts.py
docker build -f docker/freuid/Dockerfile -t freuid-frozen-stack .
```

Expected runtime contract for the challenge container:

```bash
docker run --rm -v /path/to/test_images:/data:ro -v /path/to/submissions:/submissions freuid-frozen-stack
```

The container recursively discovers images under `/data` and writes `/submissions/submission.csv` with `id,label`, where `label` is a rank-fused fraud score in `[0, 1]`. Inference does not require network access.
