# FREUID Frozen Runtime Artifacts

This directory is the Docker/runtime artifact staging area for the current FREUID stack:

- `combined_v4_hgb.joblib`
- `convnext_tiny_logreg.joblib`
- `fusion_summary.json`
- `torch/hub/checkpoints/convnext_tiny-983f1562.pth`

The binary model/checkpoint files are intentionally ignored by Git because they are generated artifacts and the ConvNeXt checkpoint is large. Recreate the local artifact set before building the Docker image:

```bash
python scripts/freeze_freuid_submission_artifacts.py
docker build -f docker/freuid/Dockerfile -t freuid-frozen-stack .
```

Expected runtime contract for the challenge container:

```bash
docker run --rm -v /path/to/test_images:/data:ro -v /path/to/submissions:/submissions freuid-frozen-stack
```

The container writes `/submissions/submission.csv` with `id,label`, where `label` is a fraud score in `[0, 1]`.
