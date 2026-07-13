# FREUID Frozen Runtime Artifacts

This directory is the Docker/runtime artifact staging area for the current FREUID stack:

- `checkpoints/template_convnext224.pt`
- `checkpoints/forensic_efficientnet384.pt`
- `loto_egypt_clean_fusion_summary.json`
- `loto_egypt_screenshot_fusion_summary.json`
- `freeze_manifest.json`

The first network is the strongest known-layout/public detector. The second is a 384-pixel residual model trained with document-type adversarial learning and capture augmentation. The default runtime loads them sequentially, converts each score vector to percentage ranks, and applies the validation-selected conservative `0.85/0.15` blend.

The binary checkpoints are intentionally ignored by Git because they are generated artifacts and the ConvNeXt checkpoint exceeds GitHub's regular file limit. Recreate the exact local artifact set and hash manifest before building the Docker image:

```bash
python scripts/freeze_freuid_submission_artifacts.py
docker build -f docker/freuid/Dockerfile -t freuid-frozen-stack .
```

## Final Submission Variants

Both selected Kaggle submissions use the same frozen checkpoints and preprocessing. `FREUID_VARIANT` only changes inference orchestration:

| Variant | Kaggle ref | Timestamp (UTC) | Submitted CSV SHA-256 |
| --- | --- | --- | --- |
| `public_specialist` | `54624136` | `2026-07-12 22:49:39` | `35454097181d7430ea0e322e5c3dd8a73b2db3519a5215fc472b7c105000bae0` |
| `ood_rank` | `54627101` | `2026-07-13 00:50:51` | `cbc3e6c0fbb0bbd9d35e7f4e1d33fc21835afa3f5f0b6f33b17824393fcf700d` |

Build once, then run either selected pick without network access:

```bash
docker build -f docker/freuid/Dockerfile -t freuid-frozen-stack .

docker run --rm --network none \
  -e FREUID_VARIANT=ood_rank \
  -v /path/to/test_images:/data:ro \
  -v /path/to/submissions:/submissions \
  freuid-frozen-stack

docker run --rm --network none \
  -e FREUID_VARIANT=public_specialist \
  -v /path/to/test_images:/data:ro \
  -v /path/to/submissions:/submissions \
  freuid-frozen-stack
```

The container discovers `.jpeg`, `.jpg`, `.png`, `.webp`, `.bmp`, `.tif`, and `.tiff` images under `/data` and writes `/submissions/submission.csv` with `id,label`. The default variant is `ood_rank`. The organizer runtime limit is six hours on one A100 40GB with 24 CPUs.

The A100-oriented defaults are batch size `96` and `12` data-loader workers. They can be reduced for local hardware with `FREUID_BATCH_SIZE` and `FREUID_NUM_WORKERS`; these controls do not change checkpoints or preprocessing. The RTX 3080 Ti benchmark uses batch size `32` and four workers.

## Runtime Check

The exact frozen OOD path scored `1,000` images in `211.4 s` on the development RTX 3080 Ti 12GB at batch size `32` with four workers (`4.73 images/s` wall clock). This small run includes model deserialization, CUDA initialization, and Windows worker startup twice. A naive linear projection is `8.39 h` on the slower local GPU; the organizer target is an A100 40GB with 24 CPUs, for which the container uses a larger batch and more workers. Final six-hour compliance must still be confirmed in the organizer sandbox.
