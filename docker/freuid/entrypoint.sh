#!/bin/sh
set -eu

BATCH_SIZE="${FREUID_BATCH_SIZE:-96}"
NUM_WORKERS="${FREUID_NUM_WORKERS:-12}"

case "${FREUID_VARIANT:-ood_rank}" in
  ood_rank)
    exec python /app/scripts/infer_freuid_checkpoint_ensemble.py \
      --input-dir /data \
      --recursive \
      --output-csv /submissions/submission.csv \
      --checkpoint /opt/freuid_artifacts/checkpoints/template_convnext224.pt \
      --checkpoint /opt/freuid_artifacts/checkpoints/forensic_efficientnet384.pt \
      --weight 0.85 \
      --weight 0.15 \
      --normalization rank \
      --batch-size "${BATCH_SIZE}" \
      --num-workers "${NUM_WORKERS}" \
      --device auto
    ;;
  public_specialist)
    exec python /app/scripts/infer_freuid_finetune.py \
      --input-dir /data \
      --recursive \
      --output-csv /submissions/submission.csv \
      --checkpoint /opt/freuid_artifacts/checkpoints/template_convnext224.pt \
      --batch-size "${BATCH_SIZE}" \
      --num-workers "${NUM_WORKERS}" \
      --device auto
    ;;
  *)
    echo "Unknown FREUID_VARIANT: ${FREUID_VARIANT}. Expected ood_rank or public_specialist." >&2
    exit 64
    ;;
esac
