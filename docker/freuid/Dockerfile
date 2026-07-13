FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    TORCH_HOME=/opt/freuid_artifacts/torch

WORKDIR /app

RUN python -m pip install --no-cache-dir --upgrade pip

COPY docker/freuid/requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt

COPY src /app/src
COPY scripts/infer_freuid_finetune.py /app/scripts/infer_freuid_finetune.py
COPY scripts/infer_freuid_checkpoint_ensemble.py /app/scripts/infer_freuid_checkpoint_ensemble.py
COPY artifacts/freuid_2026 /opt/freuid_artifacts

RUN mkdir -p /submissions

ENTRYPOINT ["python", "/app/scripts/infer_freuid_checkpoint_ensemble.py", "--input-dir", "/data", "--recursive", "--output-csv", "/submissions/submission.csv", "--checkpoint", "/opt/freuid_artifacts/checkpoints/template_convnext224.pt", "--checkpoint", "/opt/freuid_artifacts/checkpoints/forensic_efficientnet384.pt", "--weight", "0.85", "--weight", "0.15", "--normalization", "rank", "--batch-size", "32", "--num-workers", "4", "--device", "auto"]
