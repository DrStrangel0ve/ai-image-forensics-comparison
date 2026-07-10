FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    TORCH_HOME=/opt/freuid_artifacts/torch

WORKDIR /app

RUN python -m pip install --no-cache-dir --upgrade pip

COPY docker/freuid/requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt

COPY src /app/src
COPY scripts/infer_freuid_frozen_stack.py /app/scripts/infer_freuid_frozen_stack.py
COPY artifacts/freuid_2026 /opt/freuid_artifacts

RUN mkdir -p /submissions

ENTRYPOINT ["python", "/app/scripts/infer_freuid_frozen_stack.py", "--input-dir", "/data", "--output-csv", "/submissions/submission.csv", "--combined-v4-model", "/opt/freuid_artifacts/combined_v4_hgb.joblib", "--convnext-model", "/opt/freuid_artifacts/convnext_tiny_logreg.joblib", "--torch-home", "/opt/freuid_artifacts/torch", "--device", "auto"]
