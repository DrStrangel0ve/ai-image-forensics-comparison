FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    TORCH_HOME=/opt/freuid_artifacts/torch \
    FREUID_VARIANT=ood_rank

WORKDIR /app

RUN python -m pip install --no-cache-dir --upgrade pip

COPY docker/freuid/requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt

COPY src /app/src
COPY scripts/infer_freuid_finetune.py /app/scripts/infer_freuid_finetune.py
COPY scripts/infer_freuid_checkpoint_ensemble.py /app/scripts/infer_freuid_checkpoint_ensemble.py
COPY docker/freuid/entrypoint.sh /app/docker/entrypoint.sh
COPY artifacts/freuid_2026 /opt/freuid_artifacts

RUN mkdir -p /submissions && sed -i 's/\r$//' /app/docker/entrypoint.sh && chmod +x /app/docker/entrypoint.sh

ENTRYPOINT ["/app/docker/entrypoint.sh"]
