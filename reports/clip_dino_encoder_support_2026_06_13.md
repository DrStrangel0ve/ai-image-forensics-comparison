# CLIP/DINO Frozen Encoder Support

Run date: 2026-06-13

This update closes a roadmap gap from the research deep dive: the frozen-encoder baseline can now target CLIP and DINO-style foundation embeddings, not only torchvision classifiers such as ConvNeXt.

## Added Encoders

`scripts/run_frozen_encoder_baseline.py --encoder` now includes:

- `clip_vit_b_32` using `openai/clip-vit-base-patch32`;
- `dinov2_vits14` using `facebook/dinov2-small`;
- `dinov2_vitb14` using `facebook/dinov2-base`.

The implementation lives in `src/forensic_compare/foundation.py` and wraps Hugging Face vision models behind the same frozen-embedding interface used by ConvNeXt, ResNet, ViT, Swin, and EfficientNet.

## Why This Matters

The current strongest frozen baseline is ConvNeXt-Tiny. That is a useful pretrained image model, but it does not directly test the CLIP/DINO hypothesis from the literature: that broad foundation embeddings can generalize better across unseen generators than narrow supervised detectors.

This update makes those baselines first-class experiments:

- same embedding extraction path;
- same logistic/MLP head options;
- same saved `metrics.json`, `predictions.csv`, `classifier.joblib`, and `embeddings.npz`;
- same cross-dataset evaluator;
- explicit CLIP normalization constants and DINO/ImageNet normalization constants.

## Current Status

Code support and tests are in place. The initial support pass did not benchmark CLIP/DINO because the local venv did not yet have `transformers` installed. A follow-up bounded DINOv2-small smoke run is now checked into `reports/dinov2_frozen_encoder_smoke_2026_06_13.md`: 0.7250 accuracy / 0.8225 AUC on the 80/40 Ishu smoke split, and 0.6080 accuracy / 0.7059 AUC on Ishu -> source-balanced MS COCOAI transfer.

This means the next run can start directly from the normal install command:

```powershell
python -m pip install -r requirements.txt
```

## Reproduce

Example Ishu CLIP run:

```powershell
python scripts\run_frozen_encoder_baseline.py `
  --data-dir data\raw\ishu_ai_vs_real_2026 `
  --output-dir runs\frozen_encoder_clip_vit_b32\ishu_seed7 `
  --encoder clip_vit_b_32 `
  --pretrained `
  --classifier logistic_regression `
  --batch-size 32 `
  --device auto `
  --seed 7
```

Example Ishu DINOv2-small run:

```powershell
python scripts\run_frozen_encoder_baseline.py `
  --data-dir data\raw\ishu_ai_vs_real_2026 `
  --output-dir runs\frozen_encoder_dinov2_vits14\ishu_seed7 `
  --encoder dinov2_vits14 `
  --pretrained `
  --classifier logistic_regression `
  --batch-size 32 `
  --device auto `
  --seed 7
```

Cross-dataset evaluation reuses the saved classifier:

```powershell
python scripts\evaluate_frozen_encoder_model.py `
  --model-dir runs\frozen_encoder_clip_vit_b32\ishu_seed7 `
  --target-dir data\raw\ms_cocoai_2026_validation_source_balanced_100 `
  --output-dir runs\frozen_encoder_clip_vit_b32\ishu_seed7_to_ms_cocoai_all4 `
  --target-split all `
  --batch-size 32 `
  --device auto
```

## Verification

Focused checks run in this heartbeat:

```powershell
.\.venv\Scripts\python -m pytest tests\test_frozen_encoder_baseline.py -q
.\.venv\Scripts\python scripts\run_frozen_encoder_baseline.py --help
```

Result: `2 passed`; the CLI help lists `clip_vit_b_32`, `dinov2_vits14`, and `dinov2_vitb14`.

## Next Experiment

Run CLIP ViT-B/32 and DINOv2-small on the same three Ishu seeds used by ConvNeXt, then evaluate Ishu -> source-balanced MS COCOAI. The first paper table should compare:

- frozen ConvNeXt-Tiny;
- frozen CLIP ViT-B/32;
- frozen DINOv2-small;
- SCP-Fusion v0 with and without the best new foundation score.
