# Post-freeze high-resolution document forensics track

Date: 2026-07-13

## Eligibility boundary

This track was created after the FREUID code freeze. It is independent research and must not replace, alter, or be blended into the frozen competition submissions or the `freuid-final-2026-07-13` release. Public competitor descriptions are cited as motivation; the implementation in this repository is independent and uses standard `timm`, multi-crop, LoRA, and rank-ensemble components.

## Why this replaces the photometric branch

The FREUID task is dominated by localized document edits, high-resolution text and portrait changes, recapture shift, and unseen document types. Single-image photometric normal consistency is weak for this setting: there is no controlled multi-light acquisition, flat printed documents have little recoverable surface geometry, and resizing a full page destroys the small evidence that matters. Photometric and residual features remain useful as explainability or diversity signals, but they should not be the primary detector.

The stronger track therefore emphasizes:

1. DINOv2 and MaxViT encoders with 512/518 px inputs.
2. Deterministic zoomed five-crop document views (four corners plus center).
3. Mean-logit view aggregation, which prevents a learned attention head from ignoring a small altered region.
4. Frozen-encoder probes for low-memory baselines and QKV LoRA for parameter-efficient adaptation.
5. Per-epoch checkpoints and equal-weight percentile-rank snapshot ensembles.
6. Leave-one-document-type-out evaluation instead of random paired-image splits.

The design is consistent with the broad ingredients disclosed by strong public FREUID systems, including the high-resolution DINOv2 snapshot study from [seantangth](https://github.com/seantangth/freuid-challenge-2026), the five-window MaxViT ensemble from [RayServe](https://github.com/rayserve1999/freuid2026-ensemble), and the recapture-focused multi-backbone work from [hoppery](https://github.com/hoppery/freuid-challenge-2026). It does not import their code or weights.

## Implemented controls

`scripts/train_freuid_finetune.py` now supports:

- `--model dinov2_base_518|dinov2_large_518|maxvit_base_512|maxvit_large_512`
- `--view-mode five_crop --five-crop-zoom 1.15`
- `--view-pooling mean_logits`
- `--freeze-encoder`
- `--lora-rank 16 --lora-alpha 16`
- `--gradient-checkpointing`
- `--gradient-accumulation-steps N`
- `--view-chunk-size N` for memory-bounded inference and frozen feature probes
- `--save-every-epoch` for snapshot ensembles

Every new setting is stored in the checkpoint and reconstructed by inference. Checkpoints made before this branch retain their previous defaults.

## First experiment: DINOv2-B frozen probe

Run the same command for each available leave-one-type-out split. Egypt is shown here:

```powershell
.\.venv\Scripts\python.exe scripts\train_freuid_finetune.py `
  --train-csv outputs\freuid_2026\loto_egypt_train.csv `
  --val-csv outputs\freuid_2026\loto_egypt_val.csv `
  --image-root data\raw\freuid_2026\images `
  --output-dir runs\post_freeze_dinov2b_fivecrop_loto_egypt `
  --model dinov2_base_518 --image-size 518 --epochs 4 `
  --batch-size 1 --gradient-accumulation-steps 16 `
  --view-mode five_crop --view-pooling mean_logits --view-chunk-size 1 `
  --freeze-encoder --lr 5e-4 --save-every-epoch
```

This is the safest first run on a 12 GiB GPU. It trains only about 6,000 head parameters while retaining high-resolution foundation features.

## Second experiment: DINOv2-B QKV LoRA

```powershell
.\.venv\Scripts\python.exe scripts\train_freuid_finetune.py `
  --train-csv outputs\freuid_2026\loto_egypt_train.csv `
  --val-csv outputs\freuid_2026\loto_egypt_val.csv `
  --image-root data\raw\freuid_2026\images `
  --output-dir runs\post_freeze_dinov2b_lora_fivecrop_loto_egypt `
  --model dinov2_base_518 --image-size 518 --epochs 4 `
  --batch-size 1 --gradient-accumulation-steps 16 `
  --view-mode five_crop --view-pooling mean_logits --view-chunk-size 1 `
  --lora-rank 16 --lora-alpha 16 --gradient-checkpointing `
  --lr 1e-4 --save-every-epoch
```

The QKV adapters expose about 0.68% of DINOv2-B parameters to training. If this configuration exceeds 12 GiB, reduce to a single 518 px view for LoRA and retain five-crop inference as an ablation rather than silently lowering resolution.

## Snapshot rank inference

The existing sequential ensemble runner loads one checkpoint at a time, which bounds GPU memory:

```powershell
.\.venv\Scripts\python.exe scripts\infer_freuid_checkpoint_ensemble.py `
  --input-dir data\raw\freuid_2026\images\public_test --recursive `
  --checkpoint runs\post_freeze_dinov2b_fivecrop_loto_egypt\model_epoch1.pt --weight 1 `
  --checkpoint runs\post_freeze_dinov2b_fivecrop_loto_egypt\model_epoch2.pt --weight 1 `
  --checkpoint runs\post_freeze_dinov2b_fivecrop_loto_egypt\model_epoch3.pt --weight 1 `
  --checkpoint runs\post_freeze_dinov2b_fivecrop_loto_egypt\model_epoch4.pt --weight 1 `
  --normalization rank `
  --output-csv runs\post_freeze_dinov2b_fivecrop_loto_egypt\snapshot_submission.csv `
  --batch-size 1 --num-workers 2
```

## Evaluation gates

A model is worth carrying forward only if it improves the mean and worst held-out-type results across all available LOTO folds. Report ROC AUC, AuDET proxy, APCER at 1% BPCER, calibration error, and transform robustness. Do not use the random type/label split as primary evidence because paired pristine/tampered documents can leak templates across train and validation.

No metric from the 20-image smoke test is scientific evidence. Its sole purpose is to verify the training, checkpoint, and inference contracts.

## Current execution status

- Focused tests: 11 passed.
- DINOv2-B LoRA construction: 12 QKV adapters, 87,174,150 total parameters, 594,438 trainable (0.682%).
- Five-crop end-to-end smoke: passed; epoch and best checkpoints were both emitted and reloaded for inference.
- Full GPU experiment: pending. At verification time the RTX 3080 Ti had only 956 MiB free because unrelated desktop processes occupied the device; those processes were not interrupted.

## Kaggle GPU execution

The first full probe is configured as a private Kaggle script at `kaggle/freuid_post_freeze_highres_research`. It requests the `NvidiaTeslaT4` accelerator, mounts the official competition data, and never creates a competition submission. The run caches normalized DINOv2-B embeddings once, then compares fixed logistic probes over mean, mean-plus-max, and mean-plus-max-plus-standard-deviation crop statistics.

The predeclared primary result is the `mean_max_std` probe on an `EGYPT/DL` leave-one-type-out fold, capped deterministically at 12,000 training and 4,000 validation documents. This makes the first remote run useful while staying within the available Kaggle GPU allocation. All outputs carry `post_freeze_research_only` eligibility metadata.
