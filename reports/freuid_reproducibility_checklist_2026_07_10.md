# FREUID Reproducibility Checklist - 2026-07-10

This checklist records competition requirements pulled from the Kaggle competition pages through the Kaggle API during the July 10 heartbeat.

## Current Leaderboard State

- Current best submitted ref: `54511333`.
- Current best public score: `0.37009`.
- Current best candidate artifact: `outputs/freuid_2026/public_12k_fourway_fusion_submission_packaged/submission.csv`.
- Candidate validation: four-way fusion AuDET proxy `0.0341`, APCER @ 1% BPCER `0.2135`, AUC `0.9661`.
- The first upload attempt for that candidate failed with Kaggle `400 Bad Request`, likely because the July 9 UTC submission count was already `5 / 5`.
- The retry after UTC reset succeeded as submission `54511333`.

## Submission Guard

Use the guarded helper for the next upload attempt:

```powershell
python scripts\submit_freuid_candidate.py `
  --sample-submission data\raw\freuid_2026\small_files\sample_submission.csv `
  --submission outputs\freuid_2026\public_12k_fourway_fusion_submission_packaged\submission.csv `
  --message "12k four-way fusion v3: combined_v4 plus CUDA ConvNeXt with metadata fallback" `
  --manifest-out outputs\freuid_2026\public_12k_fourway_fusion_submission_packaged\submit_manifest.json
```

The helper lints score-valued labels, estimates daily submission count from Kaggle's UTC timestamps, blocks when the default `5`-per-day guard is exhausted, submits when a slot is available, and polls for ref/status/public score.

## Eligibility Requirements

- Public repository must contain training code, model weights, and inference pipeline before private test images are released.
- Code freeze date: July 13, 2026.
- Final package deadline: July 15, 2026, 23:59 AoE.
- Required package: Kaggle prediction files, source code under an OSI-approved license in a public git repo, short technical report PDF, runnable Docker container for no-network inference, and exactly one reply on the pinned Kaggle discussion thread.
- After code freeze, allowed work is limited to private-test inference with existing weights plus documentation, Docker, and orchestration fixes that do not change weights/checkpoints/hyperparameters.
- After code freeze, do not train, fine-tune, change checkpoints, alter architecture, change hyperparameters, or use test images for self-supervised/unsupervised updates.

## Docker Target

The organizer sandbox mounts flat test images at `/data/` and expects:

```text
/submissions/submission.csv
```

with:

```csv
id,label
<filename_without_extension>,<fraud_score>
```

The container must run with no network, keep weights inside the image, and write only under `/submissions/`.

## Next Practical Work

- Freeze the four-way fusion stack from submission `54511333` as the current reproducible candidate unless a clearly better validated candidate emerges before code freeze.
- Build a no-network Docker inference path for the frozen conventional + ConvNeXt fusion stack.
- Add a short report section describing public-score history, validation mismatch, and why the final candidate uses source/feature fusion rather than `combined_v4` alone.
