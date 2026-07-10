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

- Freeze the four-way fusion stack from submission `54511333` as the current reproducible candidate unless a clearly better validated candidate emerges before code freeze. Status: two nonzero branches are now captured as a runtime recipe.
- Build a no-network Docker inference path for the frozen conventional + ConvNeXt fusion stack. Status: scaffold added and 5-image smoke test passed; image build blocked locally because Docker Desktop's Linux engine is not running.
- Add a short report section describing public-score history, validation mismatch, and why the final candidate uses source/feature fusion rather than `combined_v4` alone.

## Runtime Recipe

The current submitted stack is reproducible as:

```text
fraud_score = 0.7 * combined_v4_hgb(image) + 0.3 * convnext_tiny_logreg(image)
```

The original four-way fusion search also included `combined_v3_hgb` and `photometric`, but their selected weights were zero. The Docker path therefore computes only the two nonzero branches while preserving the submitted score formula.

Added freeze helpers:

- `scripts/infer_freuid_frozen_stack.py`
- `scripts/freeze_freuid_submission_artifacts.py`
- `docker/freuid/Dockerfile`
- `artifacts/freuid_2026/README.md`

Frozen runtime artifact release:

- Tag: `freuid-freeze-2026-07-10`
- URL: `https://github.com/DrStrangel0ve/ai-image-forensics-comparison/releases/tag/freuid-freeze-2026-07-10`
- Asset: `freuid_frozen_stack_2026_07_10.zip`
- Report asset: `freuid_short_report_draft_2026_07_10.pdf`
- Final-package draft asset: `freuid_final_package_draft_2026_07_10.zip`
- Paired code commit: `46b57a2`

Smoke-test output:

- Command: `python scripts/infer_freuid_frozen_stack.py --max-images 5 ...`
- Output: `outputs/freuid_2026/smoke_frozen_stack_submission.csv`
- Rows: `5`
- Mean score: `0.132092`
- Max absolute difference versus the previously materialized public fusion scores on the same 5 IDs: `0.000337`

Docker build attempt:

- Command: `docker build -f docker/freuid/Dockerfile -t freuid-frozen-stack:local .`
- Result: blocked before build start; Docker daemon pipe `dockerDesktopLinuxEngine` was unavailable.

Follow-up Docker startup attempt:

- Command: start Docker Desktop, then poll `docker version` and `docker info`.
- Result: Docker client was reachable, but the Linux engine kept returning HTTP 500 on the `dockerDesktopLinuxEngine` API pipe for several minutes.
- Added smoke harness: `scripts/smoke_test_freuid_docker.py`.

Second Docker diagnostic:

- `wsl.exe -l -v` showed `docker-desktop` stopped.
- `wsl.exe -d docker-desktop -- echo ...` failed because WSL2 could not start without virtualization / Virtual Machine Platform.
- Error code included `HCS_E_HYPERV_NOT_INSTALLED`.

Final-package draft release:

- Asset: `freuid_final_package_draft_2026_07_10.zip`
- SHA256: `6ff5fff838e242012647e42552e7e4b15aaac6542dbca13b1bece519f0257442`
- Contents: submitted Kaggle CSV, lint/submit manifests, short report PDF/Markdown, pinned discussion reply draft, runbook, reproducibility checklist.
