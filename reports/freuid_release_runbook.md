# FREUID 2026 Release Runbook

This note tracks the current path from local validation to a real Kaggle leaderboard submission.

## Current State

- Competition: `the-freuid-challenge-2026-ijcai-ecai`
- Deadline observed by Kaggle CLI: `2026-07-16 11:59:00 UTC`
- Account entry state: entered
- Current best official submission: `54511333`
- Current best public score: `0.37009`
- Current frozen public-test candidate: `outputs/freuid_2026/public_12k_fourway_fusion_submission_packaged/submission.csv`
- Current frozen runtime release: `https://github.com/DrStrangel0ve/ai-image-forensics-comparison/releases/tag/freuid-freeze-2026-07-10`
- Best local validation candidate: raw score fusion, `0.9661` AUC, `0.2135` APCER at 1% BPCER, `0.0341` AuDET proxy
- Selected fusion formula: `0.7 * combined_v4_hgb + 0.3 * convnext_tiny_logreg`

The local model has produced a valid leaderboard submission and is now frozen as the release candidate unless a clearly better validated candidate appears before code freeze.

## Historical Release Blocker

The Kaggle API accepts pushes for:

```text
arnavmalani/freuid-photometric-offline-submission
```

but the server-side notebook metadata comes back with:

```json
"competition_sources": []
```

The notebook then failed because `/kaggle/input` was empty and `train_labels.csv` could not be found. This persisted after upgrading the local Kaggle package to `2.2.2`. The issue was bypassed by completing the official archive download locally after cooldown and scoring the extracted public-test images on the desktop GPU.

## Fastest Manual Recovery If Kaggle Notebook Execution Is Needed Again

Use the Kaggle web UI for the notebook above:

1. Open the notebook settings or the right-side input/data panel.
2. Add competition data for `The FREUID Challenge 2026 - IJCAI-ECAI`.
3. Verify that the notebook input panel shows the competition dataset before running.
4. Run all cells with internet disabled.
5. Download the notebook output `submission.csv`.
6. Lint locally:

```powershell
.\.venv\Scripts\python.exe scripts\lint_freuid_submission.py `
  --sample-submission data\raw\freuid_2026\small_files\sample_submission.csv `
  --submission outputs\freuid_2026\kaggle_photometric_kernel\submission.csv `
  --manifest-out outputs\freuid_2026\kaggle_photometric_kernel\submission_lint.json
```

7. Submit only if the lint manifest passes and the output is not an all-constant canary.

## Readiness Check

Run this before any upload:

```powershell
.\.venv\Scripts\python.exe scripts\check_freuid_release_state.py `
  --kernel-source-required `
  --manifest-out outputs\freuid_2026\release_state_manifest.json
```

The checker does not submit anything. It records competition entry/submission state, kernel status, whether the competition source is attached, public-test image availability, and whether a real candidate submission exists.

## Freeze Checklist

- Keep `54511333` as the best official submission unless a new candidate beats it under local validation and score-aware linting.
- Keep the repo public and pushed before the July 13, 2026 code freeze.
- Keep the runtime release asset aligned with the repo commit and Dockerfile.
- Build and smoke-test `docker/freuid/Dockerfile` once Docker Desktop's Linux engine is available.
- Review/exported short report PDF: `output/pdf/freuid_short_report_draft_2026_07_10.pdf`.
- Post exactly one reply from `reports/freuid_pinned_discussion_reply_draft_2026_07_10.md` on the pinned Kaggle discussion thread when the organizers open/confirm it.
