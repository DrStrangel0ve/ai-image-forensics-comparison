# FREUID 2026 Release Runbook

This note tracks the current path from local validation to a real Kaggle leaderboard submission.

## Current State

- Competition: `the-freuid-challenge-2026-ijcai-ecai`
- Deadline observed by Kaggle CLI: `2026-07-16 11:59:00 UTC`
- Account entry state: entered
- Official submissions: none yet
- Best local validation candidate: four-branch rank fusion, `0.86875` accuracy, `0.94516` AUC, `0.2625` APCER at the selected `0.0` BPCER validation operating point
- Selected local fusion threshold: `0.64125`

The local model is ready enough for a first release, but a leaderboard submission still requires predictions for every public-test id in `sample_submission.csv`.

## Release Blocker

The Kaggle API accepts pushes for:

```text
arnavmalani/freuid-photometric-offline-submission
```

but the server-side notebook metadata comes back with:

```json
"competition_sources": []
```

The notebook then fails because `/kaggle/input` is empty and `train_labels.csv` cannot be found. This persisted after upgrading the local Kaggle package to `2.2.2`. Full competition archive downloads and targeted public-test image downloads are also currently rate-limited with HTTP 429.

## Fastest Manual Recovery

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
