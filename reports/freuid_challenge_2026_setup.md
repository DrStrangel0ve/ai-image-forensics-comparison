# FREUID Challenge 2026 Setup

Date: 2026-07-09

## Competition Snapshot

- Kaggle slug: `the-freuid-challenge-2026-ijcai-ecai`
- Status from Kaggle CLI: account has entered the competition.
- Kaggle API deadline: 2026-07-16 11:59:00 UTC.
- Public leaderboard snapshot: best visible score is 0.00031; lower leaderboard values appear better.
- Official task framing: identity document fraud detection across physical manipulations, GenAI-driven edits, and print-and-capture forgeries.
- Official metric family: AuDET and APCER at 1% BPCER; Kaggle exposes a combined FREUID score.

Sources:
- Kaggle competition page: https://www.kaggle.com/competitions/the-freuid-challenge-2026-ijcai-ecai
- IJCAI-ECAI 2026 competitions page: https://2026.ijcai.org/competitions/

## Local Data State

Downloaded small competition files:

- `data/raw/freuid_2026/small_files/train_labels.csv`: 69,352 labeled training rows.
- `data/raw/freuid_2026/small_files/sample_submission.csv`: 142,818 test IDs.

Training label summary from `train_labels.csv`:

- Label 0: 40,005 rows.
- Label 1: 29,347 rows.
- Document/source types: `BENIN/DL`, `EGYPT/DL`, `GUINEA/DL`, `MAURITIUS/ID`, `MOZAMBIQUE/DL`.
- `is_digital` is almost entirely true in the current labels file, with 20 false rows.

The first full Kaggle archive download stalled and left an invalid partial zip, which was removed. The next download attempt should use either a longer uninterrupted window or targeted file downloads for smoke runs.

## Added Tooling

- `src/forensic_compare/freuid.py`
  - Local APCER at bounded BPCER operating-point helper.
  - Local AuDET proxy for validation ranking only; Kaggle remains authoritative.
- `scripts/package_freuid_submission.py`
  - Converts prediction CSVs into exact Kaggle `id,label` format.
  - Preserves sample submission order and rejects missing/extra IDs.
- `scripts/lint_freuid_submission.py`
  - Validates exact `id,label` columns, row count, ID set/order, uniqueness, and binary labels.

Generated local canary artifact:

- `outputs/freuid_2026/baseline_all_zero_submission.csv`
- `outputs/freuid_2026/baseline_all_zero_submission_manifest.json`
- `outputs/freuid_2026/baseline_all_zero_submission_lint.json`

Lint result: pass, 142,818 rows, exact sample ID set/order, all labels binary.

This canary is useful for upload-format safety, but it is not a competitive submission.

## Competitive Plan

1. Finish image acquisition.
   - Preferred: rerun the full Kaggle archive download with a multi-hour timeout.
   - Fallback: targeted train/public-test file downloader for smoke experiments and partial model validation.

2. Build a FREUID validation split.
   - Stratify by `type` and label.
   - Track per-type APCER@1%BPCER and the global local AuDET proxy.
   - Avoid selecting by accuracy; the leaderboard operating point punishes false accepts at very low false-reject tolerance.

3. First competitive baselines.
   - Frozen ConvNeXt / DINO-style encoder embeddings plus calibrated logistic regression.
   - Existing conventional feature family: JPEG/noise/frequency/chroma/edge/photometric features.
   - Small ResNet or ConvNeXt fine-tune at document-friendly resolution.

4. Ensemble for low-BPCER behavior.
   - Score-level fusion between neural embeddings and conventional artifacts.
   - Tune decision thresholds against `APCER@1%BPCER`.
   - Optimize for leaderboard score only after local split stability is reasonable across document types.

5. Submission discipline.
   - Every candidate prediction CSV goes through `package_freuid_submission.py`.
   - Every packaged CSV goes through `lint_freuid_submission.py`.
   - Do not upload any file containing scores, labels, paths, or helper columns; Kaggle wants exactly `id,label`.

## Immediate Commands

Package an all-zero format canary:

```powershell
.\.venv\Scripts\python.exe scripts\package_freuid_submission.py `
  --sample-submission data\raw\freuid_2026\small_files\sample_submission.csv `
  --out-path outputs\freuid_2026\baseline_all_zero_submission.csv `
  --manifest-out outputs\freuid_2026\baseline_all_zero_submission_manifest.json
```

Lint a packaged FREUID submission:

```powershell
.\.venv\Scripts\python.exe scripts\lint_freuid_submission.py `
  --sample-submission data\raw\freuid_2026\small_files\sample_submission.csv `
  --submission outputs\freuid_2026\baseline_all_zero_submission.csv `
  --manifest-out outputs\freuid_2026\baseline_all_zero_submission_lint.json
```
