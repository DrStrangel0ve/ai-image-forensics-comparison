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

Kaggle stores image files under nested split directories:

- Training label paths like `train/<id>.jpeg` download as `train/train/<id>.jpeg`.
- Public test IDs download as `public_test/public_test/<id>.jpeg`.

This mapping is handled by `freuid_competition_path` and `scripts/download_freuid_images.py`.

Smoke verification: `scripts/download_freuid_images.py` successfully downloaded 5 training JPEGs through the nested path mapping, and PIL opened all 5 as RGB images.

## Added Tooling

- `src/forensic_compare/freuid.py`
  - Local APCER at bounded BPCER operating-point helper.
  - Local AuDET proxy for validation ranking only; Kaggle remains authoritative.
- `scripts/package_freuid_submission.py`
  - Converts prediction CSVs into exact Kaggle `id,label` format.
  - Preserves sample submission order and rejects missing/extra IDs.
- `scripts/lint_freuid_submission.py`
  - Validates exact `id,label` columns, row count, ID set/order, uniqueness, and binary labels.
- `scripts/download_freuid_images.py`
  - Downloads targeted image subsets with Kaggle's nested FREUID file paths.
  - Supports `--balance-columns type label` for small, evenly stratified acquisition waves.
- `scripts/run_freuid_feature_baseline.py`
  - Runs a CSV-based conventional baseline with the repo's photometric/noise/JPEG/frequency feature sets.

Generated local canary artifact:

- `outputs/freuid_2026/baseline_all_zero_submission.csv`
- `outputs/freuid_2026/baseline_all_zero_submission_manifest.json`
- `outputs/freuid_2026/baseline_all_zero_submission_lint.json`

Lint result: pass, 142,818 rows, exact sample ID set/order, all labels binary.

This canary is useful for upload-format safety, but it is not a competitive submission.

## Smoke Baseline

To verify the real-data path without making a leaderboard claim, a 30-image training smoke subset was downloaded and split into 21 train / 9 validation rows. `combined_v3` conventional features with logistic regression completed successfully:

- Validation accuracy: 0.6667.
- Validation ROC AUC: 0.6667.
- Local APCER at 1% BPCER: 0.6667.
- Local AuDET proxy: 0.3611.
- Skipped images: 0.

This is only a pipeline check; the validation set is too small for model selection or leaderboard expectations.

Balanced 80/40 follow-up: downloaded 80 train and 40 validation images balanced across the five document types and both labels. This is still small, but it is the first type-balanced real-data comparison. Lower APCER/AuDET proxy is better:

| Run | Accuracy | ROC AUC | APCER @ 1% BPCER | AuDET proxy |
| --- | ---: | ---: | ---: | ---: |
| `photometric_logreg` | 0.8250 | 0.9050 | 0.5000 | 0.1075 |
| `combined_v3_logreg` | 0.7500 | 0.8775 | 0.6500 | 0.1312 |
| `combined_v3_hgb` | 0.7000 | 0.8100 | 0.7000 | 0.1975 |
| `combined_v4_logreg` | 0.7000 | 0.7725 | 0.7500 | 0.2338 |
| `noise_v3_logreg` | 0.6750 | 0.7575 | 0.9000 | 0.2450 |

Interpretation: on this tiny balanced slice, the photometric proxy is surprisingly strongest. Do not treat this as a leaderboard estimate; use it to prioritize the next acquisition wave and to keep photometric features in the early FREUID ensemble.

Balanced 160/80 follow-up: expanded the same acquisition strategy to 160 train and 80 validation images, still exactly balanced across five document types and labels. Lower APCER/AuDET proxy is better:

| Run | Accuracy | ROC AUC | APCER @ 1% BPCER | AuDET proxy | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `photometric_logreg` | 0.7375 | 0.8012 | 0.5750 | 0.2041 | 0.1779 | 0.1222 |
| `combined_v3_hgb` | 0.7750 | 0.8581 | 0.6500 | 0.1462 | 0.1770 | 0.1573 |
| `combined_v4_logreg` | 0.6750 | 0.8156 | 0.7000 | 0.1881 | 0.1692 | 0.1391 |
| `combined_v3_logreg` | 0.7125 | 0.8187 | 0.7250 | 0.1847 | 0.1723 | 0.1174 |

Interpretation: photometric still wins the strict low-BPCER operating point, while `combined_v3_hgb` wins AUC and the local AuDET proxy. This points toward a calibrated score-fusion step between the photometric branch and the richer `combined_v3` branch before any Kaggle submission.

Fusion follow-up: added `scripts/fuse_freuid_scores.py` and grid-searched validation score fusion over raw, min-max, and rank-normalized scores.

| Fusion | Accuracy | ROC AUC | APCER @ 1% BPCER | AuDET proxy | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `photometric_logreg` + `combined_v3_hgb`, rank weights 0.45/0.55 | 0.7500 | 0.8553 | 0.5750 | 0.1500 | 0.1571 | 0.0873 |
| four-branch conventional fusion, rank weights 0.50/0.50/0.00/0.00 | 0.7500 | 0.8494 | 0.5750 | 0.1562 | 0.1584 | 0.0711 |

Interpretation: the two-branch rank fusion keeps the photometric low-BPCER operating point and substantially improves its AuDET proxy and calibration. The four-branch search gives zero weight to the extra logreg branches, so the current FREUID conventional ensemble should stay focused on photometric + `combined_v3_hgb` until the data slice is larger.

Run the same baseline after downloading a larger split:

```powershell
.\.venv\Scripts\python.exe scripts\run_freuid_feature_baseline.py `
  --train-csv outputs\freuid_2026\split_train.csv `
  --val-csv outputs\freuid_2026\split_val.csv `
  --image-root data\raw\freuid_2026\images `
  --output-dir runs\freuid_combined_v3 `
  --feature-set combined_v3 `
  --classifier logistic_regression
```

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
