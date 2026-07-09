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
- `scripts/materialize_freuid_download_manifest.py`
  - Rebuilds labeled train/validation metadata CSVs from download manifests and `train_labels.csv`.
  - Preserves the downloaded/skipped manifest order so feature runs are reproducible from acquisition logs.
- `scripts/run_freuid_feature_baseline.py`
  - Runs a CSV-based conventional baseline with the repo's photometric/noise/JPEG/frequency feature sets.
- `scripts/run_freuid_frozen_encoder_baseline.py`
  - Runs pretrained frozen image encoders on FREUID CSV metadata.
  - Writes FREUID metrics, cached embeddings, and validation predictions compatible with score fusion.
- `scripts/apply_freuid_fusion.py`
  - Applies a saved fusion summary to unlabeled prediction CSVs.
  - Enforces source-name order, supports raw/min-max/rank normalization, and writes `id,fraud_score,label` predictions.
- `scripts/select_freuid_threshold.py`
  - Selects a reproducible FREUID operating threshold from validation scores at a bounded BPCER target.
  - Writes a threshold/metric manifest plus optional thresholded validation predictions for later public-test packaging.
- `kaggle/freuid_photometric_submission/`
  - Kaggle kernel scaffold for the first leaderboard release path.
  - Runs inside Kaggle against mounted competition data, trains a type/label-balanced photometric baseline, writes `submission.csv`, and lints it before output.

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

Feature-cache follow-up: `scripts/run_freuid_feature_baseline.py` now supports `--feature-cache-dir`. On the 160/80 photometric slice, the warm run populated 240 cached vectors and the repeat run reported 160 train hits / 80 validation hits with identical metrics. This removes repeated JPEG/feature extraction as the main blocker for larger conventional sweeps.

Manifest materialization follow-up: `scripts/materialize_freuid_download_manifest.py` regenerated the balanced 160-train and 80-validation metadata CSVs from their download manifests. The regenerated files exactly matched the existing local CSVs in ID order, label counts, and per-type counts, so future acquisition waves can be reconstructed directly from manifest logs.

Balanced 320/160 follow-up: downloaded and materialized a larger targeted slice with 320 training images and 160 validation images. Both splits are exactly balanced across the five document types and both labels, with 32 train rows and 16 validation rows per type/label cell. Lower APCER/AuDET proxy is better:

| Run | Accuracy | ROC AUC | APCER @ 1% BPCER | AuDET proxy | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `combined_v3_hgb` | 0.8063 | 0.9198 | 0.3375 | 0.0843 | 0.1326 | 0.1332 |
| `photometric_logreg` | 0.8063 | 0.8709 | 0.4875 | 0.1323 | 0.1412 | 0.0728 |
| rank fusion, weights 0.30/0.70 photometric/`combined_v3_hgb` | 0.8313 | 0.9135 | 0.3125 | 0.0908 | 0.1258 | 0.1347 |

Interpretation: on the larger local FREUID slice, `combined_v3_hgb` is now the strongest single conventional branch for AUC and AuDET, while the rank fusion buys a better strict APCER operating point, accuracy, and Brier score. This is promising but still too small for a public-test submission claim; the next competitive step is to add a neural embedding branch and validate the same operating point on a larger holdout.

Frozen-encoder follow-up on the same 320/160 slice: added a CSV-native FREUID frozen encoder runner and tested pretrained ConvNeXt-Tiny and ResNet-18 embeddings with calibrated logistic regression. Lower APCER/AuDET proxy is better:

| Run | Accuracy | ROC AUC | APCER @ 1% BPCER | AuDET proxy | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `convnext_tiny_logreg` | 0.8500 | 0.9227 | 0.3500 | 0.0814 | 0.1149 | 0.0846 |
| `resnet18_logreg` | 0.8000 | 0.8497 | 0.8750 | 0.1511 | 0.1592 | 0.1370 |
| rank fusion: photometric + `combined_v3_hgb` + ConvNeXt | 0.8563 | 0.9423 | 0.2625 | 0.0623 | 0.1145 | 0.1261 |
| rank fusion: photometric + `combined_v3_hgb` + ConvNeXt + ResNet-18 | 0.8688 | 0.9452 | 0.2625 | 0.0595 | 0.1143 | 0.1356 |

Interpretation: ConvNeXt adds useful complementary signal to the conventional branch; ResNet-18 is poor alone, but the coarse rank-fusion grid assigns it only 10% weight and slightly improves AuDET/AUC. The current best local validation candidate is the four-branch rank fusion, but this is still validation-only. Do not upload until we either validate on a larger holdout or train/package a matching public-test inference path.

Embedding-cache proof: rerunning the ConvNeXt branch with `--embedding-cache-dir outputs\freuid_2026\embedding_cache` reported 320 train hits / 160 validation hits, zero misses, and unchanged metrics. Frozen neural sweeps can now be repeated without re-encoding the current 320/160 slice.

Threshold-selection follow-up: `scripts/select_freuid_threshold.py` selected a reproducible threshold for the current best local validation file, `runs\freuid_balanced320_160_fusion_conventional_convnext_resnet18\fused_predictions.csv`. The selected threshold is 0.64125 at BPCER target 1%, with validation accuracy 0.8688, ROC AUC 0.9452, APCER 0.2625, BPCER 0.0000, AuDET proxy 0.0595, and thresholded label counts 101/59 for labels 0/1. The manifest is `outputs\freuid_2026\best_local_fusion_threshold_manifest.json`.

Public-test fusion packaging follow-up: `scripts/apply_freuid_fusion.py` now freezes the best local fusion recipe for unlabeled public-test branch scores. It expects prediction names in the saved `source_names` order from `runs\freuid_balanced320_160_fusion_conventional_convnext_resnet18\fusion_summary.json` (`photometric_logreg`, `combined_v3_hgb`, `convnext_tiny_logreg`, `resnet18_logreg`), applies rank normalization with weights 0.0/0.4/0.5/0.1, and thresholds with the selected 0.64125 operating point. The resulting `id,fraud_score,label` file can then be passed through `scripts/package_freuid_submission.py` and `scripts/lint_freuid_submission.py` before any Kaggle upload.

First-release kernel follow-up: local public-test image access is still the submission blocker, so the repo now includes `kaggle\freuid_photometric_submission`. This private Kaggle script kernel avoids local per-file downloads by using the competition data mounted inside Kaggle. The kernel is self-contained and internet-disabled: it prepares a type/label-stratified split, trains a 640-sample photometric logistic baseline, scores the full sample submission, packages `submission.csv`, and lints it before exposing the output. This is a conservative first leaderboard release path; it is not expected to match the local fusion candidate, but it should produce a real official score once the remote run completes with the competition source attached.

Kernel-source blocker: Kaggle API pushes for both `arnavmalani/freuid-photometric-first-submission` and `arnavmalani/freuid-photometric-offline-submission` accepted the script, but the server-side metadata stripped `competition_sources` to `[]`, leaving `/kaggle/input` empty. The offline kernel is now self-contained and submit-eligible if the competition data is attached manually in Kaggle UI, but API-only kernel launch is not yet producing mounted FREUID data.

Acquisition retry follow-up: upgrading the local Kaggle CLI from 2.2.1 to 2.2.2 did not fix the server-side kernel metadata issue. A 50-image public-test acquisition wave failed immediately with 8 consecutive HTTP 429 responses, and the full-archive `competitions download` endpoint also returned HTTP 429. The current release blocker is Kaggle API access/rate limiting, not local model or submission formatting code.

Balanced 640-train follow-up: completed and materialized a larger 640-image training slice, exactly balanced across the five document types and both labels, with 64 rows per type/label cell. A matching 320-validation download was started but Kaggle returned HTTP 429 rate-limit responses after 164 usable rows, so that partial validation manifest should not be used for model selection.

Using the clean existing 160-validation slice, larger training alone did not beat the 320-trained fusion:

| Run | Accuracy | ROC AUC | APCER @ 1% BPCER | AuDET proxy |
| --- | ---: | ---: | ---: | ---: |
| `photometric_logreg`, train640/val160 | 0.8000 | 0.8672 | 0.3875 | 0.1366 |
| `combined_v3_hgb`, train640/val160 | 0.8063 | 0.9042 | 0.4750 | 0.0991 |
| `convnext_tiny_logreg`, train640/val160 | 0.8500 | 0.9177 | 0.5875 | 0.0849 |
| rank/raw fusion, train640/val160 | 0.8313 | 0.9233 | 0.2625 | 0.0813 |

Interpretation: the 640-trained fusion recovers the same APCER@1%BPCER as the 320-trained best, but its AuDET/AUC are worse. The current best local candidate remains the 320-trained four-branch rank fusion. Next acquisition should wait/retry the 320-validation manifest rather than keep increasing train size against the same small holdout.

Downloader-safety follow-up: after a cooldown retry of the same 320-validation plan, local availability remained 164/320 usable images and the slow retry timed out before writing a new manifest. `scripts/download_freuid_images.py` now checkpoints manifests during acquisition and supports `--stop-after-failures` so rate-limited runs preserve partial progress and can stop before spending long windows on repeated HTTP 429 bursts.

Checkpointed retry follow-up: a bounded retry of the 320-validation plan wrote `outputs/freuid_2026/balanced320_val_download_manifest_retry_checkpoint.json`, skipped 163 already-local files, then stopped after 8 consecutive HTTP 429 failures. No new validation images were acquired. The downloader now avoids sleeping on local skips so future top-up attempts reach the first real download request quickly while still honoring `--sleep-seconds` after successful network downloads.

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

Apply the frozen local fusion recipe once unlabeled public-test branch predictions exist:

```powershell
.\.venv\Scripts\python.exe scripts\apply_freuid_fusion.py `
  --predictions outputs\freuid_2026\public_photometric.csv outputs\freuid_2026\public_combined_v3_hgb.csv outputs\freuid_2026\public_convnext_tiny.csv outputs\freuid_2026\public_resnet18.csv `
  --names photometric_logreg combined_v3_hgb convnext_tiny_logreg resnet18_logreg `
  --fusion-summary runs\freuid_balanced320_160_fusion_conventional_convnext_resnet18\fusion_summary.json `
  --threshold-json outputs\freuid_2026\best_local_fusion_threshold_manifest.json `
  --out-predictions outputs\freuid_2026\public_fused_predictions.csv
```

Run the first-release Kaggle kernel after verifying the competition source is attached:

```powershell
.\.venv\Scripts\python.exe -m kaggle kernels push `
  -p kaggle\freuid_photometric_submission `
  --timeout 43200
```

After the remote run completes, download and submit the linted output:

```powershell
.\.venv\Scripts\python.exe -m kaggle kernels output arnavmalani/freuid-photometric-offline-submission `
  -p outputs\freuid_2026\kaggle_photometric_kernel

.\.venv\Scripts\python.exe -m kaggle competitions submit `
  -c the-freuid-challenge-2026-ijcai-ecai `
  -f outputs\freuid_2026\kaggle_photometric_kernel\submission.csv `
  -m "Photometric balanced 640 first release"
```
