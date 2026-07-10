# FREUID Kaggle Submission Log - 2026-07-09

Generated during the first successful FREUID leaderboard upload session.

## Official Metric Check

The Kaggle Evaluation page states that `label` must be a numeric fraud score, not a hard class label:

- Higher score means more likely fraudulent.
- The FREUID Score combines AuDET and APCER at 1% BPCER.
- Lower public score is better.

This corrected the earlier local assumption that `label` had to be binary.

## Submissions

| Ref | Description | Public score | Notes |
| --- | --- | ---: | --- |
| 54481645 | metadata size baseline v1: released public sizes, conservative fallback | 1.00000 | Binary labels; valid upload but wrong output semantics for this metric. |
| 54481898 | metadata size score baseline v2: continuous fraud scores | 0.98832 | Correct score-valued format; weak metadata-only ranking. |
| 54481939 | metadata size score baseline v3: inverted direction check | 0.97578 | Inversion improved slightly, suggesting metadata direction/domain shift is weak but not enough. |
| 54503265 | image four-way fusion v1: 1276 local split rank fusion with metadata fallback | 0.38042 | First pixel-based public-test submission; replaces 7,821 public image IDs with rank-fused image scores and keeps metadata-score fallback for hidden/sample-only IDs. |

## Current Best

Best public score so far: `0.38042` from submission `54503265`.

This is a large improvement over metadata-only scoring, but it is still not competitive with the public leaderboard top rows near `0.0003`. The next useful step is to train the same recipe on a larger official-image split now that the archive is local.

## Data Access Notes

- Kaggle notebook CLI metadata still strips `competition_sources`, so the remote notebook launched with empty `/kaggle/input`.
- Browser Add Input found the competition after account access was granted, but the UI stalled at `Adding data source` and did not persist the input.
- The official Data page reports `77,189` files and `17.58 GB`.
- Kaggle API download attempts hit `429 Too Many Requests` after file-manifest pagination; retry after cooldown.

## Local Artifacts

- Score builder: `scripts/build_freuid_metadata_submission.py`
- Corrected linter: `scripts/lint_freuid_submission.py --allow-score-labels`
- Local-image materializer: `scripts/materialize_freuid_local_images.py`
- Ignored generated outputs:
  - `outputs/freuid_2026/metadata_size_score_submission/`
  - `outputs/freuid_2026/metadata_size_score_inverted_submission/`

## 2026-07-09 Cooldown Iteration

The official archive and single-file Kaggle download endpoints are still returning `429 Too Many Requests`, so no image-based public-test candidate was submitted in this iteration.

While waiting for the throttle to clear, the local train image set was materialized from disk:

- Local train images matched: `1,276` of `69,352` train rows.
- Labels: `635` bona fide / `641` fraud.
- Types: `BENIN/DL 254`, `EGYPT/DL 257`, `GUINEA/DL 256`, `MAURITIUS/ID 257`, `MOZAMBIQUE/DL 252`.
- Full local split, seed 37: `1,019` train / `257` validation, type+label stratified.

Validation results on that fuller local split:

| Method | Accuracy | AUC | APCER @ 1% BPCER | AuDET proxy |
| --- | ---: | ---: | ---: | ---: |
| `combined_v3` + HGB | 0.8327 | 0.9300 | 0.3023 | 0.0719 |
| Pretrained ConvNeXt-Tiny + logistic head | 0.8093 | 0.9016 | 0.4419 | 0.0998 |
| Rank fusion, 0.65 `combined_v3` / 0.35 ConvNeXt | 0.8716 | 0.9419 | 0.2868 | 0.0608 |

The takeaway is that local document fraud signal is real, but the leaderboard remains metadata-only until public-test images can be acquired or the Kaggle notebook data source attaches successfully.

## 2026-07-09 Two-Hour Follow-Up

After another cooldown, both the full-archive endpoint and public-test image access remained `429`-blocked, so no additional leaderboard candidate was submitted.

The local split was extended with the physics branch:

| Method | Accuracy | AUC | APCER @ 1% BPCER | AuDET proxy |
| --- | ---: | ---: | ---: | ---: |
| Photometric logistic baseline | 0.7821 | 0.8593 | 0.4419 | 0.1428 |
| Rank fusion, 0.65 `combined_v3` / 0.30 ConvNeXt / 0.05 photometric | 0.8599 | 0.9412 | 0.2481 | 0.0614 |

The small photometric weight does not improve ranking, but it improves the low-BPCER fraud operating point. This is useful for the next image-based public-test candidate once Kaggle data access recovers.

## 2026-07-09 Archive-Stall Follow-Up

The cooldown retry progressed past the immediate `429` failure and wrote a partial archive, but then stalled at `131,072,000` bytes for roughly ten minutes. The hung Kaggle download process was stopped so it would not linger. The partial file remains at `data/raw/freuid_2026/archive/the-freuid-challenge-2026-ijcai-ecai.zip` as evidence that the endpoint is intermittently available but not yet completing.

The full local split was extended with `combined_v4` and a four-way fusion:

| Method | Accuracy | AUC | APCER @ 1% BPCER | AuDET proxy |
| --- | ---: | ---: | ---: | ---: |
| `combined_v4` + HGB | 0.8444 | 0.9378 | 0.3643 | 0.0642 |
| Rank fusion, 0.45 `combined_v3` / 0.20 ConvNeXt / 0.10 photometric / 0.25 `combined_v4` | 0.8599 | 0.9422 | 0.2403 | 0.0604 |

The current local submission recipe is therefore a rank-normalized ensemble of conventional `combined_v3`, ConvNeXt, photometric, and `combined_v4` scores. It is ready to apply to public-test images once the archive finishes or Kaggle source attachment starts working.

## 2026-07-10 First Image-Based Leaderboard Submission

The official Kaggle archive completed after cooldown and was extracted locally.

- Archive entries extracted under `data/raw/freuid_2026/images`: `77,186`.
- Matched train images: `69,352`.
- Matched public-test images: `7,821`.
- Matched train-sample images: `13`.

The first public-test candidate used the existing seed-37 local split (`1,019` train / `257` validation) to avoid introducing a new risky training protocol before confirming leaderboard flow. Public-test scoring branches:

| Branch | Validation APCER @ 1% BPCER | Validation AuDET proxy | Public-test rows |
| --- | ---: | ---: | ---: |
| `combined_v3` + HGB | 0.3333 | 0.0742 | 7,821 |
| Pretrained ConvNeXt-Tiny + logistic head | 0.4419 | 0.1000 | 7,821 |
| Photometric logistic baseline | 0.4419 | 0.1427 | 7,821 |
| `combined_v4` + HGB | 0.3643 | 0.0642 | 7,821 |
| Four-way rank fusion | 0.2403 | 0.0604 | 7,821 |

The submitted file preserves full Kaggle sample order and row count:

- Rows: `142,818`.
- Public image IDs replaced by four-way fusion scores: `7,821`.
- Remaining IDs filled with the prior best metadata-score fallback: `134,997`.
- Lint: `scripts/lint_freuid_submission.py --allow-score-labels` passed.
- Kaggle submission ref: `54503265`.
- Public score: `0.38042`.

Operational note: the installed local Torch build is CPU-only, so the ConvNeXt branch was run on CPU despite the desktop having a 3060 Ti. Installing a CUDA-enabled Torch build should make the next neural/foundation iterations faster, but it is not required for correctness.

## 2026-07-10 Larger Conventional Submission

The full local train set was split with seed 37 using type+label stratification, then capped to a type+label-balanced `12,000` train / `4,000` validation run for fast iteration. All public-test image IDs were scored from the extracted official archive.

| Branch | Accuracy | AUC | APCER @ 1% BPCER | AuDET proxy |
| --- | ---: | ---: | ---: | ---: |
| `combined_v3` + HGB | 0.8678 | 0.9407 | 0.2670 | 0.0595 |
| Photometric logistic baseline | 0.8130 | 0.8880 | 0.3935 | 0.1121 |
| `combined_v4` + HGB | 0.8850 | 0.9584 | 0.2385 | 0.0417 |
| Coarse three-way fusion, 0.20 `combined_v3` / 0.10 photometric / 0.70 `combined_v4` | 0.8848 | 0.9568 | 0.2380 | 0.0434 |

Because the three-way fusion only improved APCER by `0.0005` while worsening AuDET proxy and AUC, the submitted candidate used the single `combined_v4` image score branch plus the prior metadata fallback.

Submitted file:

- Rows: `142,818`.
- Public image IDs replaced by `combined_v4` image scores: `7,821`.
- Remaining IDs filled with the prior best metadata-score fallback: `134,997`.
- Lint: `scripts/lint_freuid_submission.py --allow-score-labels` passed.
- Kaggle submission ref: `54505114`.
- Public score: `0.39303`.

This did not beat the current best `0.38042` from submission `54503265`. The larger local split improved validation metrics but transferred worse to the public leaderboard, so the next useful direction is source/domain-robust validation and fusion rather than simply increasing conventional-feature training rows.

## 2026-07-10 CUDA ConvNeXt Fusion Candidate

The global Python install is CPU-only (`torch 2.7.1+cpu`), but the repo `.venv` has CUDA Torch (`torch 2.11.0+cu128`) and sees the local RTX 3080 Ti. The larger ConvNeXt branch should therefore run through `.venv\Scripts\python.exe`, not the global interpreter.

An aligned type+label-balanced `12,000` train / `4,000` validation split was materialized so the ConvNeXt branch uses the same validation IDs as the conventional branches:

- Train CSV: `outputs/freuid_2026/local_train_full_seed37_train_balanced_type_label_12000.csv`.
- Validation CSV: `outputs/freuid_2026/local_train_full_seed38_val_balanced_type_label_4000.csv`.
- Validation overlap with the conventional `combined_v4` run: `4,000 / 4,000`.

Validation results:

| Branch | Accuracy | AUC | APCER @ 1% BPCER | AuDET proxy |
| --- | ---: | ---: | ---: | ---: |
| Aligned pretrained ConvNeXt-Tiny + logistic head | 0.8628 | 0.9457 | 0.2985 | 0.0550 |
| Four-way fusion, 0.00 `combined_v3` / 0.30 ConvNeXt / 0.00 photometric / 0.70 `combined_v4` | n/a | 0.9661 | 0.2135 | 0.0341 |

The fusion is the best local FREUID validation candidate so far by a large margin. The score-valued submission package was regenerated with `scripts/package_freuid_score_submission.py` and linted with `--allow-score-labels`:

- Rows: `142,818`.
- Public image IDs replaced by four-way fusion scores: `7,821`.
- Remaining IDs filled with the prior best metadata-score fallback: `134,997`.
- Candidate path: `outputs/freuid_2026/public_12k_fourway_fusion_submission_packaged/submission.csv`.
- Lint: passed.

Kaggle upload attempt:

- Message: `12k four-way fusion v3: combined_v4 plus CUDA ConvNeXt with metadata fallback`.
- Result: rejected after upload with `400 Client Error: Bad Request`.
- No new submission ref appeared in `kaggle competitions submissions`.

This candidate is ready to retry when the Kaggle submission limit/window clears or when the server-side cause of the `400` can be inspected. Current leaderboard best remains `54503265` with public score `0.38042`.

## 2026-07-10 Submission Guard Follow-Up

The Kaggle competition rules page states that the standard daily limit is anticipated to be up to `5` submissions per team per day. A dry run with `scripts/submit_freuid_candidate.py` at `2026-07-09T23:10:55Z` found `5 / 5` submissions for July 9 UTC, matching the earlier `400 Bad Request` failure pattern.

The guarded helper now:

- lints score-valued `id,label` submissions before upload;
- estimates same-day Kaggle submissions from structured Kaggle API timestamps;
- blocks by default when the estimated daily limit is exhausted;
- submits and polls for ref/status/public score when a slot is available.

Use it to retry the best candidate after the UTC day rolls over:

```powershell
python scripts\submit_freuid_candidate.py `
  --sample-submission data\raw\freuid_2026\small_files\sample_submission.csv `
  --submission outputs\freuid_2026\public_12k_fourway_fusion_submission_packaged\submission.csv `
  --message "12k four-way fusion v3: combined_v4 plus CUDA ConvNeXt with metadata fallback" `
  --manifest-out outputs\freuid_2026\public_12k_fourway_fusion_submission_packaged\submit_manifest.json
```

## 2026-07-10 Accepted CUDA ConvNeXt Fusion Submission

After the UTC daily submission window reset, the guarded helper successfully uploaded the best local candidate.

- Kaggle submission ref: `54511333`.
- Message: `12k four-way fusion v3: combined_v4 plus CUDA ConvNeXt with metadata fallback`.
- Status: complete.
- Public score: `0.37009`.
- Previous best: `54503265` at `0.38042`.
- Absolute improvement: `0.01033` lower FREUID score.
- Candidate path: `outputs/freuid_2026/public_12k_fourway_fusion_submission_packaged/submission.csv`.
- Submit manifest: `outputs/freuid_2026/public_12k_fourway_fusion_submission_packaged/submit_manifest.json`.

Current best public leaderboard submission is now `54511333`.
