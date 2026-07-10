# FREUID Challenge 2026 Short Report Draft

## Summary

This submission uses a frozen two-branch image-forensics stack for document-fraud detection:

```text
score = 0.7 * combined_v4_hgb(image) + 0.3 * convnext_tiny_logreg(image)
```

The public leaderboard submission is `54511333`, with public score `0.37009`. Lower is better for the FREUID score. The submitted file contains `142,818` rows in Kaggle sample-submission order; `7,821` IDs with locally available public-test images receive image-model scores, and the remaining sample-only IDs receive the prior metadata fallback used in earlier valid submissions.

## Data And Validation

The official FREUID archive was downloaded through Kaggle after API throttling cleared and extracted locally.

Local materialization:

- Matched train images: `69,352`
- Matched public-test images: `7,821`
- Working validation split: type+label-balanced `12,000` train / `4,000` validation, seed-controlled

The validation objective prioritizes ranking and low-BPCER operation:

- AUC
- APCER at 1% BPCER
- AuDET proxy
- calibration checks where available

## Model Branches

The conventional branch is `combined_v4_hgb`, a histogram-gradient-boosted classifier over photometric consistency, noise residual, JPEG/blocking, frequency, chroma, and reconstruction-lite features.

The neural branch is `convnext_tiny_logreg`, a frozen ImageNet-pretrained ConvNeXt-Tiny encoder with a calibrated logistic classifier trained on frozen embeddings. It uses public pretrained weights and does not fine-tune the backbone.

The original fusion search evaluated `combined_v3_hgb`, `photometric`, `combined_v4_hgb`, and `convnext_tiny_logreg`. The best validation recipe assigned zero weight to `combined_v3_hgb` and `photometric`, so the frozen runtime computes only the two nonzero branches.

## Validation Results

On the aligned `12,000 / 4,000` split:

| Method | Accuracy | AUC | APCER @ 1% BPCER | AuDET proxy |
| --- | ---: | ---: | ---: | ---: |
| `combined_v3_hgb` | 0.8678 | 0.9407 | 0.2670 | 0.0595 |
| `photometric_logreg` | 0.8130 | 0.8880 | 0.3935 | 0.1121 |
| `combined_v4_hgb` | 0.8850 | 0.9584 | 0.2385 | 0.0417 |
| `convnext_tiny_logreg` | 0.8628 | 0.9457 | 0.2985 | 0.0550 |
| frozen score fusion | 0.8938 | 0.9661 | 0.2135 | 0.0341 |

The fusion improves ranking and low-BPCER validation behavior over either branch alone. The public leaderboard also improved over previous submissions:

| Submission | Description | Public score |
| --- | --- | ---: |
| `54481645` | metadata binary canary | 1.00000 |
| `54481898` | metadata score baseline | 0.98832 |
| `54481939` | inverted metadata score | 0.97578 |
| `54503265` | first image four-way fusion with metadata fallback | 0.38042 |
| `54505114` | larger `combined_v4` image scores with metadata fallback | 0.39303 |
| `54511333` | frozen `combined_v4` + ConvNeXt score fusion | 0.37009 |

## Reproducibility

Repository:

- `https://github.com/DrStrangel0ve/ai-image-forensics-comparison`

Frozen artifact release:

- `https://github.com/DrStrangel0ve/ai-image-forensics-comparison/releases/tag/freuid-freeze-2026-07-10`

Important runtime files:

- `scripts/infer_freuid_frozen_stack.py`
- `scripts/freeze_freuid_submission_artifacts.py`
- `docker/freuid/Dockerfile`
- `artifacts/freuid_2026/README.md`
- `artifacts/freuid_2026/fusion_summary.json`

Runtime contract:

```text
Input:  flat image directory mounted at /data
Output: /submissions/submission.csv
Format: id,label
ID:     filename stem
Label:  fraud score in [0, 1]
```

Local smoke test:

- Rows: `5`
- Mean score: `0.132092`
- Max absolute difference versus the previously materialized public fusion scores on the same IDs: `0.000337`

Docker build status:

- Dockerfile scaffold is present.
- Local Docker image build is pending because Docker Desktop's Linux engine was unavailable during the July 10 heartbeat.

## Limitations

The public-test archive currently exposes only `7,821` public-test image files locally, while the sample submission contains `142,818` rows. The submitted file therefore uses image scores where images are available and a metadata fallback elsewhere. This is valid for Kaggle upload, but it means the public score partly reflects fallback behavior for rows without local images.

The validation split is source/type stratified, but it is still a local proxy. The gap between `combined_v4` validation strength and its weaker public score shows that domain shift matters. The final fusion is chosen because it improves both local low-BPCER metrics and public score relative to the prior best.

## Final Package Status

Completed:

- Valid Kaggle submission: `54511333`
- Public repo pushed
- Frozen runtime release uploaded
- Score-aware submission linter and guarded submit helper
- Docker inference scaffold
- PDF draft exported: `output/pdf/freuid_short_report_draft_2026_07_10.pdf`

Remaining:

- Build and run Docker image once Docker Desktop's Linux engine is available
- Post the pinned Kaggle discussion reply
