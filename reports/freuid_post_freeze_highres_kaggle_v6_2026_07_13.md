# FREUID Post-freeze High-resolution Kaggle Result

This is a post-freeze research-only evaluation. It did not create a competition submission and does not alter the two selected frozen final entries.

## Reproducibility

- Kaggle kernel: `arnavmalani/freuid-post-freeze-high-resolution-research`, version 6
- Git tag: `post-freeze-highres-kaggle-v5`
- Git commit: `85893f1695ccd4959287073f04e2f7683c946721`
- Encoder: pretrained DINOv2-B/14, 518 px, frozen
- Views: five crops per document, zoom 1.15
- Probe: standardized balanced logistic regression, `C=0.1`
- Training data: 12,000 documents from four document types, balanced by type and label
- Holdout: 4,000 `EGYPT/DL` documents, 2,000 per label
- Selection SHA-256: `cd653e800790764331091706458b892acf746e908a1a174065ba1adff4caa52f`
- Compute: two Tesla T4 GPUs, 15,636,037,632 bytes each
- Wall time: 1,330.56 seconds

## Results

Lower is better for APCER, AuDET proxy, and the selection objective. Higher is better for ROC AUC.

| View aggregation | Accuracy at 0.5 | ROC AUC | APCER @ 1% BPCER | AuDET proxy | Selection objective |
|---|---:|---:|---:|---:|---:|
| mean | 0.5000 | **0.8585** | **0.6005** | **0.1415** | **0.2916** |
| mean + max | 0.5000 | 0.8428 | 0.6495 | 0.1572 | 0.3196 |
| mean + max + std | 0.5000 | 0.7998 | 0.8350 | 0.2002 | 0.4089 |

The original run manifest names `mean_max_std` as the configured primary aggregation, but the measured objective and every operating-point metric favor plain mean pooling. Results should therefore be compared using the full table, not only the configured primary row.

## Interpretation

The mean DINOv2 embedding provides useful cross-document-type ranking signal, reaching 0.8585 AUC on the held-out EGYPT domain. Adding max and standard-deviation statistics hurts generalization.

Calibration fails badly under this domain shift. At the fixed 0.5 threshold every holdout document is classified as fraud, yielding 2,000 false positives and 2,000 true positives. The threshold required for 1% BPCER is approximately 1.0 for all aggregations. Accuracy and F1 at 0.5 are therefore not useful measures of this probe.

This result supports a narrow conclusion: high-resolution frozen foundation features improve rank separation, but they do not solve the low-BPCER operating point required by FREUID. A future research-only follow-up should estimate calibration without touching the final test set, using source-aware out-of-fold calibration or a nested leave-one-type-out protocol. The frozen competition selections remain unchanged.
