# Source-Heldout Triage Mode

Run date: 2026-06-12

This follow-up adds a practical forensic triage diagnostic. Instead of forcing every image into a binary real/fake decision, it selects two thresholds on a source-heldout calibration split:

- scores below the lower threshold are `likely real`;
- scores above the upper threshold are `likely fake`;
- scores in the middle are `uncertain`.

The thresholds are selected on all non-heldout generated sources plus the real-image calibration split. The evaluation is then run on the held-out generated source plus the real-image test split. This tests whether the detector can make fewer but more reliable decisions under generator shift.

Implementation:

- `scripts/summarize_source_holdout_triage.py`
- score modes:
  - raw saved scores;
  - class-balanced temperature-scaled scores.
- seeds: 7, 17, 29
- held-out generated sources per seed: SD2.1, SDXL, SD3, DALL-E 3, Midjourney 6
- summary CSVs now include 95% deterministic bootstrap confidence intervals over the 15 held-out source/seed rows.

Because temperature scaling is monotonic, it does not change the triage membership when thresholds are selected after calibration. It is still included as a sanity check: calibration can improve probability quality while leaving rank-based triage decisions unchanged.

## 5% Calibration Error Budget

This operating point constrains calibration real false-positive rate to at most 5% and calibration fake false-clearance rate to at most 5%.

| method | decided coverage | triage accuracy | held-out real FPR | held-out fake false clearance | held-out fake detection | held-out real clearance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen ConvNeXt-Tiny | 0.2371 | 0.7542 | 0.0547 | 0.0600 | 0.2600 | 0.1493 |
| SCP-Fusion v0 | 0.2143 | 0.7476 | 0.0520 | 0.0527 | 0.2467 | 0.1283 |
| ResNet-18 | 0.1861 | 0.6892 | 0.0573 | 0.0567 | 0.1513 | 0.1200 |
| physics-guided fusion | 0.1909 | 0.6554 | 0.0693 | 0.0533 | 0.1880 | 0.1013 |
| `combined_v3` | 0.0998 | 0.6033 | 0.0333 | 0.0573 | 0.0767 | 0.0528 |

ConvNeXt and SCP-Fusion are the most useful high-confidence triage detectors at this strict operating point. They only decide on about one quarter of target images, but those decisions are much cleaner than default-threshold full-coverage classification.

The checked-in 5% summary asset now reports uncertainty columns. Frozen ConvNeXt-Tiny has triage accuracy `0.7542` with a 95% bootstrap interval of `[0.7027, 0.7990]`; SCP-Fusion v0 has triage accuracy `0.7476` with interval `[0.7091, 0.7845]`.

## 10% Calibration Error Budget

This looser operating point increases coverage but shows the source-shift cost more clearly.

| method | decided coverage | triage accuracy | held-out real FPR | held-out fake false clearance | held-out fake detection | held-out real clearance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SCP-Fusion v0 | 0.4373 | 0.7160 | 0.1307 | 0.1060 | 0.4093 | 0.2755 |
| frozen ConvNeXt-Tiny | 0.4055 | 0.6976 | 0.1253 | 0.1140 | 0.3813 | 0.2443 |
| physics-guided fusion | 0.3438 | 0.6572 | 0.1200 | 0.1140 | 0.2840 | 0.2021 |
| ResNet-18 | 0.3341 | 0.6347 | 0.1253 | 0.1080 | 0.2600 | 0.1952 |
| `combined_v3` | 0.2430 | 0.5424 | 0.1120 | 0.1100 | 0.1780 | 0.1131 |

The 10% setting roughly doubles decided coverage for SCP-Fusion, from 0.2143 to 0.4373, but held-out real FPR rises to 0.1307 and fake false-clearance rises to 0.1060. That is acceptable as a diagnostic curve, not as a deployment claim.

At the 10% setting, SCP-Fusion v0 has triage accuracy `0.7160` with a 95% bootstrap interval of `[0.6933, 0.7374]`, and decided coverage `0.4373` with interval `[0.4242, 0.4499]`.

## Interpretation

This is a cleaner practical story than a single default threshold:

- Full-coverage thresholded accuracy is brittle under source shift.
- Strict triage can give investigators a smaller high-confidence set.
- Frozen ConvNeXt and SCP-Fusion are the strongest triage candidates.
- The monotonic temperature result confirms that probability calibration and rank-based triage are separate knobs.

For DFRWS, this can be presented as an investigative triage workflow. For WIFS/DFF, it supports a stronger claim: robust AI-image forensics needs both score ranking and explicit operating-point calibration, especially when the generator source is unknown.

## Reproduce

5% operating point:

```powershell
python scripts/summarize_source_holdout_triage.py `
  --metadata data\raw\ms_cocoai_2026_validation_source_balanced_100\metadata.csv `
  --out-dir runs\source_holdout_triage\ishu_to_ms_cocoai_all4 `
  --split validation `
  --seed 7 `
  --max-real-fpr 0.05 `
  --max-fake-clearance 0.05 `
  --score-mode raw `
  --score-mode temperature_balanced `
  --predictions seed7:combined_v3=runs\ishu_to_ms_cocoai_source_balanced_seed7\combined_v3\predictions.csv `
  --predictions seed7:resnet18=runs\ishu_to_ms_cocoai_source_balanced_seed7\resnet18\predictions.csv `
  --predictions seed7:physics_guided=runs\ishu_to_ms_cocoai_source_balanced_seed7\physics_guided_resnet18_combined_v3\predictions.csv `
  --predictions seed7:convnext_tiny_frozen=runs\ishu_to_ms_cocoai_source_balanced_seed7\convnext_tiny_frozen\predictions.csv `
  --predictions seed7:scp_fusion_v0=runs\score_fusion\ishu_seed7_to_ms_cocoai_all4\ms_cocoai\predictions.csv `
  --predictions seed17:combined_v3=runs\ishu_to_ms_cocoai_source_balanced_seed17\combined_v3\predictions.csv `
  --predictions seed17:resnet18=runs\ishu_to_ms_cocoai_source_balanced_seed17\resnet18\predictions.csv `
  --predictions seed17:physics_guided=runs\ishu_to_ms_cocoai_source_balanced_seed17\physics_guided_resnet18_combined_v3\predictions.csv `
  --predictions seed17:convnext_tiny_frozen=runs\ishu_to_ms_cocoai_source_balanced_seed17\convnext_tiny_frozen\predictions.csv `
  --predictions seed17:scp_fusion_v0=runs\score_fusion\ishu_seed17_to_ms_cocoai_all4\ms_cocoai\predictions.csv `
  --predictions seed29:combined_v3=runs\ishu_to_ms_cocoai_source_balanced_seed29\combined_v3\predictions.csv `
  --predictions seed29:resnet18=runs\ishu_to_ms_cocoai_source_balanced_seed29\resnet18\predictions.csv `
  --predictions seed29:physics_guided=runs\ishu_to_ms_cocoai_source_balanced_seed29\physics_guided_resnet18_combined_v3\predictions.csv `
  --predictions seed29:convnext_tiny_frozen=runs\ishu_to_ms_cocoai_source_balanced_seed29\convnext_tiny_frozen\predictions.csv `
  --predictions seed29:scp_fusion_v0=runs\score_fusion\ishu_seed29_to_ms_cocoai_all4\ms_cocoai\predictions.csv
```

For the 10% operating point, change `--max-real-fpr` and `--max-fake-clearance` to `0.10` and write to `runs\source_holdout_triage\ishu_to_ms_cocoai_all4_10pct`.

Checked-in summaries:

- `reports\assets\source_holdout_triage_summary_ishu_ms_cocoai_all4_5pct.csv`
- `reports\assets\source_holdout_triage_summary_ishu_ms_cocoai_all4_10pct.csv`
