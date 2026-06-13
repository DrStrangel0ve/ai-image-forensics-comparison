# Source-Utility Threshold Strategy

Run date: 2026-06-13

This implementation pass adds a utility-aware binary threshold strategy to `scripts/fuse_prediction_scores.py`.

## What Changed

`--threshold-strategy source_utility` now selects the fused-score decision threshold on source-domain rows using:

```text
utility =
  fake_detection_weight * fake_recall
  + real_clearance_weight * real_specificity
  - real_fpr_penalty * real_false_positive_rate
  - fake_miss_penalty * fake_false_negative_rate
```

The strategy works with the existing source-threshold machinery:

- threshold selection can use the full source train rows or the held-out source calibration split via `--calibration-fraction`;
- `--threshold-tiebreak higher` can prefer conservative fake-call thresholds when utility ties;
- `--threshold-max-positive-rate` can cap the source predicted-fake rate before threshold selection;
- selected utility weights, penalties, threshold source, and source fake-call rate are written to `metrics.json` and `summary.csv`.

## Why This Matters

The current reverse-transfer results show a split between ranking and operating-point quality:

- branch-dropout fusion gives the best reverse AUC frontier at `0.8406`, but it still over-calls generated images;
- capped source-threshold fusion gives the best reverse decision point so far at `0.7222` accuracy, but it is still a post-hoc threshold policy;
- physics-guided ResNet remains an interpretable calibration anchor.

`source_utility` is the next bridge between those pieces. It lets the reverse suite tune the exact forensic tradeoff we care about, instead of optimizing only source accuracy, F1, Youden's J, or a fixed threshold.

## Smoke Coverage

`tests/test_score_fusion.py` now includes a CLI regression test for `source_utility` with:

- held-out source calibration rows;
- asymmetric real-FPR and fake-miss penalties;
- conservative threshold tie-breaking;
- source predicted-fake-rate cap.

## Next Experiment

The MS-COCOAI-to-Ishu all-branch fusion grid has now been run in `reports/ms_cocoai_to_ishu_source_utility_threshold_2026_06_13.md`.

Result:

- best source-utility setting: `0.7222` accuracy / `0.8291` AUC with the `0.48` source fake-rate cap;
- best uncapped utility setting: `0.7193` accuracy with `0.6608` target fake-call rate;
- conclusion: source-utility thresholding matches the previous capped source-accuracy operating point, but does not improve beyond it.

The next model-side experiment should use `source_utility` as the evaluator while changing how the fusion head is selected or trained. A useful grid would sweep:

| parameter | suggested values |
| --- | --- |
| fusion C | `0.03`, `0.1` |
| branch dropout | current best dropout and no-dropout settings |
| real-FPR penalty | `2`, `4`, `8` |
| fake-miss penalty | `1`, `1.5`, `2` |
| source fake-rate cap | none, `0.50`, `0.48`, `0.45` |

The comparison should keep the existing reverse suite fixed and report accuracy, AUC, Brier, ECE, recall, precision, and predicted fake rate. A true SCP-Fusion v1 candidate should beat or match the `0.7222` capped source-threshold operating point while improving fake-call bias or calibration, not merely reproduce the same threshold.
