# MS COCOAI to Ishu Source-Utility Threshold Sweep

This report summarizes source-utility threshold selection for the strongly regularized all-branch fusion head.

The score model is unchanged from the `C=0.03` reverse fusion family; only the source-side threshold objective changes.

## Target Results

| config | source_fake_rate_cap | real_fpr_penalty | fake_miss_penalty | accuracy_mean | roc_auc_mean | brier_score_mean | expected_calibration_error_mean | precision_mean | recall_mean | predicted_positive_rate_mean | threshold_mean | threshold_source_predicted_positive_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| score_fusion_all6_c003_source_utility_rfp2_fmp1_cap0p48 | 0.4800 | 2.0000 | 1.0000 | 0.7222 | 0.8291 | 0.2188 | 0.2060 | 0.6711 | 0.8512 | 0.6228 | 0.6657 | 0.4750 |
| score_fusion_all6_c003_source_utility_rfp2_fmp1p5_cap0p48 | 0.4800 | 2.0000 | 1.5000 | 0.7222 | 0.8291 | 0.2188 | 0.2060 | 0.6711 | 0.8512 | 0.6228 | 0.6657 | 0.4750 |
| score_fusion_all6_c003_source_utility_rfp2_fmp2_cap0p48 | 0.4800 | 2.0000 | 2.0000 | 0.7222 | 0.8291 | 0.2188 | 0.2060 | 0.6711 | 0.8512 | 0.6228 | 0.6657 | 0.4750 |
| score_fusion_all6_c003_source_utility_rfp4_fmp1_cap0p48 | 0.4800 | 4.0000 | 1.0000 | 0.7222 | 0.8291 | 0.2188 | 0.2060 | 0.6711 | 0.8512 | 0.6228 | 0.6657 | 0.4750 |
| score_fusion_all6_c003_source_utility_rfp4_fmp1p5_cap0p48 | 0.4800 | 4.0000 | 1.5000 | 0.7222 | 0.8291 | 0.2188 | 0.2060 | 0.6711 | 0.8512 | 0.6228 | 0.6657 | 0.4750 |
| score_fusion_all6_c003_source_utility_rfp4_fmp2_cap0p48 | 0.4800 | 4.0000 | 2.0000 | 0.7222 | 0.8291 | 0.2188 | 0.2060 | 0.6711 | 0.8512 | 0.6228 | 0.6657 | 0.4750 |
| score_fusion_all6_c003_source_utility_rfp8_fmp1_cap0p48 | 0.4800 | 8.0000 | 1.0000 | 0.7222 | 0.8291 | 0.2188 | 0.2060 | 0.6711 | 0.8512 | 0.6228 | 0.6657 | 0.4750 |
| score_fusion_all6_c003_source_utility_rfp8_fmp1p5_cap0p48 | 0.4800 | 8.0000 | 1.5000 | 0.7222 | 0.8291 | 0.2188 | 0.2060 | 0.6711 | 0.8512 | 0.6228 | 0.6657 | 0.4750 |
| score_fusion_all6_c003_source_utility_rfp8_fmp2_cap0p48 | 0.4800 | 8.0000 | 2.0000 | 0.7222 | 0.8291 | 0.2188 | 0.2060 | 0.6711 | 0.8512 | 0.6228 | 0.6657 | 0.4750 |
| score_fusion_all6_c003_source_utility_rfp2_fmp1_cap0p45 | 0.4500 | 2.0000 | 1.0000 | 0.7193 | 0.8291 | 0.2188 | 0.2060 | 0.6717 | 0.8393 | 0.6140 | 0.6847 | 0.4500 |
| score_fusion_all6_c003_source_utility_rfp2_fmp1p5_cap0p45 | 0.4500 | 2.0000 | 1.5000 | 0.7193 | 0.8291 | 0.2188 | 0.2060 | 0.6717 | 0.8393 | 0.6140 | 0.6847 | 0.4500 |
| score_fusion_all6_c003_source_utility_rfp2_fmp2_cap0p45 | 0.4500 | 2.0000 | 2.0000 | 0.7193 | 0.8291 | 0.2188 | 0.2060 | 0.6717 | 0.8393 | 0.6140 | 0.6847 | 0.4500 |

## Best Operating Point By Source Fake-Rate Cap

| source_fake_rate_cap_label | accuracy_mean | precision_mean | recall_mean | predicted_positive_rate_mean | threshold_mean | threshold_source_predicted_positive_rate_mean |
| --- | --- | --- | --- | --- | --- | --- |
| 0.48 | 0.7222 | 0.6711 | 0.8512 | 0.6228 | 0.6657 | 0.4750 |
| 0.45 | 0.7193 | 0.6717 | 0.8393 | 0.6140 | 0.6847 | 0.4500 |
| 0.50 | 0.7193 | 0.6624 | 0.8869 | 0.6608 | 0.6280 | 0.4917 |
| none | 0.7193 | 0.6624 | 0.8869 | 0.6608 | 0.6280 | 0.4917 |

## Read

The best source-utility operating point is `score_fusion_all6_c003_source_utility_rfp2_fmp1_cap0p48` with 0.7222 mean accuracy, 0.8291 AUC, and 0.6228 target fake-call rate.
This matches the previous `cap_0p48` source-accuracy threshold result rather than exceeding it. The sweep therefore strengthens the conclusion that the source fake-rate cap is doing the useful operating-point work for this score model.
Without a source fake-rate cap, the best utility-selected point reaches 0.7193 accuracy with 0.6608 target fake-call rate, so the cap still gives the cleaner decision frontier.

Use this as operating-point evidence, not as a new scoring-model result: AUC/Brier/ECE reflect the same fused scores for each seed/config family, while accuracy and predicted fake rate move with threshold selection.

Next step: move the utility from post-hoc threshold selection into fusion training or validation selection, because threshold-only source utility does not improve beyond the capped source-threshold baseline.

## Rebuild

```powershell
.\.venv\Scripts\python.exe scripts\run_reverse_source_utility_sweep.py --python .\.venv\Scripts\python.exe --skip-existing
```
