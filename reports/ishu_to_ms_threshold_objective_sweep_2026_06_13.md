# Ishu to MS Threshold Objective Sweep

This diagnostic sweeps source-selected decision objectives on saved Ishu-trained all-foundation SCP-Fusion scores. It does not retrain branches; it asks whether the fixed score ranker has a better source-derived operating point.

## Top Objective Sensitivity Rows

| policy | n_seeds | target_accuracy_mean | target_roc_auc_mean | target_brier_score_mean | target_expected_calibration_error_mean | target_predicted_positive_rate_mean | source_predicted_positive_rate_mean | threshold_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| uncapped_fd0p5_rc1_rfp0p5_fmp3 | 3 | 0.6470 | 0.7995 | 0.3118 | 0.3263 | 0.1863 | 0.5263 | 0.3420 |
| cap0p55_fd0p5_rc1_rfp0p5_fmp3 | 3 | 0.6447 | 0.7995 | 0.3118 | 0.3263 | 0.1840 | 0.5205 | 0.3473 |
| uncapped_fd0p5_rc1_rfp0p5_fmp1 | 3 | 0.6267 | 0.7995 | 0.3118 | 0.3263 | 0.1540 | 0.5175 | 0.4406 |
| cap0p55_fd0p5_rc1_rfp0p5_fmp1 | 3 | 0.6243 | 0.7995 | 0.3118 | 0.3263 | 0.1517 | 0.5117 | 0.4459 |
| cap0p5_fd0p5_rc1_rfp0p5_fmp1 | 3 | 0.6173 | 0.7995 | 0.3118 | 0.3263 | 0.1340 | 0.4971 | 0.5045 |
| fixed_0p5 | 3 | 0.6163 | 0.7995 | 0.3118 | 0.3263 | 0.1323 | 0.4942 | 0.5000 |
| uncapped_fd0p5_rc1_rfp1_fmp1 | 3 | 0.6090 | 0.7995 | 0.3118 | 0.3263 | 0.1343 | 0.5000 | 0.5233 |
| cap0p5_fd0p5_rc1_rfp1_fmp1 | 3 | 0.6020 | 0.7995 | 0.3118 | 0.3263 | 0.1167 | 0.4854 | 0.5820 |
| uncapped_fd0p5_rc1_rfp2_fmp1 | 3 | 0.5813 | 0.7995 | 0.3118 | 0.3263 | 0.0920 | 0.4649 | 0.6886 |
| uncapped_fd0p5_rc1_rfp4_fmp1 | 3 | 0.5640 | 0.7995 | 0.3118 | 0.3263 | 0.0727 | 0.4532 | 0.7366 |
| cap0p45_fd0p5_rc1_rfp0p5_fmp1 | 3 | 0.5583 | 0.7995 | 0.3118 | 0.3263 | 0.0623 | 0.4444 | 0.8089 |
| cap0p45_fd0p5_rc1_rfp1_fmp1 | 3 | 0.5543 | 0.7995 | 0.3118 | 0.3263 | 0.0577 | 0.4386 | 0.8170 |

## Fixed Threshold Baseline

| policy | n_seeds | target_accuracy_mean | target_roc_auc_mean | target_brier_score_mean | target_expected_calibration_error_mean | target_predicted_positive_rate_mean | source_predicted_positive_rate_mean | threshold_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_0p5 | 3 | 0.6163 | 0.7995 | 0.3118 | 0.3263 | 0.1323 | 0.4942 | 0.5000 |

## Reverse-Like Utility Family

| policy | n_seeds | target_accuracy_mean | target_roc_auc_mean | target_brier_score_mean | target_expected_calibration_error_mean | target_predicted_positive_rate_mean | source_predicted_positive_rate_mean | threshold_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| uncapped_fd1_rc1_rfp4_fmp1p5 | 3 | 0.5813 | 0.7995 | 0.3118 | 0.3263 | 0.0920 | 0.4649 | 0.6886 |
| cap0p5_fd1_rc1_rfp4_fmp1p5 | 3 | 0.5813 | 0.7995 | 0.3118 | 0.3263 | 0.0920 | 0.4649 | 0.6886 |
| cap0p55_fd1_rc1_rfp4_fmp1p5 | 3 | 0.5813 | 0.7995 | 0.3118 | 0.3263 | 0.0920 | 0.4649 | 0.6886 |
| cap0p6_fd1_rc1_rfp4_fmp1p5 | 3 | 0.5813 | 0.7995 | 0.3118 | 0.3263 | 0.0920 | 0.4649 | 0.6886 |
| cap0p45_fd1_rc1_rfp4_fmp1p5 | 3 | 0.5543 | 0.7995 | 0.3118 | 0.3263 | 0.0577 | 0.4386 | 0.8170 |

## Per-Seed Rows for Best Policy

| policy | seed | threshold | target_accuracy | target_roc_auc | target_predicted_positive_rate | source_predicted_positive_rate |
| --- | --- | --- | --- | --- | --- | --- |
| uncapped_fd0p5_rc1_rfp0p5_fmp3 | 7 | 0.2196 | 0.7020 | 0.8074 | 0.2540 | 0.5263 |
| uncapped_fd0p5_rc1_rfp0p5_fmp3 | 17 | 0.5415 | 0.5960 | 0.7819 | 0.1160 | 0.4912 |
| uncapped_fd0p5_rc1_rfp0p5_fmp3 | 29 | 0.2648 | 0.6430 | 0.8091 | 0.1890 | 0.5614 |

## Read

The strongest target-accuracy diagnostic row is `uncapped_fd0p5_rc1_rfp0p5_fmp3` at 0.6470 accuracy / 0.7995 AUC, with a 0.1863 target fake-call rate.
The fixed 0.5 threshold baseline is 0.6163 accuracy / 0.7995 AUC with a 0.1323 target fake-call rate.
The reverse-transfer utility family is too conservative in this direction: source fake-rate caps mostly raise thresholds and make the model miss generated images. A less punitive real-FPR objective improves decision accuracy, but this sweep should be treated as a sensitivity result until the utility family is selected without target feedback.

## Rebuild

```powershell
.\.venv\Scripts\python.exe scripts\sweep_ishu_to_ms_threshold_objectives.py
```
