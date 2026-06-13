# MS COCOAI to Ishu Source-Holdout Tuned Fusion

This experiment trains score-level fusion heads directly under leave-one-generator-out source validation. It searches logistic-regression regularization, branch-dropout augmentation, and source fake-rate caps; then it retrains the selected configuration on all source rows before evaluating Ishu.

The default policy is intentionally conservative: select the grid point with the best worst-source utility under a 0.48 source fake-call cap.

## Final Target Result

| selection_policy | n_seeds | target_accuracy_mean | target_roc_auc_mean | target_brier_score_mean | target_expected_calibration_error_mean | target_predicted_positive_rate_mean | source_predicted_positive_rate_mean | selection_validation_utility_mean_mean | selected_configs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| source_holdout_tuned_fusion | 3 | 0.7339 | 0.8341 | 0.2748 | 0.2954 | 0.6813 | 0.4700 | 1.4260 | seed7:C1:none:cap0.48; seed17:C1:none:cap0.48; seed29:C0.01:mean0p35x8:cap0.48 |

## Selected Per Seed

| seed | fusion_c | dropout_config | source_fake_rate_cap | selection_validation_utility_mean | selection_validation_utility_min | target_accuracy | target_roc_auc | target_predicted_positive_rate | threshold_source_predicted_positive_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7 | 1.0000 | none | 0.4800 | 1.3523 | 0.8706 | 0.7632 | 0.8390 | 0.7281 | 0.4800 |
| 17 | 1.0000 | none | 0.4800 | 1.4508 | 0.8043 | 0.6579 | 0.7783 | 0.7105 | 0.4750 |
| 29 | 0.0100 | mean0p35x8 | 0.4800 | 1.4747 | 0.7500 | 0.7807 | 0.8849 | 0.6053 | 0.4550 |

## Source-Holdout Grid Frontier

| seed | fusion_c | dropout_config | source_fake_rate_cap | validation_utility_mean | validation_utility_min | validation_accuracy_mean | validation_predicted_positive_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 29 | 1.0000 | none | 0.4800 | 1.4775 | 0.6250 | 0.9401 | 0.2254 |
| 29 | 0.0100 | mean0p35x8 | 0.4800 | 1.4747 | 0.7500 | 0.9401 | 0.2253 |
| 29 | 0.0300 | mean0p35x8 | 0.4800 | 1.4520 | 0.7500 | 0.9373 | 0.2225 |
| 29 | 0.0100 | none | 0.4800 | 1.4520 | 0.7500 | 0.9373 | 0.2225 |
| 17 | 1.0000 | none | 0.4800 | 1.4508 | 0.8043 | 0.9355 | 0.2184 |
| 29 | 0.1000 | mean0p35x8 | 0.4800 | 1.4497 | 0.6250 | 0.9372 | 0.2224 |
| 29 | 0.0300 | none | 0.4800 | 1.4293 | 0.7500 | 0.9345 | 0.2197 |
| 17 | 0.3000 | none | 0.4800 | 1.4214 | 0.8043 | 0.9325 | 0.2154 |
| 29 | 1.0000 | mean0p35x8 | 0.4800 | 1.4043 | 0.6250 | 0.9316 | 0.2169 |
| 29 | 0.3000 | mean0p35x8 | 0.4800 | 1.4043 | 0.6250 | 0.9316 | 0.2169 |
| 29 | 0.3000 | none | 0.4800 | 1.4015 | 0.7500 | 0.9316 | 0.2168 |
| 17 | 1.0000 | mean0p35x8 | 0.4800 | 1.3920 | 0.6765 | 0.9295 | 0.2124 |

## Read

The tuned fusion head reaches 0.7339 mean target accuracy and 0.8341 AUC with a 0.6813 target fake-call rate.
This is the first training-side version of the constrained source-heldout utility idea. It should be compared against the fixed capped threshold family at 0.7222 accuracy / 0.8291 AUC and the branch-dropout AUC frontier at 0.8406.
This improves on the fixed capped threshold family in both accuracy and AUC, while still leaving a relatively high target fake-call rate. The remaining issue is a fusion objective that preserves this held-out-generator gain while further reducing real-image false positives.

## Rebuild

```powershell
.\.venv\Scripts\python.exe scripts\tune_reverse_fusion_source_holdout.py
```
