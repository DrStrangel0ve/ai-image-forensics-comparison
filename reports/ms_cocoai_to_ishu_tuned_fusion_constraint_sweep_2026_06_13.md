# MS COCOAI to Ishu Tuned-Fusion Constraint Sweep

This report sweeps the source fake-call cap used by source-heldout tuned fusion. Each cap is selected independently by worst-source utility, then retrained on all source rows and evaluated on Ishu.

## Constraint Summary

| constraint_policy | n_seeds | target_accuracy_mean | target_roc_auc_mean | target_brier_score_mean | target_expected_calibration_error_mean | target_predicted_positive_rate_mean | source_predicted_positive_rate_mean | selected_configs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cap_0p4 | 3 | 0.7632 | 0.8361 | 0.2851 | 0.3042 | 0.5175 | 0.4000 | seed7:C0.3:mean0p35x8:cap0.4; seed17:C0.03:mean0p35x8:cap0.4; seed29:C0.3:none:cap0.4 |
| cap_0p42 | 3 | 0.7515 | 0.8294 | 0.2269 | 0.2072 | 0.5292 | 0.4200 | seed7:C0.01:none:cap0.42; seed17:C0.01:mean0p35x8:cap0.42; seed29:C0.01:none:cap0.42 |
| cap_0p45 | 3 | 0.7339 | 0.8326 | 0.2516 | 0.2620 | 0.6111 | 0.4500 | seed7:C0.01:mean0p35x8:cap0.45; seed17:C0.01:mean0p35x8:cap0.45; seed29:C0.01:mean0p35x8:cap0.45 |
| cap_0p48 | 3 | 0.7339 | 0.8341 | 0.2748 | 0.2954 | 0.6813 | 0.4700 | seed7:C1:none:cap0.48; seed17:C1:none:cap0.48; seed29:C0.01:mean0p35x8:cap0.48 |
| cap_0p5 | 3 | 0.7076 | 0.8335 | 0.2740 | 0.2667 | 0.6842 | 0.4700 | seed7:C1:none:cap0.5; seed17:C0.01:none:cap0.5; seed29:C1:mean0p35x8:cap0.5 |

## Selected Runs

| constraint_policy | seed | fusion_c | dropout_config | source_fake_rate_cap | target_accuracy | target_roc_auc | target_predicted_positive_rate | threshold_source_predicted_positive_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cap_0p4 | 7 | 0.3000 | mean0p35x8 | 0.4000 | 0.7632 | 0.8381 | 0.5702 | 0.4000 |
| cap_0p4 | 17 | 0.0300 | mean0p35x8 | 0.4000 | 0.7105 | 0.7873 | 0.5175 | 0.4000 |
| cap_0p4 | 29 | 0.3000 | none | 0.4000 | 0.8158 | 0.8830 | 0.4649 | 0.4000 |
| cap_0p42 | 7 | 0.0100 | none | 0.4200 | 0.7456 | 0.8224 | 0.5526 | 0.4200 |
| cap_0p42 | 17 | 0.0100 | mean0p35x8 | 0.4200 | 0.6930 | 0.7839 | 0.5526 | 0.4200 |
| cap_0p42 | 29 | 0.0100 | none | 0.4200 | 0.8158 | 0.8821 | 0.4825 | 0.4200 |
| cap_0p45 | 7 | 0.0100 | mean0p35x8 | 0.4500 | 0.7544 | 0.8291 | 0.6140 | 0.4500 |
| cap_0p45 | 17 | 0.0100 | mean0p35x8 | 0.4500 | 0.6667 | 0.7839 | 0.6140 | 0.4500 |
| cap_0p45 | 29 | 0.0100 | mean0p35x8 | 0.4500 | 0.7807 | 0.8849 | 0.6053 | 0.4500 |
| cap_0p48 | 7 | 1.0000 | none | 0.4800 | 0.7632 | 0.8390 | 0.7281 | 0.4800 |
| cap_0p48 | 17 | 1.0000 | none | 0.4800 | 0.6579 | 0.7783 | 0.7105 | 0.4750 |
| cap_0p48 | 29 | 0.0100 | mean0p35x8 | 0.4800 | 0.7807 | 0.8849 | 0.6053 | 0.4550 |
| cap_0p5 | 7 | 1.0000 | none | 0.5000 | 0.6754 | 0.8390 | 0.8158 | 0.4850 |
| cap_0p5 | 17 | 0.0100 | none | 0.5000 | 0.6667 | 0.7740 | 0.6316 | 0.4700 |
| cap_0p5 | 29 | 1.0000 | mean0p35x8 | 0.5000 | 0.7807 | 0.8876 | 0.6053 | 0.4550 |

## Grid Frontier

| seed | fusion_c | dropout_config | source_fake_rate_cap | validation_utility_mean | validation_utility_min | validation_accuracy_mean | validation_predicted_positive_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 7 | 1.0000 | none | 0.5000 | 1.4687 | 1.0176 | 0.9394 | 0.2515 |
| 7 | 0.3000 | none | 0.5000 | 1.3677 | 1.0176 | 0.9280 | 0.2402 |
| 29 | 1.0000 | mean0p35x8 | 0.5000 | 1.5525 | 1.0000 | 0.9487 | 0.2340 |
| 17 | 0.0100 | none | 0.5000 | 1.4820 | 0.9130 | 0.9384 | 0.2268 |
| 17 | 0.3000 | none | 0.5000 | 1.4726 | 0.9130 | 0.9382 | 0.2211 |
| 17 | 0.1000 | none | 0.5000 | 1.4726 | 0.9130 | 0.9382 | 0.2211 |
| 29 | 0.3000 | mean0p35x8 | 0.5000 | 1.5275 | 0.8750 | 0.9459 | 0.2311 |
| 29 | 0.1000 | none | 0.5000 | 1.5275 | 0.8750 | 0.9459 | 0.2311 |
| 29 | 0.1000 | mean0p35x8 | 0.5000 | 1.5275 | 0.8750 | 0.9459 | 0.2311 |
| 29 | 0.0300 | none | 0.5000 | 1.5275 | 0.8750 | 0.9459 | 0.2311 |
| 29 | 0.0300 | mean0p35x8 | 0.5000 | 1.5275 | 0.8750 | 0.9459 | 0.2311 |
| 29 | 0.0100 | none | 0.5000 | 1.5275 | 0.8750 | 0.9459 | 0.2311 |

## Read

The best accuracy/AUC constraint is `cap_0p4` at 0.7632 accuracy / 0.8361 AUC, with a 0.5175 target fake-call rate.
The lowest target fake-call policy is `cap_0p4` at 0.5175, with 0.7632 accuracy.
The frontier is therefore useful for the paper: source caps reduce source fake calls, but target fake-call bias does not move monotonically with the source cap under this small validation set.

## Rebuild

```powershell
.\.venv\Scripts\python.exe scripts\sweep_reverse_tuned_fusion_constraints.py
```
