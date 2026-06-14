# MS COCOAI to Ishu Source-Holdout Model Selection

This report repeats reverse SCP-Fusion model selection with a stricter source-side evaluator: each MS COCOAI generated source label is held out in turn, scored against all real images, and aggregated before choosing a fusion family.

The target Ishu labels are used only after source-side selection has chosen a run for each seed.

## Selection Policies

| selection_policy | n_seeds | target_accuracy_mean | target_roc_auc_mean | target_brier_score_mean | target_expected_calibration_error_mean | target_predicted_positive_rate_mean | source_holdout_utility_mean_mean | source_predicted_positive_rate_mean | selected_candidates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| source_holdout_mean_utility_cap_0p48 | 3 | 0.7193 | 0.8291 | 0.2188 | 0.2060 | 0.6199 | 1.7049 | 0.4617 | score_fusion_all6_c003_source_acc_cap_0p50; score_fusion_all6_c003_source_acc_cap_0p48; score_fusion_all6_c003_source_acc_cap_0p45 |
| source_holdout_min_utility_cap_0p48 | 3 | 0.7193 | 0.8291 | 0.2188 | 0.2060 | 0.6199 | 1.6977 | 0.4600 | score_fusion_all6_c003_source_acc_cap_0p50; score_fusion_all6_c003_source_acc_cap_0p45; score_fusion_all6_c003_source_acc_cap_0p45 |
| source_holdout_mean_utility_unconstrained | 3 | 0.6520 | 0.8275 | 0.2944 | 0.3181 | 0.8216 | 1.8122 | 0.4967 | score_fusion_all6; score_fusion_all6; score_fusion_all6 |
| source_holdout_mean_utility_cap_0p5 | 3 | 0.6520 | 0.8275 | 0.2944 | 0.3181 | 0.8216 | 1.8122 | 0.4967 | score_fusion_all6; score_fusion_all6; score_fusion_all6 |
| source_holdout_min_utility_unconstrained | 3 | 0.6520 | 0.8275 | 0.2944 | 0.3181 | 0.8216 | 1.8122 | 0.4967 | score_fusion_all6; score_fusion_all6; score_fusion_all6 |
| source_holdout_min_utility_cap_0p5 | 3 | 0.6520 | 0.8275 | 0.2944 | 0.3181 | 0.8216 | 1.8122 | 0.4967 | score_fusion_all6; score_fusion_all6; score_fusion_all6 |

## Selected Runs

| selection_policy | seed | candidate | selection_score | source_holdout_utility_mean | source_holdout_utility_min | source_predicted_positive_rate | target_accuracy | target_roc_auc | target_predicted_positive_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| source_holdout_mean_utility_cap_0p48 | 7 | score_fusion_all6_c003_source_acc_cap_0p50 | 1.6843 | 1.6843 | 1.5088 | 0.4550 | 0.7456 | 0.8254 | 0.6053 |
| source_holdout_mean_utility_cap_0p48 | 17 | score_fusion_all6_c003_source_acc_cap_0p48 | 1.7537 | 1.7537 | 1.4118 | 0.4500 | 0.6667 | 0.7826 | 0.6140 |
| source_holdout_mean_utility_cap_0p48 | 29 | score_fusion_all6_c003_source_acc_cap_0p45 | 1.6768 | 1.6768 | 1.3500 | 0.4800 | 0.7456 | 0.8793 | 0.6404 |
| source_holdout_mean_utility_cap_0p5 | 7 | score_fusion_all6 | 1.8289 | 1.8289 | 1.6917 | 0.4950 | 0.6316 | 0.8390 | 0.8596 |
| source_holdout_mean_utility_cap_0p5 | 17 | score_fusion_all6 | 1.8280 | 1.8280 | 1.7529 | 0.4950 | 0.6754 | 0.7783 | 0.7982 |
| source_holdout_mean_utility_cap_0p5 | 29 | score_fusion_all6 | 1.7795 | 1.7795 | 1.7250 | 0.5000 | 0.6491 | 0.8651 | 0.8070 |
| source_holdout_mean_utility_unconstrained | 7 | score_fusion_all6 | 1.8289 | 1.8289 | 1.6917 | 0.4950 | 0.6316 | 0.8390 | 0.8596 |
| source_holdout_mean_utility_unconstrained | 17 | score_fusion_all6 | 1.8280 | 1.8280 | 1.7529 | 0.4950 | 0.6754 | 0.7783 | 0.7982 |
| source_holdout_mean_utility_unconstrained | 29 | score_fusion_all6 | 1.7795 | 1.7795 | 1.7250 | 0.5000 | 0.6491 | 0.8651 | 0.8070 |
| source_holdout_min_utility_cap_0p48 | 7 | score_fusion_all6_c003_source_acc_cap_0p50 | 1.5088 | 1.6843 | 1.5088 | 0.4550 | 0.7456 | 0.8254 | 0.6053 |
| source_holdout_min_utility_cap_0p48 | 17 | score_fusion_all6_c003_source_acc_cap_0p45 | 1.4118 | 1.7320 | 1.4118 | 0.4450 | 0.6667 | 0.7826 | 0.6140 |
| source_holdout_min_utility_cap_0p48 | 29 | score_fusion_all6_c003_source_acc_cap_0p45 | 1.3500 | 1.6768 | 1.3500 | 0.4800 | 0.7456 | 0.8793 | 0.6404 |
| source_holdout_min_utility_cap_0p5 | 7 | score_fusion_all6 | 1.6917 | 1.8289 | 1.6917 | 0.4950 | 0.6316 | 0.8390 | 0.8596 |
| source_holdout_min_utility_cap_0p5 | 17 | score_fusion_all6 | 1.7529 | 1.8280 | 1.7529 | 0.4950 | 0.6754 | 0.7783 | 0.7982 |
| source_holdout_min_utility_cap_0p5 | 29 | score_fusion_all6 | 1.7250 | 1.7795 | 1.7250 | 0.5000 | 0.6491 | 0.8651 | 0.8070 |
| source_holdout_min_utility_unconstrained | 7 | score_fusion_all6 | 1.6917 | 1.8289 | 1.6917 | 0.4950 | 0.6316 | 0.8390 | 0.8596 |
| source_holdout_min_utility_unconstrained | 17 | score_fusion_all6 | 1.7529 | 1.8280 | 1.7529 | 0.4950 | 0.6754 | 0.7783 | 0.7982 |
| source_holdout_min_utility_unconstrained | 29 | score_fusion_all6 | 1.7250 | 1.7795 | 1.7250 | 0.5000 | 0.6491 | 0.8651 | 0.8070 |

## Candidate Source-Holdout Frontier

| candidate | n_seeds | source_holdout_utility_mean_mean | source_holdout_utility_min_mean | source_predicted_positive_rate_mean | target_accuracy_mean | target_roc_auc_mean | target_predicted_positive_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| score_fusion_all6 | 3 | 1.8122 | 1.7232 | 0.4967 | 0.6520 | 0.8275 | 0.8216 |
| score_fusion_all6_temp_balanced | 3 | 1.7886 | 1.6899 | 0.5017 | 0.6579 | 0.8285 | 0.8099 |
| score_fusion_all6_dropout_mean_r35x8 | 3 | 1.7622 | 1.6732 | 0.5017 | 0.6520 | 0.8406 | 0.8158 |
| score_fusion_all6_c01 | 3 | 1.7357 | 1.6279 | 0.5017 | 0.6491 | 0.8366 | 0.8129 |
| score_fusion_all6_dropout_mean_r35x8_temp_balanced | 3 | 1.7288 | 1.6399 | 0.5050 | 0.6520 | 0.8403 | 0.8099 |
| score_fusion_all6_c003_source_acc_cap_0p50 | 3 | 1.7115 | 1.5132 | 0.4783 | 0.7193 | 0.8291 | 0.6608 |
| score_fusion_all6_c003_source_utility_rfp2_fmp1_cap0p5 | 3 | 1.7115 | 1.5132 | 0.4783 | 0.7193 | 0.8291 | 0.6608 |
| score_fusion_all6_c003_source_utility_rfp2_fmp1_nocap | 3 | 1.7115 | 1.5132 | 0.4783 | 0.7193 | 0.8291 | 0.6608 |
| score_fusion_all6_c003_source_acc_cap_0p48 | 3 | 1.7014 | 1.4337 | 0.4617 | 0.7222 | 0.8291 | 0.6228 |
| score_fusion_all6_c003_source_utility_rfp2_fmp1_cap0p48 | 3 | 1.7014 | 1.4337 | 0.4617 | 0.7222 | 0.8291 | 0.6228 |
| score_fusion_all6_c01_temp_balanced | 3 | 1.6764 | 1.5113 | 0.5000 | 0.6579 | 0.8323 | 0.7924 |
| score_fusion_all6_c003 | 3 | 1.6598 | 1.4946 | 0.5017 | 0.6608 | 0.8315 | 0.7895 |

## Held-Out Generator Stress

The table below keeps the selected candidate for each seed fixed, then groups those selected folds by held-out generator. Lower utility and higher fake-miss rate mark the source family that stresses the policy.

| selection_policy | heldout_source_name | n_seeds | source_holdout_utility_mean | source_holdout_utility_min | source_holdout_accuracy_mean | source_holdout_recall_mean | source_holdout_fake_miss_rate_mean | source_holdout_predicted_positive_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| source_holdout_mean_utility_cap_0p48 | sd3 | 3 | 1.4235 | 1.3500 | 0.9578 | 0.7961 | 0.2039 | 0.1325 |
| source_holdout_mean_utility_cap_0p48 | midjourney6 | 3 | 1.6315 | 1.5652 | 0.9695 | 0.8793 | 0.1207 | 0.1551 |
| source_holdout_mean_utility_cap_0p48 | sd21 | 3 | 1.6787 | 1.5333 | 0.9699 | 0.8981 | 0.1019 | 0.1721 |
| source_holdout_mean_utility_cap_0p48 | sdxl | 3 | 1.8955 | 1.8364 | 0.9861 | 0.9848 | 0.0152 | 0.1818 |
| source_holdout_mean_utility_cap_0p48 | dalle3 | 3 | 1.8955 | 1.7364 | 0.9864 | 0.9848 | 0.0152 | 0.1690 |
| source_holdout_mean_utility_cap_0p5 | sd3 | 3 | 1.7436 | 1.7250 | 0.9718 | 0.9441 | 0.0559 | 0.1637 |
| source_holdout_mean_utility_cap_0p5 | sd21 | 3 | 1.7792 | 1.6917 | 0.9727 | 0.9583 | 0.0417 | 0.1910 |
| source_holdout_mean_utility_cap_0p5 | midjourney6 | 3 | 1.8092 | 1.7364 | 0.9751 | 0.9704 | 0.0296 | 0.1774 |
| source_holdout_mean_utility_cap_0p5 | dalle3 | 3 | 1.8455 | 1.7364 | 0.9778 | 0.9848 | 0.0152 | 0.1776 |
| source_holdout_mean_utility_cap_0p5 | sdxl | 3 | 1.8833 | 1.8500 | 0.9806 | 1.0000 | 0.0000 | 0.1926 |
| source_holdout_mean_utility_unconstrained | sd3 | 3 | 1.7436 | 1.7250 | 0.9718 | 0.9441 | 0.0559 | 0.1637 |
| source_holdout_mean_utility_unconstrained | sd21 | 3 | 1.7792 | 1.6917 | 0.9727 | 0.9583 | 0.0417 | 0.1910 |
| source_holdout_mean_utility_unconstrained | midjourney6 | 3 | 1.8092 | 1.7364 | 0.9751 | 0.9704 | 0.0296 | 0.1774 |
| source_holdout_mean_utility_unconstrained | dalle3 | 3 | 1.8455 | 1.7364 | 0.9778 | 0.9848 | 0.0152 | 0.1776 |
| source_holdout_mean_utility_unconstrained | sdxl | 3 | 1.8833 | 1.8500 | 0.9806 | 1.0000 | 0.0000 | 0.1926 |
| source_holdout_min_utility_cap_0p48 | sd3 | 3 | 1.4235 | 1.3500 | 0.9578 | 0.7961 | 0.2039 | 0.1325 |
| source_holdout_min_utility_cap_0p48 | midjourney6 | 3 | 1.5952 | 1.4565 | 0.9667 | 0.8648 | 0.1352 | 0.1524 |
| source_holdout_min_utility_cap_0p48 | sd21 | 3 | 1.6787 | 1.5333 | 0.9699 | 0.8981 | 0.1019 | 0.1721 |
| source_holdout_min_utility_cap_0p48 | sdxl | 3 | 1.8955 | 1.8364 | 0.9861 | 0.9848 | 0.0152 | 0.1818 |
| source_holdout_min_utility_cap_0p48 | dalle3 | 3 | 1.8955 | 1.7364 | 0.9864 | 0.9848 | 0.0152 | 0.1690 |
| source_holdout_min_utility_cap_0p5 | sd3 | 3 | 1.7436 | 1.7250 | 0.9718 | 0.9441 | 0.0559 | 0.1637 |
| source_holdout_min_utility_cap_0p5 | sd21 | 3 | 1.7792 | 1.6917 | 0.9727 | 0.9583 | 0.0417 | 0.1910 |
| source_holdout_min_utility_cap_0p5 | midjourney6 | 3 | 1.8092 | 1.7364 | 0.9751 | 0.9704 | 0.0296 | 0.1774 |
| source_holdout_min_utility_cap_0p5 | dalle3 | 3 | 1.8455 | 1.7364 | 0.9778 | 0.9848 | 0.0152 | 0.1776 |
| source_holdout_min_utility_cap_0p5 | sdxl | 3 | 1.8833 | 1.8500 | 0.9806 | 1.0000 | 0.0000 | 0.1926 |
| source_holdout_min_utility_unconstrained | sd3 | 3 | 1.7436 | 1.7250 | 0.9718 | 0.9441 | 0.0559 | 0.1637 |
| source_holdout_min_utility_unconstrained | sd21 | 3 | 1.7792 | 1.6917 | 0.9727 | 0.9583 | 0.0417 | 0.1910 |
| source_holdout_min_utility_unconstrained | midjourney6 | 3 | 1.8092 | 1.7364 | 0.9751 | 0.9704 | 0.0296 | 0.1774 |
| source_holdout_min_utility_unconstrained | dalle3 | 3 | 1.8455 | 1.7364 | 0.9778 | 0.9848 | 0.0152 | 0.1776 |
| source_holdout_min_utility_unconstrained | sdxl | 3 | 1.8833 | 1.8500 | 0.9806 | 1.0000 | 0.0000 | 0.1926 |

## Read

Leave-one-generator-out source utility without a fake-rate cap still selects over-firing fusion heads, reaching 0.6520 target accuracy with a 0.8216 target fake-call rate.
Adding the 0.48 source fake-rate cap recovers 0.7193 accuracy and lowers the target fake-call rate to 0.6199, but it still does not exceed the fixed capped source-threshold family.
For the paper-facing `source_holdout_mean_utility_cap_0p48` policy, the weakest held-out generator is `sd3` with mean utility 1.4235, mean recall 0.7961, and mean fake-miss rate 0.2039.
The best target-labeled candidate remains `score_fusion_all6_c003_source_acc_cap_0p48` at 0.7222 accuracy, but target labels are not allowed for model selection.
The paper-facing conclusion is now sharper: source-heldout generator utility is necessary as a diagnostic, but not sufficient as a selector unless it also controls the source fake-call rate. SCP-Fusion v1 should therefore train or select against held-out-generator utility with an explicit real-image false-positive/fake-rate constraint.

## Rebuild

```powershell
.\.venv\Scripts\python.exe scripts\select_reverse_fusion_by_source_holdout.py
```
