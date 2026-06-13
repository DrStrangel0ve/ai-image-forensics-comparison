# MS COCOAI to Ishu Source-Utility Model Selection

This report tests the next SCP-Fusion question after threshold-only tuning: can source-side forensic utility choose among existing reverse fusion model families without looking at Ishu target labels?

Selection uses the train/source predictions already saved inside each run directory. Missing threshold columns are treated as the original fixed 0.5 operating point.

## Selection Policies

| selection_policy | n_seeds | target_accuracy_mean | target_roc_auc_mean | target_brier_score_mean | target_expected_calibration_error_mean | target_predicted_positive_rate_mean | source_utility_mean | source_predicted_positive_rate_mean | selected_candidates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| source_utility_cap_0p48 | 3 | 0.7193 | 0.8291 | 0.2188 | 0.2060 | 0.6199 | 1.7083 | 0.4617 | score_fusion_all6_c003_source_acc_cap_0p50; score_fusion_all6_c003_source_acc_cap_0p48; score_fusion_all6_c003_source_acc_cap_0p45 |
| source_utility_unconstrained | 3 | 0.6520 | 0.8275 | 0.2944 | 0.3181 | 0.8216 | 1.8083 | 0.4967 | score_fusion_all6; score_fusion_all6; score_fusion_all6 |
| source_utility_cap_0p5 | 3 | 0.6520 | 0.8275 | 0.2944 | 0.3181 | 0.8216 | 1.8083 | 0.4967 | score_fusion_all6; score_fusion_all6; score_fusion_all6 |

## Selected Runs

| selection_policy | seed | candidate | source_utility | source_predicted_positive_rate | target_accuracy | target_roc_auc | target_predicted_positive_rate | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| source_utility_cap_0p48 | 7 | score_fusion_all6_c003_source_acc_cap_0p50 | 1.7000 | 0.4550 | 0.7456 | 0.8254 | 0.6053 | 0.7028 |
| source_utility_cap_0p48 | 17 | score_fusion_all6_c003_source_acc_cap_0p48 | 1.7500 | 0.4500 | 0.6667 | 0.7826 | 0.6140 | 0.7029 |
| source_utility_cap_0p48 | 29 | score_fusion_all6_c003_source_acc_cap_0p45 | 1.6750 | 0.4800 | 0.7456 | 0.8793 | 0.6404 | 0.6223 |
| source_utility_cap_0p5 | 7 | score_fusion_all6 | 1.8250 | 0.4950 | 0.6316 | 0.8390 | 0.8596 | 0.5000 |
| source_utility_cap_0p5 | 17 | score_fusion_all6 | 1.8250 | 0.4950 | 0.6754 | 0.7783 | 0.7982 | 0.5000 |
| source_utility_cap_0p5 | 29 | score_fusion_all6 | 1.7750 | 0.5000 | 0.6491 | 0.8651 | 0.8070 | 0.5000 |
| source_utility_unconstrained | 7 | score_fusion_all6 | 1.8250 | 0.4950 | 0.6316 | 0.8390 | 0.8596 | 0.5000 |
| source_utility_unconstrained | 17 | score_fusion_all6 | 1.8250 | 0.4950 | 0.6754 | 0.7783 | 0.7982 | 0.5000 |
| source_utility_unconstrained | 29 | score_fusion_all6 | 1.7750 | 0.5000 | 0.6491 | 0.8651 | 0.8070 | 0.5000 |

## Candidate Frontier

| candidate | n_seeds | target_accuracy_mean | target_roc_auc_mean | target_brier_score_mean | target_expected_calibration_error_mean | target_predicted_positive_rate_mean | source_utility_mean | source_predicted_positive_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| score_fusion_all6_c003_source_acc_cap_0p48 | 3 | 0.7222 | 0.8291 | 0.2188 | 0.2060 | 0.6228 | 1.7083 | 0.4617 |
| score_fusion_all6_c003_source_utility_rfp2_fmp1_cap0p48 | 3 | 0.7222 | 0.8291 | 0.2188 | 0.2060 | 0.6228 | 1.7083 | 0.4617 |
| score_fusion_all6_c003_source_acc_cap_0p45 | 3 | 0.7193 | 0.8291 | 0.2188 | 0.2060 | 0.6140 | 1.6583 | 0.4517 |
| score_fusion_all6_c003_source_utility_rfp2_fmp1_cap0p45 | 3 | 0.7193 | 0.8291 | 0.2188 | 0.2060 | 0.6140 | 1.6583 | 0.4517 |
| score_fusion_all6_c003_source_acc_cap_0p50 | 3 | 0.7193 | 0.8291 | 0.2188 | 0.2060 | 0.6608 | 1.7167 | 0.4783 |
| score_fusion_all6_c003_source_utility_rfp2_fmp1_cap0p5 | 3 | 0.7193 | 0.8291 | 0.2188 | 0.2060 | 0.6608 | 1.7167 | 0.4783 |
| score_fusion_all6_c003_source_utility_rfp2_fmp1_nocap | 3 | 0.7193 | 0.8291 | 0.2188 | 0.2060 | 0.6608 | 1.7167 | 0.4783 |
| score_fusion_all6_c003 | 3 | 0.6608 | 0.8315 | 0.2213 | 0.2091 | 0.7895 | 1.6583 | 0.5017 |
| score_fusion_all6_c01_temp_balanced | 3 | 0.6579 | 0.8323 | 0.2794 | 0.2941 | 0.7924 | 1.6750 | 0.5000 |
| score_fusion_all6_temp_balanced | 3 | 0.6579 | 0.8285 | 0.3067 | 0.3304 | 0.8099 | 1.7833 | 0.5017 |
| score_fusion_all6_dropout_mean_r35x8_temp_balanced | 3 | 0.6520 | 0.8403 | 0.2939 | 0.3229 | 0.8099 | 1.7250 | 0.5050 |
| score_fusion_all6_dropout_mean_r35x8 | 3 | 0.6520 | 0.8406 | 0.3057 | 0.3275 | 0.8158 | 1.7583 | 0.5017 |

## Read

Unconstrained source-utility model selection reaches 0.6520 mean target accuracy with a 0.8216 target fake-call rate. It mostly selects the older high-recall fusion heads, which look excellent on source but over-call generated images after the MS-COCOAI-to-Ishu shift.
Adding a source fake-call cap of 0.48 recovers a much cleaner operating point at 0.7193 accuracy with a 0.6199 target fake-call rate, but it still does not beat the fixed capped threshold family.
The best target-labeled candidate in this suite is `score_fusion_all6_c003_source_acc_cap_0p48` at 0.7222 accuracy and 0.8291 AUC; that row is diagnostic only, because target labels are not allowed in model selection.
The useful negative result is that source-train utility alone is not a reliable model selector under generator/domain shift. The source fake-rate constraint remains the active ingredient, so the next version should use source-heldout generator splits or train-time utility regularization rather than selecting on the full source train set.

## Rebuild

```powershell
.\.venv\Scripts\python.exe scripts\select_reverse_fusion_by_source_utility.py
```
