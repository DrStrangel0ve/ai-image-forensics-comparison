# MS COCOAI to Ishu Tuned-Fusion jpeg30 Robustness

This evaluates the already-selected reverse tuned-fusion operating point on `jpeg30`-transformed Ishu target splits. The fusion configuration and source threshold are selected from clean MS source scores only; `jpeg30` labels are used only for final evaluation.

## Robustness Summary

| variant | variant_policy | n_seeds | target_accuracy_mean | target_roc_auc_mean | target_brier_score_mean | target_expected_calibration_error_mean | target_predicted_positive_rate_mean | source_predicted_positive_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| jpeg30 | cap_0p4 | 3 | 0.7076 | 0.8167 | 0.2332 | 0.2215 | 0.3450 | 0.4000 |

## Per-Seed Detail

| variant | seed | fusion_c | dropout_config | threshold | target_accuracy | target_roc_auc | target_predicted_positive_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| jpeg30 | 7 | 0.3000 | mean0p35x8 | 0.9862 | 0.7105 | 0.8374 | 0.4123 |
| jpeg30 | 17 | 0.0300 | mean0p35x8 | 0.9541 | 0.6842 | 0.7666 | 0.3509 |
| jpeg30 | 29 | 0.3000 | none | 0.9397 | 0.7281 | 0.8461 | 0.2719 |

## Clean Comparator

The clean `cap_0p4` tuned-fusion result was 0.7632 accuracy / 0.8361 AUC with a 0.5175 target fake-call rate.
Under `jpeg30`, the same source-selected policy reaches 0.7076 accuracy / 0.8167 AUC with a 0.3450 target fake-call rate.
This is a bounded robustness check for the tuned-fusion cap frontier. Interpret it together with the clean-target result and the other transformed target variants before treating the 0.40 cap as robust rather than clean-target optimized.

## Rebuild

Prerequisite branch prediction folders are expected under `runs\ms_cocoai_to_ishu_jpeg30_robustness/seed*/`. They are produced by evaluating the saved MS-trained `combined_v3`, ResNet-18, physics-guided, ConvNeXt, CLIP, and DINOv2 branches on the seed-specific Ishu `jpeg30` folders.

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_reverse_tuned_fusion_robustness.py --variant jpeg30
```
