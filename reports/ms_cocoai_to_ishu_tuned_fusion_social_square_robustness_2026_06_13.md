# MS COCOAI to Ishu Tuned-Fusion social_square Robustness

This evaluates the already-selected reverse tuned-fusion operating point on `social_square`-transformed Ishu target splits. The fusion configuration and source threshold are selected from clean MS source scores only; `social_square` labels are used only for final evaluation.

## Robustness Summary

| variant | variant_policy | n_seeds | target_accuracy_mean | target_roc_auc_mean | target_brier_score_mean | target_expected_calibration_error_mean | target_predicted_positive_rate_mean | source_predicted_positive_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| social_square | cap_0p4 | 3 | 0.7778 | 0.8474 | 0.2782 | 0.3010 | 0.5088 | 0.4000 |

## Per-Seed Detail

| variant | seed | fusion_c | dropout_config | threshold | target_accuracy | target_roc_auc | target_predicted_positive_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| social_square | 7 | 0.3000 | mean0p35x8 | 0.9862 | 0.7368 | 0.8550 | 0.5263 |
| social_square | 17 | 0.0300 | mean0p35x8 | 0.9541 | 0.7544 | 0.8076 | 0.5088 |
| social_square | 29 | 0.3000 | none | 0.9397 | 0.8421 | 0.8796 | 0.4912 |

## Clean Comparator

The clean `cap_0p4` tuned-fusion result was 0.7632 accuracy / 0.8361 AUC with a 0.5175 target fake-call rate.
Under `social_square`, the same source-selected policy reaches 0.7778 accuracy / 0.8474 AUC with a 0.5088 target fake-call rate.
This is a bounded robustness check for the tuned-fusion cap frontier. Interpret it together with the clean-target result and the other transformed target variants before treating the 0.40 cap as robust rather than clean-target optimized.

## Rebuild

Prerequisite branch prediction folders are expected under `runs\ms_cocoai_to_ishu_social_square_robustness/seed*/`. They are produced by evaluating the saved MS-trained `combined_v3`, ResNet-18, physics-guided, ConvNeXt, CLIP, and DINOv2 branches on the seed-specific Ishu `social_square` folders.

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_reverse_tuned_fusion_robustness.py --variant social_square
```
