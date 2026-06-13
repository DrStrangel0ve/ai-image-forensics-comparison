# MS COCOAI to Ishu Tuned-Fusion screenshot Robustness

This evaluates the already-selected reverse tuned-fusion operating point on `screenshot`-transformed Ishu target splits. The fusion configuration and source threshold are selected from clean MS source scores only; `screenshot` labels are used only for final evaluation.

## Robustness Summary

| variant | variant_policy | n_seeds | target_accuracy_mean | target_roc_auc_mean | target_brier_score_mean | target_expected_calibration_error_mean | target_predicted_positive_rate_mean | source_predicted_positive_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| screenshot | cap_0p4 | 3 | 0.7310 | 0.7965 | 0.3188 | 0.3392 | 0.5263 | 0.4000 |

## Per-Seed Detail

| variant | seed | fusion_c | dropout_config | threshold | target_accuracy | target_roc_auc | target_predicted_positive_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| screenshot | 7 | 0.3000 | mean0p35x8 | 0.9862 | 0.7281 | 0.8023 | 0.5526 |
| screenshot | 17 | 0.0300 | mean0p35x8 | 0.9541 | 0.7018 | 0.7445 | 0.5263 |
| screenshot | 29 | 0.3000 | none | 0.9397 | 0.7632 | 0.8427 | 0.5000 |

## Clean Comparator

The clean `cap_0p4` tuned-fusion result was 0.7632 accuracy / 0.8361 AUC with a 0.5175 target fake-call rate.
Under `screenshot`, the same source-selected policy reaches 0.7310 accuracy / 0.7965 AUC with a 0.5263 target fake-call rate.
This is a bounded robustness check for the tuned-fusion cap frontier. Interpret it together with the clean-target result and the other transformed target variants before treating the 0.40 cap as robust rather than clean-target optimized.

## Rebuild

Prerequisite branch prediction folders are expected under `runs\ms_cocoai_to_ishu_screenshot_robustness/seed*/`. They are produced by evaluating the saved MS-trained `combined_v3`, ResNet-18, physics-guided, ConvNeXt, CLIP, and DINOv2 branches on the seed-specific Ishu `screenshot` folders.

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_reverse_tuned_fusion_robustness.py --variant screenshot
```
