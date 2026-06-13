# MS COCOAI to Ishu Tuned-Fusion jpeg50 Robustness

This evaluates the already-selected reverse tuned-fusion operating point on `jpeg50`-transformed Ishu target splits. The fusion configuration and source threshold are selected from clean MS source scores only; `jpeg50` labels are used only for final evaluation.

## Robustness Summary

| variant | variant_policy | n_seeds | target_accuracy_mean | target_roc_auc_mean | target_brier_score_mean | target_expected_calibration_error_mean | target_predicted_positive_rate_mean | source_predicted_positive_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| jpeg50 | cap_0p4 | 3 | 0.7515 | 0.8309 | 0.2486 | 0.2517 | 0.4240 | 0.4000 |

## Per-Seed Detail

| variant | seed | fusion_c | dropout_config | threshold | target_accuracy | target_roc_auc | target_predicted_positive_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| jpeg50 | 7 | 0.3000 | mean0p35x8 | 0.9862 | 0.7544 | 0.8424 | 0.4561 |
| jpeg50 | 17 | 0.0300 | mean0p35x8 | 0.9541 | 0.7193 | 0.7950 | 0.4386 |
| jpeg50 | 29 | 0.3000 | none | 0.9397 | 0.7807 | 0.8553 | 0.3772 |

## Clean Comparator

The clean `cap_0p4` tuned-fusion result was 0.7632 accuracy / 0.8361 AUC with a 0.5175 target fake-call rate.
Under `jpeg50`, the same source-selected policy reaches 0.7515 accuracy / 0.8309 AUC with a 0.4240 target fake-call rate.
This is a bounded robustness check for the tuned-fusion cap frontier. Interpret it together with the clean-target result and the other transformed target variants before treating the 0.40 cap as robust rather than clean-target optimized.

## Rebuild

Prerequisite branch prediction folders are expected under `runs\ms_cocoai_to_ishu_jpeg50_robustness/seed*/`. They are produced by evaluating the saved MS-trained `combined_v3`, ResNet-18, physics-guided, ConvNeXt, CLIP, and DINOv2 branches on the seed-specific Ishu `jpeg50` folders.

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_reverse_tuned_fusion_robustness.py --variant jpeg50
```
