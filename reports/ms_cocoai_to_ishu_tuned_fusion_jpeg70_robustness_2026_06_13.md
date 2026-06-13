# MS COCOAI to Ishu Tuned-Fusion JPEG70 Robustness

This evaluates the already-selected reverse tuned-fusion operating point on JPEG70-transformed Ishu target splits. The fusion configuration and source threshold are selected from clean MS source scores only; JPEG70 labels are used only for final evaluation.

## Robustness Summary

| variant | variant_policy | n_seeds | target_accuracy_mean | target_roc_auc_mean | target_brier_score_mean | target_expected_calibration_error_mean | target_predicted_positive_rate_mean | source_predicted_positive_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| jpeg70 | cap_0p4 | 3 | 0.7661 | 0.8485 | 0.2599 | 0.2635 | 0.4678 | 0.4000 |

## Per-Seed Detail

| variant | seed | fusion_c | dropout_config | threshold | target_accuracy | target_roc_auc | target_predicted_positive_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| jpeg70 | 7 | 0.3000 | mean0p35x8 | 0.9862 | 0.7544 | 0.8541 | 0.4912 |
| jpeg70 | 17 | 0.0300 | mean0p35x8 | 0.9541 | 0.7368 | 0.8042 | 0.4912 |
| jpeg70 | 29 | 0.3000 | none | 0.9397 | 0.8070 | 0.8873 | 0.4211 |

## Clean Comparator

The clean `cap_0p4` tuned-fusion result was 0.7632 accuracy / 0.8361 AUC with a 0.5175 target fake-call rate.
Under JPEG70, the same source-selected policy reaches 0.7661 accuracy / 0.8485 AUC with a 0.4678 target fake-call rate.
This is a bounded first robustness check for the tuned-fusion cap frontier. The next check should add blur, resize, and crop variants before treating the 0.40 cap as robust rather than clean-target optimized.

## Rebuild

Prerequisite branch prediction folders are expected under `runs/ms_cocoai_to_ishu_jpeg70_robustness/seed*/`. They are produced by evaluating the saved MS-trained `combined_v3`, ResNet-18, physics-guided, ConvNeXt, CLIP, and DINOv2 branches on the seed-specific Ishu JPEG70 folders.

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_reverse_tuned_fusion_robustness.py
```
