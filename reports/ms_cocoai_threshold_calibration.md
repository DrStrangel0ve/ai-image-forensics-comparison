# Defactify / MS COCOAI Threshold Calibration

Run date: 2026-06-12

This follow-up analyzes whether the robustness losses in `reports/ms_cocoai_robustness_variants.md` are caused by weaker ranking signal or by the default `0.5` decision threshold becoming miscalibrated after transformations.

For each method, the clean threshold is selected on the clean source-balanced validation predictions by maximizing accuracy. The oracle threshold is selected on each transformed variant and is diagnostic only.

## Results

| method | variant | default accuracy | clean threshold | clean-threshold accuracy | oracle threshold | oracle accuracy | roc_auc |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| combined_v3 | clean | 0.7320 | 0.5873 | 0.7430 | 0.5873 | 0.7430 | 0.8027 |
| combined_v3 | blur1 | 0.6350 | 0.5873 | 0.6740 | 0.6809 | 0.7030 | 0.7665 |
| combined_v3 | crop85 | 0.7140 | 0.5873 | 0.7310 | 0.6322 | 0.7330 | 0.7977 |
| combined_v3 | jpeg70 | 0.7340 | 0.5873 | 0.7360 | 0.5333 | 0.7400 | 0.7994 |
| combined_v3 | resize_half | 0.6940 | 0.5873 | 0.7170 | 0.6265 | 0.7290 | 0.7920 |
| resnet18 | clean | 0.8160 | 0.5467 | 0.8190 | 0.5467 | 0.8190 | 0.8967 |
| resnet18 | blur1 | 0.7990 | 0.5467 | 0.8030 | 0.6712 | 0.8120 | 0.8879 |
| resnet18 | crop85 | 0.8100 | 0.5467 | 0.8100 | 0.3515 | 0.8130 | 0.8938 |
| resnet18 | jpeg70 | 0.8170 | 0.5467 | 0.8150 | 0.4986 | 0.8180 | 0.8974 |
| resnet18 | resize_half | 0.8110 | 0.5467 | 0.8120 | 0.8463 | 0.8160 | 0.8938 |

## Interpretation

Clean threshold calibration gives `combined_v3` a small clean-validation lift from 0.7320 to 0.7430 accuracy. It also recovers some robustness loss: under blur, `combined_v3` improves from 0.6350 at the default threshold to 0.6740 with the clean-calibrated threshold, and an oracle threshold reaches 0.7030.

That means the blur weakness is both calibration and ranking. The AUC under blur is still 0.7665, so the conventional signal has not collapsed, but the transformed score distribution shifts enough that the default threshold is too permissive.

ResNet-18 is much less threshold-sensitive in these runs. Clean thresholding moves most transformed ResNet accuracies by 0.0 to 0.4 points, with oracle thresholds adding at most about 1.3 points.

## Reproduce

```powershell
python scripts/summarize_threshold_calibration.py `
  --out-dir runs/ms_cocoai_2026_robustness_eval/threshold_calibration `
  --clean combined_v3=runs/ms_cocoai_2026_source_balanced_eval/combined_v3/predictions.csv `
  --clean resnet18=runs/ms_cocoai_2026_source_balanced_eval/resnet18/predictions.csv `
  --variant jpeg70:combined_v3=runs/ms_cocoai_2026_robustness_eval/jpeg70/combined_v3/predictions.csv `
  --variant jpeg70:resnet18=runs/ms_cocoai_2026_robustness_eval/jpeg70/resnet18/predictions.csv `
  --variant blur1:combined_v3=runs/ms_cocoai_2026_robustness_eval/blur1/combined_v3/predictions.csv `
  --variant blur1:resnet18=runs/ms_cocoai_2026_robustness_eval/blur1/resnet18/predictions.csv `
  --variant resize_half:combined_v3=runs/ms_cocoai_2026_robustness_eval/resize_half/combined_v3/predictions.csv `
  --variant resize_half:resnet18=runs/ms_cocoai_2026_robustness_eval/resize_half/resnet18/predictions.csv `
  --variant crop85:combined_v3=runs/ms_cocoai_2026_robustness_eval/crop85/combined_v3/predictions.csv `
  --variant crop85:resnet18=runs/ms_cocoai_2026_robustness_eval/crop85/resnet18/predictions.csv
```
