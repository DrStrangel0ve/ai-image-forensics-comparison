# Frozen Encoder Baseline

Run date: 2026-06-12

This report adds the first SCP-Fusion foundation-style baseline: a frozen pretrained ConvNeXt-Tiny image encoder plus logistic regression. It is not CLIP or DINO yet, but it tests the same broad hypothesis from the research deep dive: pretrained representation quality may matter more than training a small detector end-to-end on a narrow real/fake split.

Implementation:

- `scripts/run_frozen_encoder_baseline.py`
- `scripts/evaluate_frozen_encoder_model.py`
- encoder: `convnext_tiny`
- weights: `ConvNeXt_Tiny_Weights.IMAGENET1K_V1`
- classifier: balanced logistic regression on frozen embeddings
- image size: 224
- hardware: local RTX 3060 Ti

## Ishu In-Dataset Results

| seed | method | accuracy | AUC |
| ---: | --- | ---: | ---: |
| 7 | `combined_v3` | 0.8158 | 0.8938 |
| 7 | ResNet-18 | 0.7719 | 0.8608 |
| 7 | physics-guided ResNet-18 + `combined_v3` | 0.7982 | 0.8808 |
| 7 | frozen ConvNeXt-Tiny | 0.8947 | 0.9554 |
| 17 | `combined_v3` | 0.8246 | 0.9089 |
| 17 | ResNet-18 | 0.8509 | 0.8981 |
| 17 | physics-guided ResNet-18 + `combined_v3` | 0.8596 | 0.9372 |
| 17 | frozen ConvNeXt-Tiny | 0.9035 | 0.9646 |
| 29 | `combined_v3` | 0.8333 | 0.8799 |
| 29 | ResNet-18 | 0.8509 | 0.9190 |
| 29 | physics-guided ResNet-18 + `combined_v3` | 0.8772 | 0.9350 |
| 29 | frozen ConvNeXt-Tiny | 0.8860 | 0.9569 |

Three-seed mean:

| method | mean accuracy | mean AUC |
| --- | ---: | ---: |
| `combined_v3` | 0.8246 | 0.8942 |
| ResNet-18 | 0.8246 | 0.8926 |
| physics-guided ResNet-18 + `combined_v3` | 0.8450 | 0.9177 |
| frozen ConvNeXt-Tiny | 0.8947 | 0.9589 |

The frozen ConvNeXt baseline is now the strongest same-domain model on Ishu by a large margin. This is a useful publication result because it reframes the earlier physics-vs-neural comparison: the neural baseline should include stronger pretrained encoders, not only six-epoch ResNet-18 fine-tuning.

## Ishu to MS COCOAI Transfer

The same three Ishu-trained ConvNeXt classifiers were evaluated on the 1,000-image source-balanced Defactify/MS COCOAI validation slice.

| seed | method | default accuracy | AUC | precision | recall | F1 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 7 | `combined_v3` | 0.5430 | 0.5750 | 0.6201 | 0.2220 | 0.3270 |
| 7 | ResNet-18 | 0.5820 | 0.6593 | 0.6602 | 0.3380 | 0.4471 |
| 7 | physics-guided fusion | 0.6450 | 0.6681 | 0.6799 | 0.5480 | 0.6069 |
| 7 | frozen ConvNeXt-Tiny | 0.6240 | 0.7234 | 0.8370 | 0.3080 | 0.4503 |
| 17 | `combined_v3` | 0.5540 | 0.5818 | 0.6467 | 0.2380 | 0.3480 |
| 17 | ResNet-18 | 0.5880 | 0.6854 | 0.7366 | 0.2740 | 0.3994 |
| 17 | physics-guided fusion | 0.5400 | 0.6306 | 0.6515 | 0.1720 | 0.2722 |
| 17 | frozen ConvNeXt-Tiny | 0.6160 | 0.7059 | 0.7871 | 0.3180 | 0.4530 |
| 29 | `combined_v3` | 0.5430 | 0.5840 | 0.6352 | 0.2020 | 0.3065 |
| 29 | ResNet-18 | 0.5700 | 0.6016 | 0.6190 | 0.3640 | 0.4584 |
| 29 | physics-guided fusion | 0.6330 | 0.6923 | 0.7401 | 0.4100 | 0.5277 |
| 29 | frozen ConvNeXt-Tiny | 0.6090 | 0.7124 | 0.8114 | 0.2840 | 0.4207 |

Three-seed mean:

| method | default accuracy | AUC |
| --- | ---: | ---: |
| `combined_v3` | 0.5467 | 0.5803 |
| ResNet-18 | 0.5800 | 0.6488 |
| physics-guided fusion | 0.6060 | 0.6637 |
| frozen ConvNeXt-Tiny | 0.6163 | 0.7139 |

ConvNeXt is now the best ranking model on this transfer test and slightly best by default accuracy. Its behavior differs from the fusion model: it is high precision and low recall. That means it is conservative about calling images generated, but when it does, it is often right.

## Threshold Calibration

Source-domain threshold calibration does not automatically help ConvNeXt. In seeds 7 and 29, the clean Ishu threshold selected for maximum source accuracy is too high for MS COCOAI and reduces default target accuracy.

| method | default accuracy mean | source-threshold accuracy mean | oracle accuracy mean | AUC mean |
| --- | ---: | ---: | ---: | ---: |
| `combined_v3` | 0.5467 | 0.5463 | 0.5743 | 0.5803 |
| ResNet-18 | 0.5800 | 0.5690 | 0.6207 | 0.6488 |
| physics-guided fusion | 0.6060 | 0.6070 | 0.6357 | 0.6637 |
| frozen ConvNeXt-Tiny | 0.6163 | 0.5953 | 0.6757 | 0.7139 |

The paper takeaway is crisp: frozen ConvNeXt gives the best transferable score ranking, while physics-guided fusion currently gives more stable source-threshold accuracy. SCP-Fusion should combine these two strengths rather than replacing one with the other.

## SCP-Fusion v0 Score Fusion Probe

The first lightweight SCP-Fusion probe uses `scripts/fuse_prediction_scores.py` to train a logistic-regression meta-classifier over saved prediction scores:

- `combined_v3`
- ResNet-18
- physics-guided ResNet-18 + `combined_v3`
- frozen ConvNeXt-Tiny

The meta-classifier is trained on each Ishu clean validation split and evaluated on the matching MS COCOAI transfer predictions.

| seed | default accuracy | source-threshold accuracy | oracle accuracy | AUC |
| ---: | ---: | ---: | ---: | ---: |
| 7 | 0.6180 | 0.6360 | 0.6800 | 0.7315 |
| 17 | 0.5680 | 0.5750 | 0.6780 | 0.7239 |
| 29 | 0.5870 | 0.5990 | 0.6800 | 0.7291 |
| mean | 0.5910 | 0.6033 | 0.6793 | 0.7282 |

This is now the best cross-domain ranking result in the repo. It improves mean AUC over frozen ConvNeXt alone, 0.7282 vs 0.7139, and over physics-guided fusion, 0.7282 vs 0.6637. It does not improve default-threshold accuracy because the fused scores remain conservative on generated MS COCOAI images. Source-domain threshold calibration helps modestly, raising mean accuracy from 0.5910 to 0.6033, but the oracle mean accuracy of 0.6793 shows substantial calibration headroom.

For the paper, this is useful evidence that the branches are complementary, but also that cross-source calibration is still the hard part.

## Reproduce

Ishu seed 7 training:

```powershell
python scripts/run_frozen_encoder_baseline.py `
  --data-dir data\raw\ishu_ai_vs_real_2026 `
  --output-dir runs\ishu_ai_vs_real_2026_frozen_encoder\convnext_tiny_seed7 `
  --encoder convnext_tiny `
  --pretrained `
  --classifier logistic_regression `
  --batch-size 32 `
  --num-workers 0 `
  --device cuda `
  --seed 7 `
  --val-fraction 0.2
```

Ishu seed 7 to MS COCOAI:

```powershell
python scripts/evaluate_frozen_encoder_model.py `
  --model-dir runs\ishu_ai_vs_real_2026_frozen_encoder\convnext_tiny_seed7 `
  --target-dir data\raw\ms_cocoai_2026_validation_source_balanced_100 `
  --output-dir runs\ishu_to_ms_cocoai_source_balanced_seed7\convnext_tiny_frozen `
  --target-split all `
  --batch-size 32 `
  --num-workers 0 `
  --device cuda
```

The same commands were run for seeds 17 and 29 by changing the seed and output/model directories.

Score-fusion seed 7:

```powershell
python scripts/fuse_prediction_scores.py `
  --out-dir runs\score_fusion\ishu_seed7_to_ms_cocoai_all4 `
  --seed 7 `
  --train combined_v3=runs\ishu_ai_vs_real_2026_initial\feature_combined_v3_logistic_regression\predictions.csv `
  --train resnet18=runs\ishu_ai_vs_real_2026_initial\resnet18\predictions.csv `
  --train physics_guided=runs\ishu_ai_vs_real_2026_physics_guided_seed7\physics_guided_resnet18_combined_v3\predictions.csv `
  --train convnext_tiny_frozen=runs\ishu_ai_vs_real_2026_frozen_encoder\convnext_tiny_seed7\predictions.csv `
  --variant ms_cocoai:combined_v3=runs\ishu_to_ms_cocoai_source_balanced_seed7\combined_v3\predictions.csv `
  --variant ms_cocoai:resnet18=runs\ishu_to_ms_cocoai_source_balanced_seed7\resnet18\predictions.csv `
  --variant ms_cocoai:physics_guided=runs\ishu_to_ms_cocoai_source_balanced_seed7\physics_guided_resnet18_combined_v3\predictions.csv `
  --variant ms_cocoai:convnext_tiny_frozen=runs\ishu_to_ms_cocoai_source_balanced_seed7\convnext_tiny_frozen\predictions.csv
```
