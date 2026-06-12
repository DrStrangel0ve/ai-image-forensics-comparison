# Cross-Dataset 2026 Generalization Stress Test

Run date: 2026-06-12

This test evaluates saved source models on a different target dataset with no target-dataset tuning. It is deliberately harsher than the same-dataset reports because it checks whether the detector learned transferable forensic signal instead of one dataset's collection or compression artifacts.

## Setup

- Source models: saved `feature_combined_logistic_regression` and pretrained `resnet18` runs from the two full 2026 dataset benchmarks.
- Target split: `all`, because the target dataset was never used to train the source model.
- Target image counts: 995 images for `rhythm_ai_vs_real_2026`, 973 images for `ai_vs_real_2026`.
- Hardware: local CUDA run on the RTX 3060 Ti for ResNet-18 evaluation.

## Results

| transfer | method | accuracy | precision | recall | f1 | roc_auc | target images |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ai_vs_real_2026` -> `rhythm_ai_vs_real_2026` | combined conventional features | 0.6553 | 0.3753 | 0.5600 | 0.4494 | 0.6681 | 995 |
| `ai_vs_real_2026` -> `rhythm_ai_vs_real_2026` | ResNet-18 | 0.6794 | 0.4131 | 0.6560 | 0.5070 | 0.7160 | 995 |
| `rhythm_ai_vs_real_2026` -> `ai_vs_real_2026` | combined conventional features | 0.4923 | 0.5753 | 0.3191 | 0.4105 | 0.5536 | 973 |
| `rhythm_ai_vs_real_2026` -> `ai_vs_real_2026` | ResNet-18 | 0.5468 | 0.6944 | 0.3247 | 0.4425 | 0.6044 | 973 |

## Ishu to MS COCOAI Fusion Transfer

This follow-up evaluates the seed-29 Ishu models against the 1,000-image source-balanced Defactify/MS COCOAI validation slice. The target contains 500 real MS COCO images and 500 generated images balanced across SD2.1, SDXL, SD3, DALL-E 3, and Midjourney v6. No MS COCOAI images are used to tune the source models.

| transfer | method | accuracy | precision | recall | f1 | roc_auc | target images |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ishu_ai_vs_real_2026` seed 29 -> source-balanced MS COCOAI | `combined_v3` | 0.5430 | 0.6352 | 0.2020 | 0.3065 | 0.5840 | 1000 |
| `ishu_ai_vs_real_2026` seed 29 -> source-balanced MS COCOAI | ResNet-18 | 0.5700 | 0.6190 | 0.3640 | 0.4584 | 0.6016 | 1000 |
| `ishu_ai_vs_real_2026` seed 29 -> source-balanced MS COCOAI | physics-guided ResNet-18 + `combined_v3` | 0.6330 | 0.7401 | 0.4100 | 0.5277 | 0.6923 | 1000 |

The fused model keeps its lead in this direction: +6.3 accuracy points and +0.0907 AUC over vanilla ResNet-18, and +9.0 accuracy points and +0.1083 AUC over standalone `combined_v3`. The absolute AUC is still only 0.6923, so the result does not solve cross-domain generalization; it shows that physics/signal features can help the neural representation transfer better than either branch alone.

## MS COCOAI to Ishu Fusion Transfer

The reverse-direction check trains on the 1,000-image Defactify/MS COCOAI subset and evaluates against all 567 Ishu images. This is stricter for threshold calibration because the target class prior is slightly uneven and the real images come from a different collection process than MS COCO.

| transfer | method | accuracy | precision | recall | f1 | roc_auc | target images |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ms_cocoai_2026_subset_500` -> `ishu_ai_vs_real_2026` | `combined_v3` | 0.5608 | 0.5443 | 0.6403 | 0.5884 | 0.5734 | 567 |
| `ms_cocoai_2026_subset_500` -> `ishu_ai_vs_real_2026` | ResNet-18 | 0.6243 | 0.5905 | 0.7626 | 0.6656 | 0.7003 | 567 |
| `ms_cocoai_2026_subset_500` -> `ishu_ai_vs_real_2026` | physics-guided ResNet-18 + `combined_v3` | 0.5873 | 0.5531 | 0.8237 | 0.6618 | 0.7089 | 567 |

In this direction, fusion improves ranking quality but not fixed-threshold accuracy. It gains +0.0086 AUC over vanilla ResNet-18 and +0.1355 AUC over standalone `combined_v3`, but trails ResNet-18 by 3.7 accuracy points because it flags many real Ishu images as generated at the default 0.5 threshold. The practical takeaway is that fusion is promising for transferable signal, but it needs cross-domain calibration before being used as a binary decision rule.

## Cross-Domain Threshold Calibration

Because the reverse transfer had better AUC but worse default-threshold accuracy, this follow-up selects each method's decision threshold on the source-domain clean validation predictions, then applies that threshold to the target-domain transfer predictions without target tuning. The oracle column is diagnostic only because it selects the best threshold on the target labels.

MS COCOAI -> Ishu:

| method | source threshold | default target accuracy | source-calibrated target accuracy | oracle target accuracy | target roc_auc |
| --- | ---: | ---: | ---: | ---: | ---: |
| `combined_v3` | 0.5873 | 0.5608 | 0.5644 | 0.5732 | 0.5734 |
| ResNet-18 | 0.5467 | 0.6243 | 0.6314 | 0.6508 | 0.7003 |
| physics-guided ResNet-18 + `combined_v3` | 0.9015 | 0.5873 | 0.6596 | 0.6737 | 0.7089 |

Ishu seed-29 -> MS COCOAI:

| method | source threshold | default target accuracy | source-calibrated target accuracy | oracle target accuracy | target roc_auc |
| --- | ---: | ---: | ---: | ---: | ---: |
| `combined_v3` | 0.5014 | 0.5430 | 0.5440 | 0.5780 | 0.5840 |
| ResNet-18 | 0.6773 | 0.5700 | 0.5600 | 0.5900 | 0.6016 |
| physics-guided ResNet-18 + `combined_v3` | 0.5000 | 0.6330 | 0.6330 | 0.6520 | 0.6923 |

The calibration result changes the reverse-transfer interpretation. MS COCOAI-trained fusion was not just producing a slightly better AUC; after source-domain calibration it also becomes the best thresholded detector on Ishu, beating source-calibrated ResNet-18 by 2.8 accuracy points. In the Ishu-to-MS COCOAI direction, calibration barely changes fusion because its source-optimal threshold is already 0.5, and it remains ahead of both unfused branches.

## Ishu Three-Seed Transfer Check

The seed-29 Ishu-to-MS COCOAI result was rerun for seed 7 and seed 17 using the already trained Ishu models. This keeps the target fixed as the same 1,000-image source-balanced MS COCOAI validation slice and checks whether the fusion transfer win survives source split variance.

| seed | method | default accuracy | source-calibrated accuracy | oracle accuracy | roc_auc |
| ---: | --- | ---: | ---: | ---: | ---: |
| 7 | `combined_v3` | 0.5430 | 0.5400 | 0.5690 | 0.5750 |
| 7 | ResNet-18 | 0.5820 | 0.5440 | 0.6330 | 0.6593 |
| 7 | physics-guided ResNet-18 + `combined_v3` | 0.6450 | 0.6480 | 0.6480 | 0.6681 |
| 17 | `combined_v3` | 0.5540 | 0.5550 | 0.5760 | 0.5818 |
| 17 | ResNet-18 | 0.5880 | 0.6030 | 0.6390 | 0.6854 |
| 17 | physics-guided ResNet-18 + `combined_v3` | 0.5400 | 0.5400 | 0.6070 | 0.6306 |
| 29 | `combined_v3` | 0.5430 | 0.5440 | 0.5780 | 0.5840 |
| 29 | ResNet-18 | 0.5700 | 0.5600 | 0.5900 | 0.6016 |
| 29 | physics-guided ResNet-18 + `combined_v3` | 0.6330 | 0.6330 | 0.6520 | 0.6923 |

Three-seed summary:

| method | default accuracy mean | source-calibrated accuracy mean | oracle accuracy mean | roc_auc mean | default accuracy wins | calibrated accuracy wins | auc wins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `combined_v3` | 0.5467 | 0.5463 | 0.5743 | 0.5803 | 0 | 0 | 0 |
| ResNet-18 | 0.5800 | 0.5690 | 0.6207 | 0.6488 | 1 | 1 | 1 |
| physics-guided ResNet-18 + `combined_v3` | 0.6060 | 0.6070 | 0.6357 | 0.6637 | 2 | 2 | 2 |

This repeats the main fusion story, but with a caveat. Fusion has the best three-seed mean and wins two of three seeds, yet seed 17 is a real failure case where ResNet-18 is stronger by both accuracy and AUC. Source-threshold calibration does not fix seed 17, so this is ranking instability rather than just threshold drift.

## Interpretation

Same-dataset scores were much higher: ResNet-18 reached 0.8205 accuracy on `ai_vs_real_2026` and 0.9698 on `rhythm_ai_vs_real_2026`; the best combined conventional baselines reached 0.6821 and 0.8693 respectively.

The cross-dataset drop is the key finding. ResNet-18 still beats the combined conventional feature model in both directions, but the margin is small enough that both methods are clearly learning dataset-specific signal. The direction also matters: `ai_vs_real_2026` -> `rhythm_ai_vs_real_2026` generalizes better than the reverse direction.

The conventional baseline remains valuable because it fails differently from the neural net. Its photometric, noise, compression, FFT, and chroma proxies are not enough for strong zero-shot detection, but they provide a cheap sanity check for whether the neural model is mostly exploiting dataset shortcuts.

The two Ishu/MS COCOAI fusion directions now show why both AUC and thresholded accuracy need to be reported. Ishu-trained fusion wins the three-seed mean on MS COCOAI, but not every seed. MS COCOAI-trained fusion initially loses default-threshold accuracy on Ishu, but source-domain threshold calibration raises it to the best transfer accuracy in that direction. The next fair check is to reserve a separate source calibration split and diagnose the seed-17 fusion miss by category/source label rather than only by aggregate metrics.

## New Dataset Leads Checked

- [Defactify / MS COCOAI](https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset): 96,000 images with real MS COCO images and synthetic images from Stable Diffusion 2.1, SDXL, SD3, DALL-E 3, and Midjourney v6.
- [RealHD](https://arxiv.org/abs/2602.10546): 2026 large-scale benchmark with more than 730,000 images, state-of-the-art generation methods, and metadata for generation category.
- [GenImage](https://genimage-dataset.github.io/): million-scale real/fake pairs across ImageNet-style classes and multiple generators.
- [WildFake](https://github.com/hy-zpg/AIGC-Image-Detection-Dataset): large wild-collected benchmark with diverse generators, styles, and real-world use cases.
- [Chameleon / AIDE](https://github.com/shilinyan99/AIDE): ICLR 2025 benchmark focused on AI-generated images that are intentionally hard for existing detectors.

## Reproduce

```powershell
python scripts/evaluate_feature_model.py `
  --model-dir runs/ai_vs_real_2026_full/feature_combined_logistic_regression `
  --target-key rhythm_ai_vs_real_2026 `
  --output-dir runs/cross_dataset/ai_to_rhythm_feature_combined `
  --image-size 128 `
  --target-split all

python scripts/evaluate_neural_net.py `
  --model-dir runs/ai_vs_real_2026_full/resnet18 `
  --target-key rhythm_ai_vs_real_2026 `
  --output-dir runs/cross_dataset/ai_to_rhythm_resnet18 `
  --image-size 128 `
  --batch-size 64 `
  --num-workers 0 `
  --device cuda `
  --target-split all

python scripts/evaluate_feature_model.py `
  --model-dir runs/rhythm_ai_vs_real_2026_full/feature_combined_logistic_regression `
  --target-key ai_vs_real_2026 `
  --output-dir runs/cross_dataset/rhythm_to_ai_feature_combined `
  --image-size 128 `
  --target-split all

python scripts/evaluate_neural_net.py `
  --model-dir runs/rhythm_ai_vs_real_2026_full/resnet18 `
  --target-key ai_vs_real_2026 `
  --output-dir runs/cross_dataset/rhythm_to_ai_resnet18 `
  --image-size 128 `
  --batch-size 64 `
  --num-workers 0 `
  --device cuda `
  --target-split all
```

Ishu seed-29 to source-balanced MS COCOAI:

```powershell
python scripts/evaluate_feature_model.py `
  --model-dir runs/ishu_ai_vs_real_2026_repeated_splits_auto/seed29/feature_combined_v3_logistic_regression `
  --target-dir data/raw/ms_cocoai_2026_validation_source_balanced_100 `
  --output-dir runs/ishu_to_ms_cocoai_source_balanced_seed29/combined_v3 `
  --image-size 128 `
  --target-split all `
  --skip-errors

python scripts/evaluate_neural_net.py `
  --model-dir runs/ishu_ai_vs_real_2026_repeated_splits_auto/seed29/resnet18 `
  --target-dir data/raw/ms_cocoai_2026_validation_source_balanced_100 `
  --output-dir runs/ishu_to_ms_cocoai_source_balanced_seed29/resnet18 `
  --image-size 128 `
  --batch-size 64 `
  --num-workers 0 `
  --device cuda `
  --target-split all

python scripts/evaluate_physics_guided_net.py `
  --model-dir runs/ishu_ai_vs_real_2026_physics_guided_seed29/physics_guided_resnet18_combined_v3 `
  --target-dir data/raw/ms_cocoai_2026_validation_source_balanced_100 `
  --output-dir runs/ishu_to_ms_cocoai_source_balanced_seed29/physics_guided_resnet18_combined_v3 `
  --image-size 128 `
  --feature-image-size 128 `
  --batch-size 64 `
  --num-workers 0 `
  --device cuda `
  --target-split all `
  --skip-errors
```

MS COCOAI to Ishu:

```powershell
python scripts/train_physics_guided_net.py `
  --data-dir data/raw/ms_cocoai_2026_subset_500 `
  --output-dir runs/ms_cocoai_2026_subset_500/physics_guided_resnet18_combined_v3 `
  --model resnet18 `
  --pretrained `
  --epochs 4 `
  --batch-size 64 `
  --image-size 128 `
  --feature-image-size 128 `
  --num-workers 0 `
  --device cuda `
  --seed 7 `
  --val-fraction 0.2 `
  --physics-feature-set combined_v3 `
  --skip-errors

python scripts/evaluate_physics_guided_net.py `
  --model-dir runs/ms_cocoai_2026_subset_500/physics_guided_resnet18_combined_v3 `
  --target-dir data/raw/ishu_ai_vs_real_2026 `
  --output-dir runs/ishu_ai_vs_real_2026_cross_ms_cocoai/physics_guided_resnet18_combined_v3 `
  --image-size 128 `
  --feature-image-size 128 `
  --batch-size 64 `
  --num-workers 0 `
  --device cuda `
  --target-split all `
  --skip-errors
```

Cross-domain threshold calibration:

```powershell
python scripts/summarize_threshold_calibration.py `
  --out-dir runs/cross_domain_threshold_calibration/ms_cocoai_to_ishu `
  --objective accuracy `
  --clean combined_v3=runs/ms_cocoai_2026_source_balanced_eval/combined_v3/predictions.csv `
  --clean resnet18=runs/ms_cocoai_2026_source_balanced_eval/resnet18/predictions.csv `
  --clean physics_guided_resnet18_combined_v3=runs/ms_cocoai_2026_source_balanced_eval/physics_guided_resnet18_combined_v3/predictions.csv `
  --variant ishu:combined_v3=runs/ishu_ai_vs_real_2026_cross_ms_cocoai/combined_v3/predictions.csv `
  --variant ishu:resnet18=runs/ishu_ai_vs_real_2026_cross_ms_cocoai/resnet18/predictions.csv `
  --variant ishu:physics_guided_resnet18_combined_v3=runs/ishu_ai_vs_real_2026_cross_ms_cocoai/physics_guided_resnet18_combined_v3/predictions.csv

python scripts/summarize_threshold_calibration.py `
  --out-dir runs/cross_domain_threshold_calibration/ishu_seed29_to_ms_cocoai `
  --objective accuracy `
  --clean combined_v3=runs/ishu_ai_vs_real_2026_repeated_splits_auto/seed29/feature_combined_v3_logistic_regression/predictions.csv `
  --clean resnet18=runs/ishu_ai_vs_real_2026_repeated_splits_auto/seed29/resnet18/predictions.csv `
  --clean physics_guided_resnet18_combined_v3=runs/ishu_ai_vs_real_2026_physics_guided_seed29/physics_guided_resnet18_combined_v3/predictions.csv `
  --variant ms_cocoai:combined_v3=runs/ishu_to_ms_cocoai_source_balanced_seed29/combined_v3/predictions.csv `
  --variant ms_cocoai:resnet18=runs/ishu_to_ms_cocoai_source_balanced_seed29/resnet18/predictions.csv `
  --variant ms_cocoai:physics_guided_resnet18_combined_v3=runs/ishu_to_ms_cocoai_source_balanced_seed29/physics_guided_resnet18_combined_v3/predictions.csv
```

Ishu three-seed transfer extension:

The first three commands show the seed-7 evaluation paths; repeat them with `runs/ishu_ai_vs_real_2026_seed17/...` and `runs/ishu_ai_vs_real_2026_physics_guided_seed17/...` for seed 17 before running the summary command.

```powershell
python scripts/evaluate_feature_model.py `
  --model-dir runs/ishu_ai_vs_real_2026_initial/feature_combined_v3_logistic_regression `
  --target-dir data/raw/ms_cocoai_2026_validation_source_balanced_100 `
  --output-dir runs/ishu_to_ms_cocoai_source_balanced_seed7/combined_v3 `
  --image-size 128 `
  --target-split all `
  --skip-errors

python scripts/evaluate_neural_net.py `
  --model-dir runs/ishu_ai_vs_real_2026_initial/resnet18 `
  --target-dir data/raw/ms_cocoai_2026_validation_source_balanced_100 `
  --output-dir runs/ishu_to_ms_cocoai_source_balanced_seed7/resnet18 `
  --image-size 128 `
  --batch-size 64 `
  --num-workers 0 `
  --device cuda `
  --target-split all

python scripts/evaluate_physics_guided_net.py `
  --model-dir runs/ishu_ai_vs_real_2026_physics_guided_seed7/physics_guided_resnet18_combined_v3 `
  --target-dir data/raw/ms_cocoai_2026_validation_source_balanced_100 `
  --output-dir runs/ishu_to_ms_cocoai_source_balanced_seed7/physics_guided_resnet18_combined_v3 `
  --image-size 128 `
  --feature-image-size 128 `
  --batch-size 64 `
  --num-workers 0 `
  --device cuda `
  --target-split all `
  --skip-errors

python scripts/summarize_threshold_calibration.py `
  --out-dir runs/cross_domain_threshold_calibration/ishu_3seed_to_ms_cocoai `
  --objective accuracy `
  --clean seed7_combined_v3=runs/ishu_ai_vs_real_2026_initial/feature_combined_v3_logistic_regression/predictions.csv `
  --clean seed7_resnet18=runs/ishu_ai_vs_real_2026_initial/resnet18/predictions.csv `
  --clean seed7_physics_guided_resnet18_combined_v3=runs/ishu_ai_vs_real_2026_physics_guided_seed7/physics_guided_resnet18_combined_v3/predictions.csv `
  --clean seed17_combined_v3=runs/ishu_ai_vs_real_2026_seed17/feature_combined_v3_logistic_regression/predictions.csv `
  --clean seed17_resnet18=runs/ishu_ai_vs_real_2026_seed17/resnet18/predictions.csv `
  --clean seed17_physics_guided_resnet18_combined_v3=runs/ishu_ai_vs_real_2026_physics_guided_seed17/physics_guided_resnet18_combined_v3/predictions.csv `
  --clean seed29_combined_v3=runs/ishu_ai_vs_real_2026_repeated_splits_auto/seed29/feature_combined_v3_logistic_regression/predictions.csv `
  --clean seed29_resnet18=runs/ishu_ai_vs_real_2026_repeated_splits_auto/seed29/resnet18/predictions.csv `
  --clean seed29_physics_guided_resnet18_combined_v3=runs/ishu_ai_vs_real_2026_physics_guided_seed29/physics_guided_resnet18_combined_v3/predictions.csv `
  --variant ms_cocoai:seed7_combined_v3=runs/ishu_to_ms_cocoai_source_balanced_seed7/combined_v3/predictions.csv `
  --variant ms_cocoai:seed7_resnet18=runs/ishu_to_ms_cocoai_source_balanced_seed7/resnet18/predictions.csv `
  --variant ms_cocoai:seed7_physics_guided_resnet18_combined_v3=runs/ishu_to_ms_cocoai_source_balanced_seed7/physics_guided_resnet18_combined_v3/predictions.csv `
  --variant ms_cocoai:seed17_combined_v3=runs/ishu_to_ms_cocoai_source_balanced_seed17/combined_v3/predictions.csv `
  --variant ms_cocoai:seed17_resnet18=runs/ishu_to_ms_cocoai_source_balanced_seed17/resnet18/predictions.csv `
  --variant ms_cocoai:seed17_physics_guided_resnet18_combined_v3=runs/ishu_to_ms_cocoai_source_balanced_seed17/physics_guided_resnet18_combined_v3/predictions.csv `
  --variant ms_cocoai:seed29_combined_v3=runs/ishu_to_ms_cocoai_source_balanced_seed29/combined_v3/predictions.csv `
  --variant ms_cocoai:seed29_resnet18=runs/ishu_to_ms_cocoai_source_balanced_seed29/resnet18/predictions.csv `
  --variant ms_cocoai:seed29_physics_guided_resnet18_combined_v3=runs/ishu_to_ms_cocoai_source_balanced_seed29/physics_guided_resnet18_combined_v3/predictions.csv
```
