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

## Interpretation

Same-dataset scores were much higher: ResNet-18 reached 0.8205 accuracy on `ai_vs_real_2026` and 0.9698 on `rhythm_ai_vs_real_2026`; the best combined conventional baselines reached 0.6821 and 0.8693 respectively.

The cross-dataset drop is the key finding. ResNet-18 still beats the combined conventional feature model in both directions, but the margin is small enough that both methods are clearly learning dataset-specific signal. The direction also matters: `ai_vs_real_2026` -> `rhythm_ai_vs_real_2026` generalizes better than the reverse direction.

The conventional baseline remains valuable because it fails differently from the neural net. Its photometric, noise, compression, FFT, and chroma proxies are not enough for strong zero-shot detection, but they provide a cheap sanity check for whether the neural model is mostly exploiting dataset shortcuts.

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
