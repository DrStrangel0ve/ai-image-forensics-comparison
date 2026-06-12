# Research Deep Dive: AI Image Forensics Direction

Run date: 2026-06-12

This note reviews recent AI-generated image detection research and turns it into a concrete next direction for this repo. The short version: our current physics-guided ResNet plus `combined_v3` feature fusion is directionally right, but the literature suggests it needs three additions to become competitive in the wild:

1. frozen foundation features, especially CLIP/DINO-style embeddings;
2. spectral/reconstruction features that stress diffusion and latent-diffusion artifacts;
3. source-heldout validation and calibration, because same-dataset accuracy is too forgiving.

## What Recent Papers Say

### 1. Generalization is the main problem

[Towards Universal Fake Image Detectors that Generalize Across Generative Models](https://arxiv.org/abs/2302.10174) showed that a detector trained directly as real-vs-fake can collapse on unseen generators, while simple nearest-neighbor or linear probes over large pretrained vision-language features can generalize surprisingly well. This is the clearest argument that our vanilla ResNet-18 should not be the final neural baseline. A frozen or lightly tuned CLIP/DINO branch should be added.

[GenImage](https://arxiv.org/abs/2306.08571), with its [project page](https://genimage-dataset.github.io/), is still an important large-scale benchmark because it explicitly evaluates cross-generator and degraded-image settings. It is less recent than MS COCOAI/RealHD, but its evaluation philosophy maps directly to our current cross-dataset tests.

[A Sanity Check for AI-generated Image Detection](https://openreview.net/forum?id=ODRHZrkOQM), ICLR 2025, introduced the Chameleon stress test and AIDE. The useful design point is that AIDE is multi-expert: CLIP semantics plus patchwise high/low frequency artifact features. That closely matches the failure pattern we saw: physics/signal fusion helps on average, but a single fused branch can become too conservative for unseen generators.

[RealHD](https://arxiv.org/abs/2602.10546) is one of the most relevant 2026 dataset leads. It has more than 730,000 images across real and AI-generated categories, with state-of-the-art generation methods and metadata for generation method/category. Its own baseline uses noise entropy over Non-Local Means residuals, which reinforces the value of expanding our noise branch beyond basic residual summary statistics.

### 2. Reconstruction error is a strong physics-adjacent signal

[DIRE](https://arxiv.org/abs/2303.09295) detects diffusion-generated images by comparing an input with a reconstruction from a pretrained diffusion model. The key assumption is that diffusion images reconstruct differently from real images. It is powerful but expensive.

[AEROBLADE](https://arxiv.org/abs/2401.17879), CVPR 2024, is more practical for us: it uses autoencoder reconstruction error from latent diffusion autoencoders and does not require training a detector. For a 3060 Ti, an AEROBLADE-style feature extractor is a better first implementation target than full DIRE.

[FIRE](https://arxiv.org/abs/2412.07140), CVPR 2025, combines reconstruction error with frequency decomposition and reports robustness to perturbations and unseen diffusion models. This is especially relevant because our conventional `combined_v3` branch already has FFT/JPEG/noise features; the next step is not just more scalar FFT stats, but reconstruction-error features measured per frequency band.

### 3. Frequency and high-resolution details matter

[Any-Resolution AI-Generated Image Detection by Spectral Learning](https://arxiv.org/abs/2411.19417), CVPR 2025, argues that real-image spectral distributions are a more invariant target than generator-specific fake artifacts. Its SPAI detector uses masked spectral reconstruction and spectral context attention, and reports robustness across recent generators and online perturbations.

[No Pixel Left Behind](https://arxiv.org/abs/2508.17346) argues that resizing/cropping high-resolution images discards forensic evidence. Its high-resolution detector fuses local tiles with a global view. Our 128x128 experiments are fine for rapid iteration, but any contest-quality system should add tiled inference at native or higher resolution.

[Fake or JPEG?](https://arxiv.org/abs/2403.17608) is an important warning: generated-image datasets often leak JPEG/size biases, and detectors learn those shortcuts. This validates the audit and robustness code already in this repo, but also says our metrics should always include JPEG/resize robustness and source-heldout tests before we trust a gain.

[A Bias-Free Training Paradigm for More General AI-generated Image Detection](https://arxiv.org/html/2412.17671v2), CVPR 2025, creates semantically aligned real/fake pairs by generating from real images via conditioning. That is a good training recipe for a future model: reduce semantic/content bias so the detector has to learn generation artifacts.

### 4. Physics-informed cues are still underused

[A Geometric and Photometric Exploration of GAN and Diffusion Synthesized Faces](https://openaccess.thecvf.com/content/CVPR2023W/WMF/html/Bohacek_A_Geometric_and_Photometric_Exploration_of_GAN_and_Diffusion_Synthesized_CVPRW_2023_paper.html) estimated 3D face geometry and photometric environments to analyze synthetic faces. It is not a generic detector for all images, but it supports the core intuition behind our photometric branch: generated images can be locally plausible while globally physically inconsistent.

[Light2Lie](https://www.ndss-symposium.org/ndss-paper/light2lie-detecting-deepfake-images-using-physical-reflectance-laws/), NDSS 2026, is the most direct physics-informed reference I found. It uses Fresnel/specular reflectance cues and a learned reflectance-estimation stage to expose light-surface inconsistencies. This suggests our current pseudo-normal integrability proxy is a useful start, but the stronger version should model reflectance/specular consistency, not only luminance-gradient normals.

## What This Means For Our Current Results

Current best direction:

- Same-domain Ishu repeated results make fusion look promising: `combined_v3` plus ResNet-18 had strong mean accuracy/AUC and beat vanilla ResNet in two of three seeds for Ishu-to-MS-COCOAI transfer.
- The seed-17 failure is the warning sign. Regularization improved default transfer accuracy slightly but reduced AUC, so the issue is not just dropout or weight decay.
- The model needs source-aware training/validation and more diverse branches. The literature agrees: top systems are heterogeneous ensembles or hybrid-feature detectors, not a single monolithic CNN.

The photometric branch should not be discarded. It should become one branch in a larger forensic system, alongside spectral, reconstruction, noise, compression, chroma, and foundation-embedding branches.

## Proposed Method: SCP-Fusion

Name: Source-Calibrated Physical-Spectral Foundation Fusion (SCP-Fusion)

Goal: maximize cross-generator and transform robustness, not same-dataset accuracy.

Architecture:

1. Pixel branch: pretrained ConvNeXt/ResNet or ViT trained on images.
2. Foundation branch: frozen CLIP or DINO-style embeddings with a small linear/MLP head.
3. Physics branch: current photometric normal-consistency features plus new reflectance/specular summaries inspired by Light2Lie.
4. Spectral/noise branch: `combined_v4` handcrafted features with multiscale FFT, DCT/JPEG periodicity, residual entropy, chroma residuals, and patchwise high/low frequency stats.
5. Reconstruction branch: AEROBLADE-style latent autoencoder reconstruction error, plus optional FIRE-style band-limited reconstruction deltas.
6. Fusion head: calibrated logit-level fusion, with branch dropout during training so the model cannot over-rely on one shortcut.

Training:

- Use source-balanced sampling when generator/source metadata exists.
- Use leave-one-source-out validation on MS COCOAI/RealHD/GenImage-style data.
- Add JPEG, resize, crop, blur, noise, and screenshot-style transforms in training.
- Add a calibration split that is separate from validation, then report source-calibrated accuracy, AUC, Brier score, and ECE.
- Keep three-seed reporting. A single seed is too noisy for this problem.

Evaluation:

1. In-domain: Ishu, Rhythm, AI-vs-real Kaggle datasets.
2. Cross-domain: Ishu -> MS COCOAI and MS COCOAI -> Ishu.
3. Leave-one-generator-out: train on four MS COCOAI generator labels, test on the held-out generator.
4. Robust transforms: JPEG70/50, blur, resize, crop, screenshot/noise.
5. High-resolution tiling: compare 128x128 resize against local-tile plus global-view inference.

## Immediate Implementation Plan

1. Add `scripts/run_foundation_baseline.py`:
   - extract frozen CLIP/DINO embeddings;
   - train logistic regression or a small MLP;
   - evaluate using the existing prediction CSV format.

2. Add `combined_v4` features:
   - patchwise FFT/DCT bands;
   - JPEG quality/block periodicity;
   - NLM residual entropy, inspired by RealHD;
   - multiscale photometric normal statistics;
   - optional face/object-region features only when a detector is present.

3. Add an AEROBLADE-lite extractor:
   - run a pretrained VAE/autoencoder reconstruction;
   - write reconstruction error, residual spectrum, and patch residual summaries as feature columns;
   - start as an offline feature cache so it is not recomputed every run.

4. Add source-aware split tooling:
   - group by metadata source/generator;
   - run leave-one-source-out evaluation;
   - output source-level AUC/accuracy and threshold calibration tables.

5. Add fusion v2:
   - combine foundation logits, ResNet logits, `combined_v4`, and reconstruction features;
   - train with branch dropout;
   - calibrate thresholds on a source-heldout split.

## Contest And Benchmark Opportunities

Status is as of 2026-06-12.

### Best fit: NTIRE/CVPR robust detection tracks

[NTIRE 2026 Robust AI-Generated Image Detection in the Wild](https://www.codabench.org/competitions/12761/) is the best match for this repo. The [CVPR workshop report](https://arxiv.org/abs/2604.11487) says the challenge used 108,750 real and 185,750 generated images from 42 generators, with 36 transforms, evaluated by ROC AUC. The official NTIRE page lists "Robust AI-Generated Image Detection in the Wild" and "Robust Deepfake Detection" as 2026 challenges. The 2026 leaderboard phase appears effectively over because the workshop already happened on June 4, 2026, but the dataset/report are perfect as a benchmark target and NTIRE 2027 is worth watching.

[NTIRE 2026 Robust Deepfake Detection Challenge](https://www.codabench.org/competitions/12795/) is adjacent. It focuses more on robust face/deepfake detection than generic generated-image detection. Our physics branch could be useful here if we add face-specific lighting/reflectance features.

### Already closed but still useful: ImageCLEF 2026

[ImageCLEF 2026 Deepfake Detection and Generation](https://www.imageclef.org/2026/deepfake-detection-and-generation) included an image detection subtask designed to study why some deepfakes are especially difficult to detect. The detection submission deadline was 2026-05-07, so it is closed as of this report. It remains useful as a model for our evaluation style because it rewards detectors that do not depend on a single generator fingerprint.

### Kaggle status

Using the Kaggle API in the project virtualenv, I searched competitions for "AI generated image", "generated image", "AI image", "deepfake detection", "fake", and "synthetic media". I did not find an active, high-value Kaggle competition directly matching generic real-vs-AI image detection as of 2026-06-12.

Useful closed Kaggle competitions:

- [Detect AI vs Human-Generated Images](https://www.kaggle.com/competitions/detect-ai-vs-human-generated-images), closed 2025-03-07, 554 teams, USD 10,000 community competition.
- [CIDAut AI Fake Scene Classification 2024](https://www.kaggle.com/competitions/cidaut-ai-fake-scene-classification-2024), closed 2025-01-18.
- [Robust/deepfake community tasks](https://www.kaggle.com/competitions/deepfake-detection-challenge), mostly older or closed.

Useful Kaggle datasets to import or audit:

- `muqaddasejaz/ai-generated-vs-real-images-dataset`, updated 2026-05-23.
- `rhythmghai/ai-vs-real-images-dataset`, updated 2026-03-21.
- `chuneeb/deepfake-detection-dataset-2026`, updated 2026-05-06.
- `muhammadbilal6305/200k-real-vs-ai-visuals-by-mbilal`, updated 2025-05-27.
- `ayushmandatta1/deepdetect-2025`, updated 2025-05-03.
- `shreyasraghav/shutterstock-dataset-for-ai-vs-human-gen-image`, updated 2025-06-19.

## Recommendation

The next serious milestone should not be "train a deeper ResNet." It should be SCP-Fusion v1:

- frozen foundation baseline;
- `combined_v4` spectral/noise/photometric features;
- AEROBLADE-lite reconstruction features;
- source-heldout training and calibration;
- NTIRE-style robustness reporting.

If this works, the project becomes contest-ready for future NTIRE/ImageCLEF-style tracks and more scientifically defensible than a same-dataset neural-vs-photometric comparison.
