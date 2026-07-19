# Type-Adversarial High-Resolution Forensics for FREUID 2026

## Introduction

We detect fraudulent identity documents with a complementary two-network ensemble. A ConvNeXt-Tiny branch captures strong visual evidence on known document layouts. A 384-pixel EfficientNet-B0 branch is explicitly regularized against document-template shortcuts using a trainable high-pass residual adapter, capture augmentation, balanced type/label sampling, and a gradient-reversal document-type objective. The frozen detector fuses percentage ranks rather than raw probabilities:

```text
score = 0.85 * rank(ConvNeXt-Tiny-224) + 0.15 * rank(residual EfficientNet-B0-384)
```

The private-complete OOD candidate is Kaggle ref `54673732` with public score `0.25799`. The private-complete public specialist is ref `54673713` with score `0.25470`. Lower is better.

## Data and Protocol

The official archive contains `69,352` locally matched training images. The public image set contains `7,821` files, while the final private release contains the remaining `134,997` images from the `142,818` submission IDs. Frozen private inference replaced exactly those `134,997` rows and preserved the `7,821` public rows. No external document images are used. Both neural encoders start from public ImageNet-1K torchvision weights; all subsequent optimization uses only official FREUID training images.

Random type/label-stratified validation is reported only as a training sanity check because image appearance and document layout overlap strongly. Model selection instead uses leave-one-document-type-out (LOTO) validation plus paired JPEG, blur, resize, noise, screenshot, and social-media transforms. EGYPT/DL is the hardest observed LOTO domain and drives the conservative fusion choice.

## Models

### Public specialist

The public specialist fine-tunes ImageNet ConvNeXt-Tiny at `224 x 224` on the full available training data with fraud and document-type heads. It is highly accurate on known layouts, but the auxiliary type head reaches `100%` type accuracy. This reveals a template shortcut that is risky when private evaluation includes unseen document types.

### OOD forensic specialist

The OOD branch uses EfficientNet-B0 at `384 x 384`. A learnable `1 x 1` adapter mixes normalized RGB with a local high-pass residual (`image - 5x5 local mean`) and is initialized as an exact RGB identity. Capture augmentation includes JPEG recompression, perspective/affine changes, blur, resampling, color variation, and sensor-like noise.

During training, a gradient-reversal layer makes the shared representation uninformative about the five known document types. Training uses balanced type/label batches and stops after one epoch because longer LOTO runs reduced strict low-BPCER generalization. On its all-type random validation set, the frozen branch obtains `0.998826` AUC and `24.0%` document-type accuracy, close to the `20%` chance level.

## Inference

The public specialist loads only ConvNeXt-Tiny. The OOD variant sequentially loads ConvNeXt-Tiny and EfficientNet-B0, releases each model after scoring, converts each complete score vector to percentage ranks, and applies the frozen `0.85/0.15` blend. Sequential loading keeps peak memory practical on the development RTX 3080 Ti 12GB and comfortably below the organizer's A100 40GB limit.

One Docker image reproduces both selected final picks through an inference-only environment variable:

```text
FREUID_VARIANT=public_specialist -> ref 54673713
FREUID_VARIANT=ood_rank          -> ref 54673732 (default)
```

The container recursively discovers supported images under `/data`, derives each ID from the filename stem, and writes one finite fraud score per image to `/submissions/submission.csv`. It performs no network access and writes nowhere else. The selected final CSV SHA-256 values are `f2a873...68b2` for the public specialist and `5ce966...df07` for the OOD rank candidate.

The unchanged frozen models scored all `134,997` private images in Kaggle kernel version 4. The two-T4 inference run completed in `3,415.81 s`; parallel worker scheduling changed only orchestration, not model behavior or fusion.

The exact OOD path scores `1,000` images in `211.4 s` on the development RTX 3080 Ti at batch size `32` with four workers. This includes two checkpoint loads, two CUDA initializations, and Windows worker startup. The corresponding naive local projection is `8.39 h`; organizer verification instead uses an A100 40GB with 24 CPUs, so the Docker defaults to batch size `96` and `12` workers. Six-hour compliance remains an organizer-hardware verification item rather than a claimed local measurement.

## Results

### Hard unseen-type validation

| Evaluation | Method | APCER @ 1% BPCER | AuDET proxy | AUC |
| --- | --- | ---: | ---: | ---: |
| clean EGYPT LOTO | global EfficientNet 384 | 0.404 | 0.171010 | 0.829248 |
| clean EGYPT LOTO | residual/type-adversarial EfficientNet | 0.442 | **0.161552** | **0.838920** |
| clean EGYPT LOTO | frozen rank ensemble | **0.402** | 0.167956 | 0.832336 |
| screenshot EGYPT LOTO | global EfficientNet 384 | 0.422 | 0.171834 | 0.828460 |
| screenshot EGYPT LOTO | frozen rank ensemble | **0.416** | **0.168740** | **0.831586** |

The standalone forensic branch gives the best average ranking but sacrifices APCER at the required operating point. The `0.85/0.15` ensemble preserves the global model's low-BPCER behavior while improving AuDET and AUC. This tradeoff is why the final weight is deliberately conservative.

### Kaggle public leaderboard

| Ref | Method | Public score |
| --- | --- | ---: |
| `54511333` | conventional + frozen-feature baseline | 0.37009 |
| `54673713` | private-complete ConvNeXt public specialist | **0.25470** |
| `54626233` | raw 65/35 neural fusion | 0.27166 |
| `54673732` | private-complete frozen 85/15 rank OOD ensemble | 0.25799 |

Rank fusion recovers most of the public specialist's performance while retaining the unseen-type branch. The OOD candidate is only `0.00329` behind the specialist on known public layouts and is selected as the reproducible private-test runtime.

## Reproducibility

Repository: <https://github.com/DrStrangel0ve/ai-image-forensics-comparison>

Runtime contract:

```text
Input:  images recursively under /data
Output: /submissions/submission.csv
Format: id,label where id is the filename stem and label is a score in [0,1]
Network: disabled during inference
```

Build and run commands are documented in `artifacts/freuid_2026/README.md`. The runtime loads one checkpoint at a time to remain practical on an RTX 3080 Ti. `freeze_manifest.json` records the exact model hashes, sizes, fusion rule, validation basis, and Kaggle references. Unit tests cover legacy and new checkpoints, residual/multi-view construction, rank ties, output format, and hash manifests. The development environment is Windows 11, Python 3.11, PyTorch/torchvision, and an NVIDIA RTX 3080 Ti 12GB; organizer verification targets one A100 40GB and 24 CPUs with a six-hour limit. **Limitations:** only five training document types and the small released public image subset are available. LOTO is a proxy for two organizer-announced unseen private types, not a direct estimate of private score. The high-pass adapter is a single-image cue, not classical multi-illumination photometric stereo. Docker execution remains locally blocked by unavailable Windows virtualization, although direct no-network-equivalent inference from the frozen artifacts passes.
