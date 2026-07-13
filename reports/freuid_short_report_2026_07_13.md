# Type-Adversarial High-Resolution Forensics for FREUID 2026

## Summary

We detect fraudulent identity documents with a complementary two-network ensemble. A ConvNeXt-Tiny branch captures strong visual evidence on known document layouts. A 384-pixel EfficientNet-B0 branch is explicitly regularized against document-template shortcuts using a trainable high-pass residual adapter, capture augmentation, balanced type/label sampling, and a gradient-reversal document-type objective. The frozen detector fuses percentage ranks rather than raw probabilities:

```text
score = 0.85 * rank(ConvNeXt-Tiny-224) + 0.15 * rank(residual EfficientNet-B0-384)
```

The submitted OOD candidate is Kaggle ref `54627101` with public score `0.25799`. The pure public specialist is ref `54624136` with score `0.25470`. Lower is better.

## Data and Protocol

The official archive contains `69,352` locally matched training images. The released public image set contains `7,821` files, while `sample_submission.csv` contains `142,818` IDs. Full submissions therefore preserve an identical organizer/sample-derived fallback for the `134,997` rows whose images are not locally released.

Random type/label-stratified validation is reported only as a training sanity check because image appearance and document layout overlap strongly. Model selection instead uses leave-one-document-type-out (LOTO) validation plus paired JPEG, blur, resize, noise, screenshot, and social-media transforms. EGYPT/DL is the hardest observed LOTO domain and drives the conservative fusion choice.

## Models

### Public specialist

The public specialist fine-tunes ImageNet ConvNeXt-Tiny at `224 x 224` on the full available training data with fraud and document-type heads. It is highly accurate on known layouts, but the auxiliary type head reaches `100%` type accuracy. This reveals a template shortcut that is risky when private evaluation includes unseen document types.

### OOD forensic specialist

The OOD branch uses EfficientNet-B0 at `384 x 384`. A learnable `1 x 1` adapter mixes normalized RGB with a local high-pass residual (`image - 5x5 local mean`) and is initialized as an exact RGB identity. Capture augmentation includes JPEG recompression, perspective/affine changes, blur, resampling, color variation, and sensor-like noise.

During training, a gradient-reversal layer makes the shared representation uninformative about the five known document types. Training uses balanced type/label batches and stops after one epoch because longer LOTO runs reduced strict low-BPCER generalization. On its all-type random validation set, the frozen branch obtains `0.998826` AUC and `24.0%` document-type accuracy, close to the `20%` chance level.

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
| `54624136` | full-data ConvNeXt public specialist | **0.25470** |
| `54626233` | raw 65/35 neural fusion | 0.27166 |
| `54627101` | frozen 85/15 rank OOD ensemble | 0.25799 |

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

The runtime loads one checkpoint at a time to remain practical on an RTX 3060 Ti. `freeze_manifest.json` records the exact model hashes, sizes, fusion rule, validation basis, and Kaggle references. Unit tests cover legacy and new checkpoints, residual/multi-view construction, rank ties, output format, and hash manifests.

## Limitations

Only five training document types and the small released public image subset are available. LOTO is a proxy for the two organizer-announced unseen private types, not a direct estimate of private score. The high-pass adapter is a single-image forensic cue, not classical multi-illumination photometric stereo. Docker image execution has not been completed locally because Windows virtualization is unavailable, although direct no-network-equivalent inference from the frozen artifacts passes.
