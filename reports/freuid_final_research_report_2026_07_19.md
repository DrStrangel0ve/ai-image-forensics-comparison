# Type-Adversarial Rank Fusion for Identity-Document Fraud Detection

## Final FREUID 2026 Research Report and Project Closeout

**Author:** Arnav Malani<br>
**Date:** 2026-07-19<br>
**Project status:** Concluded<br>
**Repository:** <https://github.com/DrStrangel0ve/ai-image-forensics-comparison><br>
**Frozen release:** <https://github.com/DrStrangel0ve/ai-image-forensics-comparison/releases/tag/freuid-final-2026-07-13>

## Abstract

This report closes the FREUID Challenge 2026 track of the AI Image Forensics Comparison project. The task is to rank fraudulent identity documents while operating at very low bona-fide error rates and generalizing from five known document types to a private distribution containing unseen types and capture conditions. We evaluated conventional photometric and residual features, frozen visual encoders, a full-data ConvNeXt-Tiny specialist, a high-resolution residual EfficientNet-B0 with document-type adversarial training, probability and percentile-rank fusion, and a post-freeze DINOv2 five-crop probe. The final competition system combined the ConvNeXt specialist and type-adversarial forensic branch using a conservative `0.85/0.15` percentile-rank blend. It scored `0.25799` on the public split and `0.73118` on the private split, finishing 46th. The public specialist scored slightly better publicly (`0.25470`) but worse privately (`0.74443`), supporting the central hypothesis that suppressing template identity and retaining a small forensic branch improves distribution-shift robustness. High-resolution DINOv2 features produced useful held-out-type ranking (`0.8585` ROC AUC) but failed at fixed-threshold calibration, while single-image photometric proxies were useful during small-data exploration but did not become the final detector. The principal lesson is that document-template shortcuts dominate random validation; leave-one-type-out evaluation, capture stress tests, rank normalization, and reproducible inference packaging were more valuable than adding unconstrained model complexity.

## 1. Introduction

Identity-document fraud detection differs from ordinary natural-image classification. Evidence may occupy a small portrait, text, or security-feature region, while most pixels encode a document template. A classifier can therefore achieve excellent random-split validation by memorizing layouts without learning fraud evidence. This shortcut is especially dangerous when evaluation contains unseen document types, recompressed screenshots, rescaled captures, or recaptured printed documents.

The FREUID work pursued two complementary objectives:

1. retain a strong known-layout specialist for the released public distribution; and
2. learn a forensic branch whose representation is less predictive of document type and more stable under capture shift.

The final system does not claim to solve unseen-document generalization. Instead, it tests whether a deliberately small contribution from a type-adversarial, high-resolution branch can improve private performance without discarding the strong public specialist.

## 2. Data and Evaluation Protocol

The official data available locally contained `69,352` matched training images across five document/source types: `BENIN/DL`, `EGYPT/DL`, `GUINEA/DL`, `MAURITIUS/ID`, and `MOZAMBIQUE/DL`. The complete submission contained `142,818` IDs: `7,821` released public images and `134,997` private images. No external document-image dataset was used. Neural encoders began from public ImageNet-1K or DINOv2 weights, after which training used official FREUID images only.

Random type/label-stratified validation was treated as a pipeline and optimization sanity check, not primary generalization evidence. Model selection emphasized:

- leave-one-document-type-out (LOTO) validation;
- the hardest observed `EGYPT/DL` holdout;
- paired JPEG, blur, resize, noise, screenshot, and social-media transforms;
- ROC AUC, the local AuDET proxy, and APCER at `1%` BPCER; and
- official public and private challenge scores, where lower is better.

This distinction matters because the all-type random split strongly rewards template recognition. The public specialist's auxiliary document-type head reached `100%` accuracy, making the shortcut measurable rather than hypothetical.

## 3. Methods

### 3.1 Conventional and photometric baselines

Early experiments combined single-image photometric-normal proxies with noise residuals, JPEG/block artifacts, error-level differences, FFT statistics, chroma consistency, and edge features. Logistic regression and histogram gradient boosting provided low-cost baselines. These methods were valuable for validating the data path and low-BPCER metric code. On the balanced `320/160` local slice, the best four-branch rank fusion reached `0.8688` accuracy, `0.9452` AUC, `0.2625` APCER at `1%` BPCER, and `0.0595` AuDET proxy.

The photometric branch should not be interpreted as true photometric stereo. FREUID provides single images without known lighting, and flat documents offer little recoverable surface geometry. Its value was as a diverse artifact cue, not as a physically complete reconstruction method.

### 3.2 Known-layout public specialist

The specialist fine-tuned ImageNet-pretrained ConvNeXt-Tiny at `224 x 224` over the full available training set with fraud and document-type heads. It captured strong visual and layout evidence and produced the best public score, but its perfect type prediction showed that its representation remained strongly coupled to known templates.

### 3.3 Type-adversarial forensic specialist

The forensic branch used EfficientNet-B0 at `384 x 384`. A trainable `1 x 1` adapter mixed normalized RGB with a local high-pass residual,

```text
residual = image - local_mean_5x5(image),
```

and was initialized as an exact RGB identity. Capture augmentation covered JPEG recompression, blur, resampling, affine and perspective changes, color variation, and sensor-like noise. Balanced type/label batches limited majority-type shortcuts.

A gradient-reversal document-type head encouraged the shared representation to discard template identity. The resulting branch achieved `0.998826` AUC on the all-type random validation set while reducing five-way type accuracy to `24.0%`, close to the `20%` chance level. Longer LOTO training degraded the strict low-BPCER objective, so the frozen branch used the one-epoch setting.

### 3.4 Conservative percentile-rank fusion

Raw probabilities from independently calibrated networks are not directly comparable. The final ensemble converted each complete test-score vector to percentage ranks before blending:

```text
fraud_score = 0.85 * rank(ConvNeXt-Tiny-224)
            + 0.15 * rank(residual EfficientNet-B0-384)
```

The small forensic weight was selected to preserve the global branch's low-BPCER behavior. Sequential checkpoint loading bounded peak GPU memory and allowed the same no-network container to reproduce both final variants.

### 3.5 Post-freeze high-resolution foundation probe

After the competition freeze, a research-only track evaluated frozen DINOv2-B/14 features at `518 px` over five deterministic crops. Logistic probes compared mean, mean-plus-max, and mean-plus-max-plus-standard-deviation aggregation on a capped `EGYPT/DL` LOTO experiment. This track did not alter, blend into, or replace the frozen competition submissions.

## 4. Results

The machine-readable values used below are collected in [assets/freuid_final_results_2026_07_19.csv](assets/freuid_final_results_2026_07_19.csv).

### 4.1 Small-data method development

| Method | Accuracy | ROC AUC | APCER @ 1% BPCER | AuDET proxy |
| --- | ---: | ---: | ---: | ---: |
| photometric logistic regression | 0.8063 | 0.8709 | 0.4875 | 0.1323 |
| `combined_v3` gradient boosting | 0.8063 | 0.9198 | 0.3375 | 0.0843 |
| conventional rank fusion | 0.8313 | 0.9135 | 0.3125 | 0.0908 |
| frozen ConvNeXt logistic probe | 0.8500 | 0.9227 | 0.3500 | 0.0814 |
| four-branch rank fusion | **0.8688** | **0.9452** | **0.2625** | **0.0595** |

These `320/160` results established that neural and conventional branches were complementary, but the slice was too small and too close to the training distribution to support a final competition claim.

### 4.2 Hard unseen-type validation

| Evaluation | Method | APCER @ 1% BPCER | AuDET proxy | ROC AUC |
| --- | --- | ---: | ---: | ---: |
| clean EGYPT LOTO | global EfficientNet 384 | 0.404 | 0.171010 | 0.829248 |
| clean EGYPT LOTO | residual/type-adversarial EfficientNet | 0.442 | **0.161552** | **0.838920** |
| clean EGYPT LOTO | frozen 85/15 rank ensemble | **0.402** | 0.167956 | 0.832336 |
| screenshot EGYPT LOTO | global EfficientNet 384 | 0.422 | 0.171834 | 0.828460 |
| screenshot EGYPT LOTO | frozen 85/15 rank ensemble | **0.416** | **0.168740** | **0.831586** |

The standalone type-adversarial branch improved ranking and average operating behavior, but not strict APCER. The conservative ensemble recovered the strongest APCER while preserving part of the branch's AUC and capture robustness. This is why the final weight was `15%`, not an equal blend.

### 4.3 Official leaderboard outcome

| Final candidate | Public score | Private score | Role |
| --- | ---: | ---: | --- |
| ConvNeXt public specialist | **0.25470** | 0.74443 | known-layout specialist |
| type-adversarial 85/15 rank ensemble | 0.25799 | **0.73118** | unseen-type/capture candidate |

The OOD rank ensemble gave up `0.00329` on the public split but improved the private score by `0.01325`. The selected score placed the team 46th on the final leaderboard. Because both candidates scored all `142,818` IDs and differed only in the frozen inference strategy, this is the cleanest evidence in the project that the conservative forensic branch helped under private distribution shift.

Earlier metadata, conventional, and raw-fusion submissions are useful as development history but are not comparable private baselines: they were created before the complete private image release and used fallback values for unreleased IDs. Their `0.98050`-range private scores should not be interpreted as full private inference results.

### 4.4 Post-freeze DINOv2 probe

| Five-crop aggregation | Accuracy at 0.5 | ROC AUC | APCER @ 1% BPCER | AuDET proxy |
| --- | ---: | ---: | ---: | ---: |
| mean | 0.5000 | **0.8585** | **0.6005** | **0.1415** |
| mean + max | 0.5000 | 0.8428 | 0.6495 | 0.1572 |
| mean + max + std | 0.5000 | 0.7998 | 0.8350 | 0.2002 |

Mean pooling provided useful held-out-type ranking, but all variants classified every holdout sample as fraud at threshold `0.5`. Adding max and standard-deviation statistics made every reported operating metric worse. The result supports high-resolution foundation features as a ranking branch, not as a calibrated replacement for the frozen final system.

## 5. What Worked

### 5.1 Leave-one-type-out evaluation exposed the real problem

Random validation was almost saturated and document type was trivially recoverable. LOTO made template dependence visible and produced actionable failure cases. It was the most important experimental-design choice in the project.

### 5.2 Type suppression improved private generalization

Reducing type accuracy from `100%` in the specialist to `24.0%` in the forensic branch did not by itself guarantee a better strict operating point. However, the private leaderboard favored the candidate that retained this branch, consistent with the intended shift-robustness role.

### 5.3 Percentile ranks were safer than raw probability fusion

The frozen `85/15` rank candidate scored `0.25799` publicly, compared with `0.27166` for the earlier `65/35` raw neural fusion. Although the weights also differ, the result and the LOTO studies both favor conservative, scale-insensitive blending over aggressive raw-score averaging.

### 5.4 Small, complementary ensembles beat isolated complexity

The final improvement came from preserving the specialist and adding a small complementary branch. The type-adversarial model was not uniformly best alone. Its value appeared when the ensemble retained the specialist's operating point while borrowing cross-type ranking evidence.

### 5.5 Reproducibility work was part of model quality

Exact score hashes, ID-order validation, finite-range checks, sequential model loading, variant-controlled Docker inference, and no-network CI verification prevented packaging errors from obscuring the scientific comparison. Frozen private inference covered `134,997` private images in `3,415.81 s` on two Tesla T4 GPUs, and the final merge preserved all `7,821` public rows.

## 6. What Did Not Work Well

### 6.1 Metadata shortcuts

File-size and released-metadata baselines produced public scores near `1.0` and did not provide meaningful fraud evidence. They were useful only as format and direction checks.

### 6.2 Single-image photometric geometry as a primary detector

Photometric features were competitive on very small balanced slices and helped early fusion, but the acquisition conditions do not support true normal recovery. As data and neural capacity increased, layout-aware and residual neural representations became the more credible primary models.

### 6.3 Random-split near-perfect validation

The all-type validation AUC of `0.998826` was a useful training check but a poor estimate of private generalization. Perfect type prediction in the public specialist demonstrated why this number could not be treated as the headline result.

### 6.4 Aggressive or uncalibrated fusion

The earlier raw `65/35` neural fusion underperformed the conservative rank blend publicly. The standalone forensic branch also improved AUC while worsening APCER. Ranking quality and low-BPCER behavior must therefore be optimized separately.

### 6.5 High-resolution DINOv2 calibration

The post-freeze DINOv2 probe separated classes but failed completely at the default decision threshold. The predeclared `mean + max + std` aggregation was also the worst measured option. Future work would require nested LOTO or source-aware out-of-fold calibration rather than test-time threshold adjustment.

## 7. Reproducibility and Release Record

The immutable release [`freuid-final-2026-07-13`](https://github.com/DrStrangel0ve/ai-image-forensics-comparison/releases/tag/freuid-final-2026-07-13) contains:

- `freuid_final_package_2026_07_13.zip`;
- `freuid_frozen_stack_v3_2026_07_13.zip`; and
- `freuid_short_report_2026_07_13.pdf`.

The release records the exact checkpoint and CSV SHA-256 values. GitHub Actions independently built the frozen container on Linux and verified both `public_specialist` and `ood_rank` under `docker run --network none`. The detailed frozen report is [freuid_short_report_2026_07_13.md](freuid_short_report_2026_07_13.md), private inference is documented in [freuid_private_final_inference_2026_07_14.md](freuid_private_final_inference_2026_07_14.md), and the final organizer audit is [freuid_final_reminder_audit_2026_07_16.md](freuid_final_reminder_audit_2026_07_16.md).

Raw competition images, local experiment directories, Python environments, and downloaded pretrained weights are intentionally excluded from Git. The release package—not the local cache—is the archival reproduction source.

## 8. Limitations

- Only five labeled training document types were available, while the private distribution was not labeled for post-hoc per-type analysis.
- The final leaderboard supplies a scalar result, so the private improvement cannot be attributed to a specific unseen type, manipulation, or capture condition.
- LOTO is a proxy for private shift, not a direct estimate of the private score.
- The two final candidates differ in architecture and fusion, so the private result is not a controlled causal estimate of gradient reversal alone.
- The high-resolution DINOv2 experiment was post-freeze, capped, and research-only.
- No external document-image data were used; broader real-world coverage remains unmeasured.
- The project studied defensive fraud detection. Scores should support review and triage rather than automatic rejection of identity documents without human oversight.

## 9. Conclusion

The FREUID track ends with a conservative result rather than a claim of solved document fraud. A strong ConvNeXt specialist was best on the released public layouts. A small type-adversarial residual branch, added through percentile-rank fusion, was better on the complete private distribution and delivered the team's final `0.73118` score and 46th-place finish. The most transferable lesson is methodological: when document layout is an easy shortcut, credible evaluation must hold out document types, stress capture conditions, separate ranking from calibration, and freeze the complete inference path before reading the final result.

The competition track is now closed. Further work, if resumed, should be treated as a new project with explicit nested cross-type calibration and independently justified high-resolution foundation features rather than continued tuning against the completed FREUID leaderboard.
