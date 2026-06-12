# Physics-Guided ResNet-18 vs Vanilla ResNet-18

Run date: 2026-06-12

This note distills the current comparison between vanilla ResNet-18 and the physics-guided ResNet-18 + `combined_v3` fusion model.

The fusion model is not a true multi-light photometric-stereo model. The available datasets are single-image real-vs-generated corpora. The "physics-guided" part is a second branch of standardized forensic features: pseudo-normal consistency, integrability proxies, noise residuals, JPEG recompression response, residual 8x8 periodicity, RGB residual correlation, local residual variance, FFT/chroma statistics, and related signal features.

## Headline

Physics-guided ResNet is the stronger forensic model on Ishu and on the Ishu robustness/transfer checks, but vanilla ResNet is still stronger on MS COCOAI when trained and evaluated in-domain.

| setting | vanilla ResNet-18 | physics-guided ResNet-18 | delta for physics-guided | read |
| --- | ---: | ---: | ---: | --- |
| Ishu same-domain, 3 seeds | 0.8246 acc / 0.8927 AUC | 0.8450 acc / 0.9177 AUC | +0.0204 acc / +0.0250 AUC | physics-guided wins |
| Ishu robustness, 12 transform checks | 0.8231 acc / 0.8904 AUC | 0.8443 acc / 0.9189 AUC | +0.0212 acc / +0.0285 AUC | physics-guided wins clearly |
| Ishu -> source-balanced MS COCOAI | 0.5800 acc / 0.6488 AUC | 0.6060 acc / 0.6637 AUC | +0.0260 acc / +0.0149 AUC | physics-guided modestly wins |
| Ishu -> MS COCOAI source-heldout diagnostic | 0.6796 default acc / 0.6308 AUC | 0.6971 default acc / 0.6481 AUC | +0.0175 acc / +0.0173 AUC | physics-guided modestly wins |
| MS COCOAI in-domain validation | 0.8160 acc / 0.8967 AUC | 0.7800 acc / 0.8790 AUC | -0.0360 acc / -0.0177 AUC | vanilla ResNet wins |
| MS COCOAI -> Ishu transfer | 0.6243 acc / 0.7003 AUC | 0.5873 acc / 0.7089 AUC | -0.0370 acc / +0.0086 AUC | physics-guided ranks better but thresholds worse |

## Where Physics Helps

On Ishu same-domain repeated splits, physics-guided fusion is the first method to beat both standalone `combined_v3` and vanilla ResNet-18 on mean accuracy and mean AUC:

| method | accuracy_mean | roc_auc_mean | accuracy_wins | auc_wins |
| --- | ---: | ---: | ---: | ---: |
| `combined_v3` | 0.8246 | 0.8942 | 1 | 1 |
| ResNet-18 | 0.8246 | 0.8927 | 0 | 0 |
| physics-guided ResNet-18 + `combined_v3` | 0.8450 | 0.9177 | 2 | 2 |

The robustness result is stronger. Across 12 seed-plus-transform checks, physics-guided fusion wins 10/12 by accuracy and 10/12 by AUC:

| method | n_checks | accuracy_mean | accuracy_delta_mean | roc_auc_mean | roc_auc_delta_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| `combined_v3` | 12 | 0.7924 | -0.0322 | 0.8751 | -0.0191 |
| ResNet-18 | 12 | 0.8231 | -0.0015 | 0.8904 | -0.0022 |
| physics-guided ResNet-18 + `combined_v3` | 12 | 0.8443 | -0.0007 | 0.9189 | +0.0012 |

This is the best evidence that the forensic branch is not just adding noise. It keeps most of ResNet's transform stability while adding signal that improves ranking.

## Where Vanilla ResNet Still Wins

On the source-balanced MS COCOAI validation slice, vanilla ResNet remains ahead:

| method | accuracy | precision | recall | f1 | roc_auc |
| --- | ---: | ---: | ---: | ---: | ---: |
| physics-guided ResNet-18 + `combined_v3` | 0.7800 | 0.7288 | 0.8920 | 0.8022 | 0.8790 |
| ResNet-18 | 0.8160 | 0.8098 | 0.8260 | 0.8178 | 0.8967 |

This matters for the paper claim. Physics guidance is not a universal replacement for visual representation learning. On MS COCOAI, the vanilla model's learned image prior is better matched to the dataset and has a cleaner threshold at 0.5.

## Transfer And Calibration

On Ishu -> source-balanced MS COCOAI, physics-guided fusion is modestly ahead of vanilla ResNet:

| method | accuracy | ROC AUC | Brier | ECE |
| --- | ---: | ---: | ---: | ---: |
| ResNet-18 | 0.5800 | 0.6488 | 0.3549 | 0.2453 |
| physics-guided fusion | 0.6060 | 0.6637 | 0.3367 | 0.2707 |

The fused model has better accuracy, AUC, and Brier score, but worse ECE. So it ranks and scores target images better on average, but it is not automatically better calibrated.

The stricter source-heldout diagnostic shows the same pattern:

| method | mean default accuracy | mean source-heldout threshold accuracy | mean oracle accuracy | mean AUC |
| --- | ---: | ---: | ---: | ---: |
| ResNet-18 | 0.6796 | 0.4206 | 0.7341 | 0.6308 |
| physics-guided fusion | 0.6971 | 0.4840 | 0.7411 | 0.6481 |

The physics-guided model is ahead, but both methods still struggle when thresholds are chosen from non-heldout generator sources. That is why source-aware calibration remains a core SCP-Fusion requirement.

## Paper Framing

The clean claim is:

> Physics-guided feature fusion improves ResNet-18 on repeated Ishu splits, common image transforms, and several cross-source diagnostics, but does not dominate every in-domain benchmark. It is best understood as a forensic robustness branch, not a universal image-classification upgrade.

This framing is stronger than claiming state of the art. It explains why the next SCP-Fusion version should combine:

- a learned image branch for dataset-adapted visual priors;
- a physical/signal branch for robustness and forensic interpretability;
- foundation encoder embeddings for broader semantic/texture generalization;
- source-aware calibration so cross-generator thresholds do not collapse.

## Source Reports

- `reports/ishu_ai_vs_real_2026_benchmark.md`
- `reports/ishu_physics_guided_robustness_3seed.md`
- `reports/ms_cocoai_source_balanced_validation.md`
- `reports/calibration_diagnostics_2026_06_12.md`
- `reports/source_holdout_diagnostics_2026_06_12.md`
