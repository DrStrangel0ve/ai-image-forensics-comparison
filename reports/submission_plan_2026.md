# Submission Plan: DFRWS, WIFS, and DFF

Run date: 2026-06-12

This project now has three realistic publication paths. The same technical core can support all three, but each target needs a different amount of polish.

## Shared Core

Working title:

**Source-Calibrated Physical-Spectral Foundation Fusion for AI-Generated Image Forensics**

Core claim:

AI-generated image detection should be evaluated as a cross-source forensic problem, not a same-dataset classification problem. A detector that combines physical/signal features, pretrained foundation-style embeddings, and source-aware calibration exposes different failure modes than either handcrafted features or standard neural classifiers alone.

Current evidence:

- `combined_v3` and ResNet-18 are tied on Ishu same-domain three-seed accuracy: 0.8246 mean accuracy.
- Physics-guided ResNet-18 + `combined_v3` improves Ishu same-domain mean accuracy to 0.8450 and AUC to 0.9177.
- Frozen ConvNeXt-Tiny improves Ishu same-domain mean accuracy to 0.8947 and AUC to 0.9589.
- On Ishu -> source-balanced MS COCOAI, frozen ConvNeXt-Tiny has the best three-seed AUC at 0.7139, while physics-guided fusion has the best source-threshold accuracy at 0.6070.
- SCP-Fusion v0 score fusion over `combined_v3`, ResNet-18, physics-guided fusion, and frozen ConvNeXt-Tiny improves Ishu -> MS COCOAI mean AUC to 0.7282, with oracle accuracy 0.6793 but default accuracy only 0.5910.
- Calibration diagnostics show SCP-Fusion v0 has the best cross-domain Brier score, 0.3190, while all strong ranking models under-call generated MS COCOAI images at the default threshold.
- Source-heldout post-hoc calibration shows class-balanced temperature scaling improves Brier/ECE without changing decisions, while Platt/isotonic calibration can overfit non-heldout source priors and inflate real-image false positives.
- Source-heldout triage mode shows frozen ConvNeXt and SCP-Fusion can make high-confidence decisions on about 21-24% of target images with roughly 75% triage accuracy at a strict 5% calibration error budget.
- Source-heldout diagnostics show that naive source-threshold transfer can produce extreme real-image false-positive rates, so calibration and source-aware validation are first-class research questions.

## Target 1: DFRWS-USA 2026 Poster

Link: https://dfrws.org/call-for-papers-is-open-for-dfrws-usa-2026/

Deadline: July 7, 2026, rolling poster deadline.

Best angle:

A concise forensic-demonstration poster. This does not need to prove a final state-of-the-art method. It should show the benchmark, failure modes, and why physical/signal features plus frozen encoders are useful for investigative triage.

Deliverables:

- 250 to 500 word poster abstract.
- One-page poster draft.
- Three figures:
  - pipeline diagram;
  - same-domain vs cross-domain result table;
  - source-heldout triage or false-positive/fake-detection diagnostic.
- Public GitHub link and reproducibility checklist.

Recommended title:

**When AI Image Detectors Travel: Source-Heldout Diagnostics for Physical, Neural, and Frozen-Encoder Forensics**

Status:

Feasible now. The current repo is already enough for a poster, especially with the ConvNeXt and source-heldout diagnostics.

## Target 2: IEEE WIFS 2026

Link: https://wifs2026.utt.fr/call-for-papers

Deadline: July 15, 2026. Papers are limited to 6 pages including bibliography and figures.

Best angle:

A compact, serious benchmark paper. WIFS will expect tighter experimental methodology, careful claims, and clean ablations. The best WIFS framing is not "we built the best detector." It is "we expose score-separation and calibration failures across physical, neural, and pretrained feature families, and propose a source-aware fusion protocol."

Minimum additions before submission:

- Add a real `combined_v4` ablation or a reconstruction-feature ablation.
- Add frozen encoder cross-domain results for at least one more dataset direction, ideally MS COCOAI -> Ishu or Ishu -> another recent Kaggle dataset.
- Expand the new source-heldout calibration split with confidence intervals and a calibration-aware model/training ablation.
- Add source-heldout tables with confidence intervals or seed variability.
- Keep claims conservative and avoid implying classic multi-light photometric stereo on single-image datasets.

Recommended title:

**Source-Heldout Evaluation of Physical, Neural, and Frozen-Encoder Signals for AI-Generated Image Detection**

Status:

Feasible with two to three more strong experiments. The core story is solid, but WIFS needs cleaner breadth.

## Target 3: DFF-2026 at ACM Multimedia

Link: https://iplab.dmi.unict.it/mfs/acm-dff-ws-2026/

Deadline: July 16, 2026.

Best angle:

This is the cleanest full-paper target for the current project. DFF explicitly asks for detection and localization of manipulated or AI-generated content, robustness and generalization, dataset bias analysis, explainability, and real-world processing. The repo already has results for robustness transforms, cross-dataset transfer, source-heldout diagnostics, and physics-inspired features.

Minimum additions before submission:

- Upgrade SCP-Fusion v0 into v1:
  - add `combined_v4` or reconstruction features;
  - add a source-heldout or temperature-scaled calibration head;
  - compare against the current saved-score logistic fusion.
- Add a short explainability section using feature coefficients, source-level score distributions, and failure-case image grids.
- Add DFF-ready qualitative examples: real false positives, missed Midjourney/SD3 examples, and images where ConvNeXt and physics disagree.
- Add a reproducibility appendix with exact commands and dataset audit notes.

Recommended title:

**SCP-Fusion: Source-Calibrated Physical and Foundation Features for Robust AI-Generated Image Forensics**

Status:

Best target for a full workshop paper. It gives room for the repo's practical engineering and diagnostic work, not just a leaderboard result.

## Next Technical Milestones

1. Improve logit-level fusion:
   - current saved-score logistic fusion reaches 0.7282 mean AUC on Ishu -> MS COCOAI;
   - next step is source-aware calibration, not merely adding more branches.

2. Add `combined_v4`:
   - multiscale FFT/DCT bands;
   - residual entropy and NLM-like denoising residual summaries;
   - stronger JPEG periodicity;
   - multiscale photometric normal/integrability features.

3. Improve calibration:
   - Brier score, expected calibration error, reliability curves, and source-heldout post-hoc calibration are now implemented;
   - next step is a calibration-aware training objective that improves held-out fake recall without inflating real false positives.

4. Add failure-case export:
   - false positives;
   - false negatives;
   - high disagreement between ConvNeXt and physics-guided fusion;
   - source labels when available.

5. Draft paper assets:
   - one shared results table;
   - one pipeline figure;
   - one source-heldout calibration or triage diagnostic figure;
   - one qualitative failure grid.

## Recommended Order

1. DFRWS poster first, because it can be submitted with current evidence.
2. DFF full paper second, because it best matches the project.
3. WIFS third, using the same core but with stricter claims and a more compact paper.
