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
- CLIP/DINO frozen-encoder aliases are implemented, and the foundation-baseline table now compares ConvNeXt, DINOv2-small, and CLIP ViT-B/32 under the same linear-probe protocol.
- A bounded DINOv2-small smoke run already reaches 0.7059 AUC on Ishu -> source-balanced MS COCOAI transfer after training on only 80 Ishu images, making a full three-seed DINOv2 run worth prioritizing.
- The full three-seed DINOv2-small run reaches 0.6133 mean accuracy / 0.7063 mean AUC on Ishu -> source-balanced MS COCOAI, close to ConvNeXt's 0.6163 / 0.7139 but with higher fake recall and lower precision.
- Adding DINOv2 as a fifth SCP-Fusion branch raises transfer AUC to 0.7503; source-calibrated five-branch fusion reaches 0.6127 accuracy / 0.3062 Brier / 0.2938 ECE and improves strict source-heldout triage to about 26% coverage at 80% decided-case accuracy.
- The full three-seed CLIP ViT-B/32 run changes the foundation-baseline story: standalone CLIP reaches 0.9152 / 0.9701 on Ishu and 0.6363 / 0.8641 on Ishu -> source-balanced MS COCOAI, making it the strongest cross-domain ranker and source-heldout triage model so far.
- The reverse foundation run keeps CLIP's ranking lead but exposes a deployment split: on MS COCOAI -> Ishu, CLIP reaches the best AUC at 0.8243, while ConvNeXt has the best default/source-threshold accuracy at 0.6579 / 0.6784.
- The reverse neural/fusion run adds the missing baselines: all-branch score fusion reaches the best reverse AUC at 0.8285, while physics-guided ResNet-18 has the best default/source-threshold accuracy at 0.6871 / 0.6813 and the best Brier/ECE.
- A reverse fusion-regularization probe improves the AUC frontier to 0.8406 with branch dropout and improves fusion Brier/ECE to 0.2213 / 0.2091 with strong regularization, making utility-aware fusion the next method-development target.
- A source-utility threshold strategy is now implemented in `scripts/fuse_prediction_scores.py`. It can tune fused-score thresholds using asymmetric fake-detection, real-clearance, real-FPR, and fake-miss weights, while preserving the existing source calibration split and fake-rate cap machinery.
- The first source-utility reverse sweep confirms that threshold-only utility selection does not beat the previous capped source-accuracy operating point: the best result is still 0.7222 accuracy / 0.8291 AUC with a 0.48 source fake-rate cap, while uncapped utility reaches 0.7193 accuracy with a higher target fake-call rate.
- The first source-utility model-selection follow-up is also negative/useful: full source-train utility selects over-firing fusion heads at 0.6520 accuracy with a 0.8216 target fake-call rate, while a 0.48 source fake-rate cap recovers 0.7193 accuracy but still trails the fixed capped threshold family.
- A source-heldout generator model-selection follow-up reaches the same conclusion under leave-one-generated-source-out scoring: unconstrained source-holdout utility still selects over-firing fusion heads at 0.6520 accuracy / 0.8216 target fake-call rate, while the 0.48 source fake-rate cap recovers 0.7193 accuracy / 0.8291 AUC.
- A source-heldout tuned fusion follow-up is the first training-side constrained utility win: selecting best worst-source utility under a 0.48 source fake-rate cap reaches 0.7339 accuracy / 0.8341 AUC on MS COCOAI -> Ishu, beating the fixed capped threshold family while still over-calling fake images at a 0.6813 target fake-call rate.
- A tuned-fusion source fake-rate constraint sweep improves the reverse SCP-Fusion frontier again: the 0.40 source cap reaches 0.7632 accuracy / 0.8361 AUC and drops the target fake-call rate to 0.5175.
- The opposite-direction Ishu -> MS threshold-objective diagnostic shows the cap lesson is directional: source fake-rate caps make the already conservative all-foundation fusion under-call generated images, while a less real-FPR-punitive source utility improves accuracy to 0.6470 but remains a sensitivity result.
- Adding CLIP to saved-score fusion improves the fusion family, with all-foundation SCP-Fusion reaching 0.7995 transfer AUC, but it still trails standalone CLIP; this motivates calibration-aware or source-heldout fusion training rather than only adding more branches.
- On Ishu -> source-balanced MS COCOAI, frozen ConvNeXt-Tiny has the best three-seed AUC at 0.7139, while physics-guided fusion has the best source-threshold accuracy at 0.6070.
- SCP-Fusion v0 score fusion over `combined_v3`, ResNet-18, physics-guided fusion, and frozen ConvNeXt-Tiny improves Ishu -> MS COCOAI mean AUC to 0.7282, with oracle accuracy 0.6793 but default accuracy only 0.5910.
- SCP-Fusion branch-dropout score fusion is implemented and exported with branch coefficients, but the first three-seed probe is negative/mixed: default accuracy 0.5923 versus 0.5910 for v0, with worse AUC, Brier, and ECE.
- Source-calibrated SCP-Fusion now reserves a held-out source calibration split and applies class-balanced temperature scaling; the first probe improves default accuracy to 0.6073 and Brier/ECE to 0.3123 / 0.2947, while AUC dips from 0.7282 to 0.7242.
- Source-heldout stress testing shows source-calibrated fusion has the best Brier/ECE and fake detection after target-source temperature scaling, but SCP-Fusion v0 remains stronger for the 10% two-threshold triage operating point.
- Calibration diagnostics show SCP-Fusion v0 has the best cross-domain Brier score, 0.3190, while all strong ranking models under-call generated MS COCOAI images at the default threshold.
- Source-heldout post-hoc calibration shows class-balanced temperature scaling improves Brier/ECE without changing decisions, while Platt/isotonic calibration can overfit non-heldout source priors and inflate real-image false positives.
- Source-heldout triage mode shows frozen ConvNeXt and SCP-Fusion can make high-confidence decisions on about 21-24% of target images with roughly 75% triage accuracy at a strict 5% calibration error budget.
- Source-heldout calibration and triage summary assets now include 95% deterministic bootstrap confidence intervals over held-out source/seed rows, so poster/paper tables can report uncertainty without recomputing predictions.
- Publication assets now include six generated diagnostic figures, captions, a pipeline diagram, and a DFRWS-style abstract draft.
- A submission-readiness snapshot now maps the current assets and gaps to DFRWS, WIFS, and DFF; reverse-direction foundation, neural, physics-guided, score-fusion, and regularization baselines are now checked in, and the next paper-critical gap is held-out-generator utility-aware fusion training.
- Public sharing assets now include `CITATION.cff` and a reproducibility checklist that separates checked-in code/reports from external datasets and ignored model artifacts.
- Qualitative failure grids now show seed-17 false positives, false negatives, and branch disagreements for SCP-Fusion on Ishu -> MS COCOAI.
- Source-heldout diagnostics show that naive source-threshold transfer can produce extreme real-image false-positive rates, so calibration and source-aware validation are first-class research questions.
- A focused physics-guided-vs-ResNet note now sharpens the claim: physics-guided ResNet is stronger on Ishu, robustness, and several transfer diagnostics, but vanilla ResNet still wins MS COCOAI in-domain validation.
- `combined_v4` now implements the planned reconstruction, multiscale frequency, chroma, and JPEG feature expansion; the first bounded smoke probe is usable but trails `combined_v3`, so the next claim needs full repeated-seed ablation rather than a single split.
- A bounded three-seed feature-selection probe makes `combined_v4` promising again: select-k60 v4 reaches 0.7389 mean accuracy / 0.8219 mean AUC versus 0.7278 / 0.8033 for `combined_v3`; select-k80 has the best bounded accuracy at 0.7611 but lower AUC.
- A medium three-seed probe with 240 training images per seed makes raw `combined_v4` the best ranking/accuracy candidate at 0.7544 mean accuracy / 0.8315 mean AUC, while select-k60 has the best Brier/ECE.
- Repeated benchmark and feature-ablation summary scripts now emit 95% deterministic bootstrap confidence intervals, and the checked-in `combined_v4` ablation assets have been regenerated with those intervals.

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

Feasible now. The current repo is already enough for a poster, especially with the CLIP frontier and source-heldout diagnostics. A first abstract and figure set are in `reports/publication_assets_2026_06_12.md`, and the submission package is organized in `reports/submission_readiness_2026_06_13.md`.

## Target 2: IEEE WIFS 2026

Link: https://wifs2026.utt.fr/call-for-papers

Deadline: July 15, 2026. Papers are limited to 6 pages including bibliography and figures.

Best angle:

A compact, serious benchmark paper. WIFS will expect tighter experimental methodology, careful claims, and clean ablations. The best WIFS framing is not "we built the best detector." It is "we expose score-separation and calibration failures across physical, neural, and pretrained feature families, and propose a source-aware fusion protocol."

Minimum additions before submission:

- Run a full repeated-seed `combined_v4` ablation against `combined_v3`, using the new `--select-k` path and at least one stronger classifier.
- Regenerate publication figures with the new CLIP/DINOv2 foundation/fusion and reverse-transfer regularization results, then add either utility-aware reverse fusion or Ishu -> another recent Kaggle dataset.
- Expand the new source-heldout calibration split with a calibration-aware model/training ablation.
- Use the new source-heldout confidence intervals in poster/paper tables and add seed-variability plots where space permits.
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
  - test whether `combined_v4` helps as a direct branch, after feature selection, or only as a source-diagnostic ablation;
  - extend the new temperature-scaled source calibration head into source-heldout calibration;
  - compare against the current saved-score logistic fusion.
  - branch dropout is now implemented and tested as a negative/mixed result, so prioritize source-aware calibration before adding more score-stack regularization.
- Expand the first qualitative grids into a short explainability section using feature coefficients and source-level score distributions.
- Add one more qualitative grid from a second seed or dataset direction.
- Add a reproducibility appendix with exact commands and dataset audit notes.

Recommended title:

**SCP-Fusion: Source-Calibrated Physical and Foundation Features for Robust AI-Generated Image Forensics**

Status:

Best target for a full workshop paper. It gives room for the repo's practical engineering and diagnostic work, not just a leaderboard result.

## Next Technical Milestones

1. Improve logit-level fusion:
   - current saved-score logistic fusion reaches 0.7282 mean AUC on Ishu -> MS COCOAI;
   - branch-dropout score fusion is implemented, but it slightly hurts AUC/Brier/ECE in the first three-seed probe;
   - held-out source-domain temperature scaling improves default accuracy and Brier/ECE, so next step is source-heldout calibration rather than merely adding more branches.
   - source-heldout stress testing shows calibration and triage prefer different score geometry, so next step is joint calibration/triage tuning.
   - utility-tuned source-heldout triage is now implemented; it selects raw SCP-Fusion v0 scores in all 15 folds and beats the calibrated variants on target utility, so the paper should report calibration quality and triage utility as separate objectives.
   - utility-aware binary source-threshold selection, model-side selection, source-heldout generator selection, constrained source-heldout fusion tuning, and source fake-rate constraint sweeps are now implemented; the opposite-direction threshold-objective sweep shows the 0.40 cap frontier is directional, so the next step is validating it under robustness transforms or a larger source split rather than assuming it transfers.

2. Validate `combined_v4`:
   - implementation is checked in with reconstruction, multiring FFT, chroma, and JPEG features;
   - raw v4 trails `combined_v3` in the first 80/40 smoke probe, but select-k v4 leads the bounded three-seed 120/60 probe by mean AUC;
   - the expanded bounded sweep suggests `k=60` for ranking/AUC and `k=80` for default accuracy, while histogram-gradient boosting is weak on the small split;
   - the medium 240-train run suggests raw v4 may overtake selected v4 once there is enough data, with select-k60 still best calibrated;
   - interval-aware summaries show the current v4 lead is narrow, so frame it as model-selection evidence until the larger repeated-seed run lands;
   - the next step is a full repeated-seed Ishu run for raw v4 and select-k60, then MS COCOAI/source-heldout transfer before adding v4 into score fusion.

3. Improve calibration:
   - Brier score, expected calibration error, reliability curves, and source-heldout post-hoc calibration are now implemented;
   - utility-tuned triage shows post-hoc temperature scaling is not enough for the high-confidence operating mode;
   - next step is a calibration-aware training objective that improves held-out fake recall without inflating real false positives.

4. Add failure-case export:
   - false positives, false negatives, and high-disagreement grids are now implemented for seed 17;
   - next step is repeating the export for another seed or reverse transfer.

5. Draft paper assets:
   - one shared results figure is generated;
   - one pipeline figure draft is written;
   - source-heldout calibration and triage figures are generated;
   - the DINOv2 SCP-Fusion gain figure is generated;
   - the CLIP transfer frontier figure is generated;
   - utility-tuned score-fusion triage now has a generated figure;
   - one qualitative failure grid set is generated.

## Recommended Order

1. DFRWS poster first, because it can be submitted with current evidence.
2. DFF full paper second, because it best matches the project.
3. WIFS third, using the same core but with stricter claims and a more compact paper.
