# Submission Readiness Snapshot

Run date: 2026-06-13

This snapshot turns the current benchmark state into an action plan for the three active publication targets. It is intentionally conservative: the project now has a strong poster story and a plausible workshop-paper story, but the paper versions still need one or two breadth checks before submission.

## Verified Targets

| target | current deadline | current fit | source |
| --- | --- | --- | --- |
| DFRWS-USA 2026 poster/demo | rolling poster deadline until 2026-07-07 | ready to draft now | https://dfrws.org/call-for-papers-is-open-for-dfrws-usa-2026/ |
| IEEE WIFS 2026 paper | 2026-07-15, 6 pages including references/figures | feasible, needs tighter breadth | https://wifs2026.utt.fr/call-for-papers |
| DFF-2026 at ACM Multimedia | 2026-07-16, OpenReview workshop deadline | best full-paper fit | https://openreview.net/group?id=acmmm.org/ACMMM/2026/Workshop/DFF |

## Current Lead Result

The strongest current story is not "SCP-Fusion beats every foundation model." The stronger and more defensible claim is:

**Single-image AI-generated image detection needs source-heldout evaluation because ranking, calibration, and high-confidence triage disagree under generator shift. Frozen CLIP is the strongest standalone transfer branch so far, while SCP-Fusion exposes which physical, neural, and foundation signals help or fail under source shift.**

Core numbers to lead with:

| result | value | interpretation |
| --- | ---: | --- |
| `combined_v3` vs ResNet-18 on Ishu | both 0.8246 mean accuracy | conventional and neural baselines are genuinely close in-domain |
| physics-guided ResNet + `combined_v3` on Ishu | 0.8450 accuracy / 0.9177 AUC | physics-informed fusion helps same-domain and robustness |
| frozen CLIP on Ishu -> MS COCOAI | 0.6363 accuracy / 0.8641 AUC | strongest standalone transfer ranking |
| all-foundation SCP-Fusion on Ishu -> MS COCOAI | 0.6163 accuracy / 0.7995 AUC | CLIP improves fusion, but fusion still trails CLIP ranking |
| CLIP source-heldout triage, strict 5% budget | 0.4747 coverage / 0.9261 decided-case accuracy | best current forensic triage operating point |
| DINOv2-enhanced source-calibrated fusion | 0.6127 accuracy / 0.3062 Brier / 0.2938 ECE | best calibrated fusion-family operating point before CLIP |

## Checked-In Figure Package

| figure | file | use |
| --- | --- | --- |
| pipeline diagram | `reports/publication_assets_2026_06_12.md` | poster overview and paper method schematic |
| transfer ranking/calibration | `reports/assets/publication_cross_domain_calibration.png` | historical baseline comparison |
| source-heldout calibration | `reports/assets/publication_source_heldout_calibration.png` | probability-quality evidence |
| source-heldout triage | `reports/assets/publication_triage_operating_points.png` | operational forensic framing |
| DINOv2 fusion upgrade | `reports/assets/publication_score_fusion_dinov2_gain.png` | branch-complementarity evidence |
| CLIP frontier | `reports/assets/publication_score_fusion_clip_frontier.png` | newest headline result |
| qualitative misses | `reports/assets/qualitative_seed17_scp_fusion_false_negatives.png` | failure-mode/explainability panel |

## DFRWS Poster Draft Plan

Recommended title:

**When AI Image Detectors Travel: Source-Heldout Diagnostics for Physical, Neural, and Frozen-Encoder Forensics**

Poster layout:

| panel | content |
| --- | --- |
| Problem | same-dataset scores hide generator/source shift |
| Methods | physical/signal features, ResNet-18, physics-guided fusion, frozen ConvNeXt/DINOv2/CLIP, saved-score fusion |
| Benchmark | Ishu same-domain, Ishu -> source-balanced MS COCOAI, source-heldout calibration/triage |
| Main result | CLIP transfer frontier plus all-foundation SCP-Fusion comparison |
| Failure modes | false negatives and branch disagreement grids |
| Reproducibility | public GitHub, dataset export commands, ignored raw data/model artifacts |

Abstract draft:

AI-generated image detectors often look reliable when trained and tested on the same benchmark, but their decisions can change sharply when generator family, dataset source, or image processing changes. We evaluate real-vs-generated image detection as a source-heldout forensic problem rather than a closed-set classification task. The project compares handcrafted physical/signal features, fine-tuned ResNet-18, a physics-guided neural fusion model, frozen ConvNeXt, DINOv2, and CLIP encoders, and a lightweight saved-score fusion model named SCP-Fusion. Experiments include repeated-seed same-domain runs, cross-dataset transfer from Ishu AI-vs-real images to a source-balanced Defactify/MS COCOAI split, calibration diagnostics, and two-threshold forensic triage.

The results show that ranking, probability calibration, and binary decision quality are separate forensic questions. `combined_v3` conventional features and ResNet-18 tie on Ishu same-domain accuracy, while physics-guided fusion improves same-domain and robustness results. Frozen CLIP ViT-B/32 is the strongest cross-domain branch so far, reaching 0.8641 mean AUC on Ishu -> MS COCOAI and the best source-heldout triage point: 47% target coverage at 92.6% decided-case accuracy under a strict 5% calibration-error budget. Adding CLIP improves all-foundation SCP-Fusion to 0.7995 mean transfer AUC, but score-level fusion still trails standalone CLIP, showing that more branches are not automatically better under source shift. These findings support source-aware evaluation, frozen foundation baselines, and calibrated triage as practical requirements for AI-image forensics.

## WIFS/DFF Paper Skeleton

| section | required content |
| --- | --- |
| Introduction | source shift, why same-domain accuracy is misleading, SCP-Fusion as diagnostic fusion rather than pure leaderboard claim |
| Related Work | AI-generated image detection, physical/signal forensics, foundation encoders, calibration and triage |
| Data and Audit | Ishu, MS COCOAI, source-balanced export, duplicate/leakage audit notes |
| Methods | `combined_v3/v4`, ResNet-18, physics-guided ResNet, frozen ConvNeXt/DINOv2/CLIP, score fusion, calibration |
| Experiments | repeated seeds, source-balanced transfer, source-heldout calibration, source-heldout triage, robustness transforms |
| Results | lead with CLIP frontier, then explain where SCP-Fusion helps and where it underperforms |
| Failure Analysis | source-wise miss patterns, false-negative grids, branch coefficients |
| Limitations | single-image photometric proxy, small source validation split for fusion, no production detector claim |
| Reproducibility | public repo, commands, checked-in compact summaries, external datasets |

## Remaining Gaps Before Paper Submission

Priority order:

1. Add one more direction, preferably MS COCOAI -> Ishu or Ishu -> another recent Kaggle dataset, for CLIP, ConvNeXt, DINOv2, ResNet, physics-guided fusion, and SCP-Fusion.
2. Add source-heldout or calibration-aware fusion training so SCP-Fusion is not merely a score stack trained on a small Ishu validation split.
3. Run `combined_v4` full repeated-seed transfer and decide whether it belongs in the main method or stays an ablation.
4. Add another qualitative grid from a second seed or reverse transfer.
5. Convert the DFRWS poster draft into a one-page visual artifact after the next experiment.

## Suggested Next Experiment

The best next technical experiment is **reverse transfer with foundation branches**:

- train on the source-balanced MS COCOAI export;
- evaluate on Ishu;
- include `combined_v3`, ResNet-18, physics-guided fusion, frozen ConvNeXt, DINOv2, CLIP, and score fusion where saved prediction alignment allows it;
- report default accuracy, AUC, Brier/ECE, and source-threshold accuracy.

Why this first: CLIP now dominates Ishu -> MS COCOAI. A reverse-direction check will reveal whether CLIP's advantage is broad transfer strength or partly a property of this source/target direction.
