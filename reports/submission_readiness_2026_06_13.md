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
| Ishu -> MS threshold-objective diagnostic | 0.6470 accuracy / 0.7995 AUC / 0.1863 fake-call rate | source utility can improve decisions on saved fusion scores, but still under-calls generated images |
| frozen CLIP on MS COCOAI -> Ishu | 0.6228 accuracy / 0.8243 AUC | reverse direction keeps CLIP's ranking lead but shows threshold shift |
| frozen ConvNeXt on MS COCOAI -> Ishu | 0.6579 default accuracy / 0.6784 source-threshold accuracy | best reverse-direction operating point among foundation branches |
| all-branch fusion on MS COCOAI -> Ishu | 0.6579 accuracy / 0.8285 AUC | best reverse-direction AUC, but still over-calls generated images |
| branch-dropout fusion on MS COCOAI -> Ishu | 0.6520 accuracy / 0.8406 AUC | new reverse AUC frontier, still poorly calibrated |
| strongly regularized fusion on MS COCOAI -> Ishu | 0.6608 accuracy / 0.2213 Brier / 0.2091 ECE | best reverse fusion probability quality |
| capped source-threshold fusion on MS COCOAI -> Ishu | 0.7222 accuracy / 0.8291 AUC / 0.2188 Brier | best reverse fusion operating point so far |
| source-heldout tuned fusion on MS COCOAI -> Ishu | 0.7339 accuracy / 0.8341 AUC / 0.2748 Brier | first training-side constrained utility win |
| tuned fusion cap sweep on MS COCOAI -> Ishu | 0.7632 accuracy / 0.8361 AUC / 0.5175 fake-call rate | best reverse SCP-Fusion operating point so far |
| tuned fusion cap_0p4 under JPEG70 | 0.7661 accuracy / 0.8485 AUC / 0.4678 fake-call rate | first robustness check for the reverse cap frontier |
| tuned fusion cap_0p4 under JPEG50 | 0.7515 accuracy / 0.8309 AUC / 0.4240 fake-call rate | stronger recompression only mildly degrades the clean cap frontier |
| tuned fusion cap_0p4 under JPEG30 | 0.7076 accuracy / 0.8167 AUC / 0.3450 fake-call rate | harsh recompression causes under-calling of generated images |
| tuned fusion cap_0p4 under blur1 | 0.7105 accuracy / 0.7872 AUC / 0.5585 fake-call rate | blur is the clearest current transform weakness |
| tuned fusion cap_0p4 under resize_half | 0.7164 accuracy / 0.7816 AUC / 0.5526 fake-call rate | half-resolution resize similarly hurts ranking and calibration |
| tuned fusion cap_0p4 under crop85 | 0.7251 accuracy / 0.8227 AUC / 0.5205 fake-call rate | crop is weaker than clean but less damaging than blur/resize |
| tuned fusion cap_0p4 under noise3 | 0.7690 accuracy / 0.8704 AUC / 0.4708 fake-call rate | small sensor-like noise is not a current weakness |
| tuned fusion cap_0p4 under screenshot | 0.7310 accuracy / 0.7965 AUC / 0.5263 fake-call rate | screenshot-style resize plus recompression is a realistic stressor |
| tuned fusion cap_0p4 under social_square | 0.7778 accuracy / 0.8474 AUC / 0.5088 fake-call rate | centered social crop plus recompression is stable in this bounded check |
| physics-guided ResNet on MS COCOAI -> Ishu | 0.6871 default accuracy / 0.6813 source-threshold accuracy | best default-threshold single-model operating point and ECE anchor |
| CLIP source-heldout triage, strict 5% budget | 0.4747 coverage / 0.9261 decided-case accuracy | best current forensic triage operating point |
| DINOv2-enhanced source-calibrated fusion | 0.6127 accuracy / 0.3062 Brier / 0.2938 ECE | best calibrated fusion-family operating point before CLIP |

## Claim Evidence Matrix

The current submission claims are audited in a generated claim-evidence matrix:

- Markdown: `reports/assets/claim_evidence_matrix.md`
- CSV: `reports/assets/claim_evidence_matrix.csv`

The matrix marks each claim as `ready`, `ready_with_caveat`, or `needs_more_evidence`, cites the exact rows from `reports/assets/publication_core_results.csv`, and names the paper/poster artifact that should carry the claim. The important editorial outcome is that the CLIP transfer frontier, source-shift split, and high-confidence triage claims are ready now; the physics-guided, SCP-Fusion, and `combined_v4` claims are ready only with caveats. The completed v4 gate is summarized in `reports/combined_v4_full_transfer_summary_2026_06_13.md` and supports v4 as an appendix ablation, not a main-method replacement.

## Checked-In Figure Package

| figure | file | use |
| --- | --- | --- |
| pipeline diagram | `reports/publication_assets_2026_06_12.md` | poster overview and paper method schematic |
| poster transfer panel | `reports/assets/dfrws_poster_transfer_panel.png` and `.svg` | large-label DFRWS headline replacement for dense paper figure |
| poster robustness panel | `reports/assets/dfrws_poster_robustness_panel.png` and `.svg` | large-label DFRWS robustness caveat panel |
| transfer ranking/calibration | `reports/assets/publication_cross_domain_calibration.png` | historical baseline comparison |
| source-heldout calibration | `reports/assets/publication_source_heldout_calibration.png` | probability-quality evidence |
| source-heldout triage | `reports/assets/publication_triage_operating_points.png` | operational forensic framing |
| DINOv2 fusion upgrade | `reports/assets/publication_score_fusion_dinov2_gain.png` | branch-complementarity evidence |
| CLIP frontier | `reports/assets/publication_score_fusion_clip_frontier.png` | newest headline result |
| reverse fusion tradeoff | `reports/assets/publication_reverse_fusion_tradeoff.png` | utility-aware fusion motivation |
| reverse operating points | `reports/assets/publication_reverse_operating_points.png` | source-threshold fusion headline |
| reverse transform robustness | `reports/assets/publication_reverse_transform_robustness.png` | tuned-fusion robustness stress panel |
| qualitative misses | `reports/assets/qualitative_seed17_scp_fusion_false_negatives.png` | failure-mode/explainability panel |

## DFRWS Poster Draft Plan

Generated brief:

- Markdown: `reports/dfrws_poster_brief_2026_06_13.md`
- Key numbers CSV: `reports/assets/dfrws_poster_key_numbers.csv`
- Rebuild: `python scripts\build_dfrws_poster_brief.py`

Editable poster draft:

- Report: `reports/dfrws_poster_draft_2026_06_13.md`
- PowerPoint: `reports/assets/dfrws_poster_draft_2026_06_13.pptx`
- PNG preview: `reports/assets/dfrws_poster_draft_2026_06_13.png`

Updated poster draft v2:

- Report: `reports/dfrws_poster_draft_v2_2026_06_13.md`
- PowerPoint: `reports/assets/dfrws_poster_draft_v2_2026_06_13.pptx`
- PNG preview: `reports/assets/dfrws_poster_draft_v2_2026_06_13.png`

Poster-native figure pack:

- Report: `reports/dfrws_poster_native_figures_2026_06_13.md`
- Transfer panel: `reports/assets/dfrws_poster_transfer_panel.png` / `.svg`
- Robustness panel: `reports/assets/dfrws_poster_robustness_panel.png` / `.svg`
- Rebuild: `python scripts\build_dfrws_poster_figures.py`

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

AI-generated image detectors often look reliable when trained and tested on the same benchmark, but their decisions can change sharply when generator family, dataset source, or image processing changes. We evaluate real-vs-generated image detection as a source-heldout forensic problem rather than a closed-set classification task. The project compares handcrafted physical/signal features, fine-tuned ResNet-18, a physics-guided neural fusion model, frozen ConvNeXt, DINOv2, and CLIP encoders, and a lightweight saved-score fusion model named SCP-Fusion. Experiments include repeated-seed same-domain runs, cross-dataset transfer between Ishu AI-vs-real images and a source-balanced Defactify/MS COCOAI split, calibration diagnostics, and two-threshold forensic triage.

The results show that ranking, probability calibration, and binary decision quality are separate forensic questions. `combined_v3` conventional features and ResNet-18 tie on Ishu same-domain accuracy, while physics-guided fusion improves same-domain and robustness results. Frozen CLIP ViT-B/32 is the strongest single cross-domain ranking branch so far, reaching 0.8641 mean AUC on Ishu -> MS COCOAI and 0.8243 mean AUC on MS COCOAI -> Ishu. Reverse branch-dropout fusion raises MS COCOAI -> Ishu AUC to 0.8406, and strongly regularized fusion improves Brier/ECE to 0.2213 / 0.2091, but physics-guided ResNet-18 still has the best reverse operating point at 0.6871 default accuracy. These findings support source-aware evaluation, frozen foundation baselines, and calibrated triage as practical requirements for AI-image forensics.

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

1. Extend tuned-fusion robustness to native-resolution and stronger in-the-wild perturbations. JPEG70, JPEG50, noise3, and social_square are comparatively stable, but JPEG30, blur1, resize_half, and screenshot-style roundtrips expose real transform weaknesses; the next gap is native-resolution tiling, stronger social-media processing, or a larger source split before calling the 0.40 reverse frontier contest-ready.
2. Try source-aware feature selection or a stronger regularized classifier for `combined_v4` only if the paper needs another ablation; the current gate keeps `combined_v3` as the main conventional baseline.
3. Add another qualitative grid from a second seed or reverse transfer.
4. If the final DFRWS format requires fully editable charts, rebuild the two v2 embedded raster panels as native PowerPoint primitives using the checked-in SVG/CSV panel sources.

## Suggested Next Experiment

The best next technical experiment is **source-heldout utility-aware reverse fusion**:

- use the checked-in `--threshold-strategy source_utility` implementation and the model-selection selector as fixed operating-point evaluators;
- extend the 0.40 source-cap tuned fusion robustness check from JPEG/blur/resize/crop/noise/screenshot/social-square to native-resolution tiling, stronger social-media processing, or a larger source split;
- combine the branch-dropout AUC gain with the strong-regularization Brier/ECE gain;
- preserve the current reverse suite as the fixed comparison table.

Why this first: branch dropout now has the best reverse AUC, strong regularization with capped held-out source thresholding gives the best fusion operating point, and physics-guided ResNet-18 remains the most interpretable calibration anchor. The paper story gets much stronger if SCP-Fusion can combine those three behaviors.
