# SCP-Fusion: Source-Calibrated Physical and Foundation Features for Robust AI-Generated Image Forensics

Run date: 2026-06-14

Generated manuscript draft assembled from generated section drafts, abstract text, and the manuscript assembly map.
Author editing is still required before copying into the official venue template.

## Abstract

Robust detection of AI-generated images is not only a model-selection problem: detectors must survive new generator sources, real-world processing, and ambiguous operating points. We introduce SCP-Fusion as a diagnostic protocol that compares and combines single-image physical/signal features, neural image classifiers, physics-guided ResNet fusion, and frozen foundation encoders for real-vs-generated image forensics. Across repeated source-heldout experiments, frozen CLIP remains the transfer-ranking frontier on Ishu to source-balanced MS COCOAI with 0.8641 AUC, while all-foundation SCP-Fusion reaches 0.7995 AUC and exposes where fused scores are conservative on generated images. Strict CLIP triage covers 0.4747 of target images at 0.9261 decided-case accuracy, motivating partial-decision forensic workflows. For MS COCOAI to Ishu, source-capped tuned fusion with a native-tiled conventional branch reaches 0.7749 accuracy and 0.8472 AUC, but transform checks show important weaknesses: JPEG30 drops to 0.7076 accuracy and blur drops to 0.7872 AUC. The capped source-holdout diagnostic identifies `sd3` as the weakest held-out generator, with 0.7961 recall and 0.2039 fake-miss rate. Tiled-DINO branch replacement is useful but mode-specific: `tile_max` is the stronger operating-point diagnostic and `tile_mean` is safer for Brier/ECE. The resulting benchmark package includes source-slice diagnostics, calibration metrics, robustness transforms, and qualitative failure grids, making SCP-Fusion a reproducible framework for studying robustness and dataset bias rather than a universal best-detector claim.

**Draft assembly note:** Frame SCP-Fusion as a diagnostic protocol for source shift, not a universal detector.

**Assets to place or cite:**
- Supporting artifact: `reports/submission_text_drafts_2026_06_14.md`
- Supporting artifact: `reports/method_family_comparison_2026_06_14.md`

## Motivation and related work

AI-generated image detection is often summarized by a single accuracy number on a familiar benchmark, but that framing hides the decision problem faced by forensic reviewers. A useful detector must keep ranking, calibration, threshold behavior, and high-confidence triage stable when the generator source or image processing pipeline changes. This project therefore treats real-vs-generated image detection as a source-heldout evaluation problem. The current evidence starts with a same-domain anchor where `combined_v3` and ResNet-18 both reach 0.8246 accuracy on Ishu, while physics-guided ResNet fusion improves to 0.8450 accuracy and 0.9177 AUC. The transfer results then separate ranking strength from deployable decisions: frozen CLIP is the strongest Ishu to MS COCOAI ranker at 0.8641 AUC, while source-aware reverse tuned fusion gives a more usable operating point at 0.7632 accuracy with a 0.5175 target fake-call rate.

Prior work motivates three design choices in this benchmark. First, universal fake-detector and cross-generator benchmark studies motivate frozen foundation encoders and source-heldout testing (`universal_fake_detectors_2023`, `genimage_2023`). Second, multi-expert and spectral detectors motivate keeping physical/signal, frequency, reconstruction, and foundation branches separate enough to inspect (`aide_chameleon_2025`, `spectral_any_resolution_2025`). Third, compression-bias and high-resolution work warn against over-interpreting gains that disappear under JPEG, resize, crop, or tiling changes (`fake_or_jpeg_2024`, `no_pixel_left_behind_2025`). The physics branch is framed with the same caution: photometric and reflectance-based forensics motivate physical cues, but this repo currently uses a single-image physical/signal proxy rather than calibrated multi-light photometric stereo (`photometric_faces_2023`, `light2lie_2026`).

**Draft assembly note:** Expand the workshop motivation around deepfake forensics, generator shift, and processing robustness.

**Assets to place or cite:**
- Supporting artifact: `references.bib`
- Supporting artifact: `reports/research_deep_dive_2026_06_12.md`

## Protocol and branches

The current submission package centers on two audited evaluation directions. Ishu AI-vs-real provides repeated same-domain splits for comparing `combined_v3`, ResNet-18, physics-guided fusion, frozen encoders, and saved-score fusion. Source-balanced MS COCOAI provides generator source labels for cross-domain and source-aware validation with SD3, SD2.1, SDXL, DALL-E 3, and MidJourney-style generated subsets represented in the exported metadata. Dataset commands, audits, and artifact manifests are part of the anonymized supplementary artifact package so that results can be traced back to specific exports rather than opaque benchmark names. The paper should state that raw datasets and large model artifacts are external, while reports, prediction-derived tables, figures, manifests, and reproduction commands are bundled for review. The source-holdout stress table adds a generator-level audit: for the paper-facing capped policy, `sd3` is currently the weakest held-out generator, with 0.7961 recall and 0.2039 fake-miss rate.

The compared methods are intentionally heterogeneous. `combined_v3` measures single-image physical/signal features such as pseudo-normal consistency, noise residuals, JPEG/block cues, frequency balance, and chroma consistency. ResNet-18 is the conventional neural baseline. The physics-guided model fuses a ResNet image embedding with standardized `combined_v3` features, which is the practical physics-informed route for single-image datasets. Frozen ConvNeXt, DINOv2, and CLIP encoders test whether broad pretrained representations transfer better than task-trained detectors. SCP-Fusion combines saved branch scores and evaluates source-aware calibration, branch constraints, and threshold policies, reporting AUC, accuracy, Brier score, ECE, fake-call rate, and partial triage coverage instead of a single headline metric.

**Draft assembly note:** Keep branch descriptions inspectable: physical/signal, neural, foundation, reconstruction, and score/source fusion.

**Assets to place or cite:**
- Table fragment: `reports/assets/latex_tables/method_family_comparison.tex`
- Supporting artifact: `reports/assets/claim_evidence_matrix.md`

## Main results and operating points

Same-domain Ishu results show why the physical branch is worth keeping: physics-guided fusion improves from the tied `combined_v3`/ResNet-18 accuracy of 0.8246 to 0.8450 accuracy and 0.9177 AUC. Cross-domain transfer changes the ranking: frozen CLIP reaches 0.6363 accuracy and 0.8641 AUC on Ishu to source-balanced MS COCOAI, ahead of all-foundation SCP-Fusion at 0.6163 accuracy and 0.7995 AUC. The triage result is more operationally useful than a forced threshold: CLIP decides 0.4747 of target images at 0.9261 decided-case accuracy. In the reverse direction, source-capped tuned fusion reaches 0.7632 accuracy and 0.8361 AUC, and the bounded native-tiled conventional branch diagnostic improves that to 0.7749 accuracy and 0.8472 AUC. Robustness remains mixed: social-style 720p processing is comparatively stable at 0.7602 accuracy and 0.8506 AUC, while JPEG30 and blur expose weaker operating points at 0.7076 and 0.7105 accuracy. Held-out-generator stress adds a source-specific failure handle: `sd3` is the weakest capped source-holdout generator at 0.7961 recall and 0.2039 fake-miss rate. The reconstruction ablation is deliberately caveated: `reconstruction_v2` changes AUC by +0.0215 same-domain but -0.0408 under Ishu-to-MS transfer versus `reconstruction_lite`. A tiled-DINO follow-up gives `tile_max` average deltas of +0.0139 accuracy and +0.0147 AUC across 4 stress probes, while `tile_mean` improves Brier on 4/4 and ECE on 3/4 probes. The calibration audit keeps operating modes separate: SCP-Fusion + CLIP leads Brier at 0.3112, while combined_v4 select-k60 leads ECE at 0.2663.

**Draft assembly note:** Carry the multi-objective story: CLIP ranks best, physics helps same-domain, source-aware fusion helps reverse operating points.

**Assets to place or cite:**
- Table fragment: `reports/assets/latex_tables/transfer_frontier.tex`
- Table fragment: `reports/assets/latex_tables/reverse_operating_points.tex`
- Table fragment: `reports/assets/latex_tables/calibration_operating_modes.tex`
- Figure candidate: `reports/assets/publication_score_fusion_clip_frontier.png`
- Figure candidate: `reports/assets/publication_triage_operating_points.png`

## Failure analysis and ablations

The DFF version should make SCP-Fusion the diagnostic protocol and organizing frame rather than a single model claim. The strongest story is that different branches answer different forensic questions: CLIP gives the best standalone transfer ranking, physics-guided fusion improves the same-domain and some calibration anchors, source-capped fusion makes reverse binary decisions less biased, and strict two-threshold triage avoids pretending every image deserves a confident binary call. Failure grids and source-slice diagnostics should be used as explainability evidence: when generated images are missed, the paper should ask whether the miss is semantic, spectral, compression-driven, or a source-threshold artifact. The DFF framing can carry the tiled-DINO mode tradeoff as a small robustness design rule, then connect it to the `combined_v4` and reconstruction roadmap. The `reconstruction_v2` table is an ablation caveat rather than a new lead method because it moves AUC by +0.0215 on Ishu but -0.0408 on transfer. The paper should keep the caveat that current `combined_v4` is an ablation candidate and that true AEROBLADE/FIRE-style reconstruction has not yet replaced the lightweight residual branch.

**Draft assembly note:** Use the selected qualitative grid and source-slice diagnostics to explain what the fused detector still misses.

**Assets to place or cite:**
- Figure candidate: `reports/assets/qualitative_seed29_scp_fusion_false_negatives.png`
- Supporting artifact: `reports/combined_v4_source_slice_diagnostics_2026_06_13.md`
- Table fragment: `reports/assets/latex_tables/source_holdout_stress.tex`
- Table fragment: `reports/assets/latex_tables/reconstruction_ablation.tex`

## Limitations, ethics, and reproducibility

The manuscript should preserve all 11 ready or caveated claim guardrails from the claim-evidence matrix. Most importantly, the physical branch is a single-image proxy rather than classic photometric stereo, SCP-Fusion does not universally beat frozen CLIP, native/foundation tiling is a bounded diagnostic rather than an official external high-resolution benchmark result, and robustness claims must name the transform being tested. The operating-mode guardrail is especially important: ranking, Brier, ECE, source-heldout decisions, and tiled-DINO stress select different winners, so comparisons must name the objective. The repo is public and contains generated result tables, LaTeX fragments, paper skeletons, lint reports, literature maps, draft BibTeX, and reproduction commands. The final paper still needs official venue templates, verified bibliography metadata, and, if time allows, broader source-balanced data or a true reconstruction branch.

**Draft assembly note:** Keep the same overclaim guardrails and call out external datasets/model checkpoints.

**Assets to place or cite:**
- Supporting artifact: `reports/reproducibility_checklist_2026_06_12.md`
- Supporting artifact: `reports/submission_package_lint_2026_06_14.md`

## Author Checklist

- Replace this markdown with the official venue template before submission.
- Verify bibliography metadata and image permissions.
- Keep all claims tied to the claim-evidence matrix and linted result tables.
- Preserve the single-image physical-proxy and no-SOTA guardrails.
