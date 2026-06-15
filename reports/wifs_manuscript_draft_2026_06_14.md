# Source-Heldout Evaluation of Physical, Neural, and Frozen-Encoder Signals for AI-Generated Image Detection

Run date: 2026-06-14

Generated manuscript draft assembled from checked-in section drafts, abstract text, and the manuscript assembly map.
Author editing is still required before copying into the official venue template.

## Abstract

AI-generated image detection is commonly reported as closed-set accuracy, yet forensic deployment requires stable ranking, calibration, and operating points under generator shift. We evaluate this gap with a repeated-seed benchmark spanning conventional physical/signal features, ResNet-18, physics-guided feature fusion, frozen foundation encoders, and score-level SCP-Fusion. On Ishu same-domain splits, `combined_v3` and ResNet-18 tie at 0.8246 and 0.8246 accuracy, while physics-guided fusion improves to 0.8450 accuracy and 0.9177 AUC. Under Ishu to source-balanced MS COCOAI transfer, frozen CLIP is the strongest standalone ranker at 0.8641 AUC, whereas fusion variants expose calibration and fake-call-rate shifts. In the reverse direction, source-capped tuned fusion reaches 0.7632 accuracy and 0.8361 AUC with a 0.5175 target fake-call rate. Per-generator source stress flags `sd3` as the weakest held-out generator for the capped policy, with 0.7961 recall and 0.2039 fake-miss rate. A tiled-DINO probe adds a mode-specific robustness rule: `tile_max` gives +0.0139 accuracy and +0.0147 AUC average deltas, while `tile_mean` is the safer Brier/ECE diagnostic. These results support reporting AI-image forensics as a source-heldout evaluation problem with separate ranking, calibration, thresholding, and triage evidence.

**Draft assembly note:** Use the compact abstract; keep the contribution wording as source-heldout benchmark evidence, not SOTA.

**Assets to place or cite:**
- Supporting artifact: `reports/submission_text_drafts_2026_06_14.md`
- Supporting artifact: `reports/wifs_breadth_decision_2026_06_14.md`

## Introduction

AI-generated image detection is often summarized by a single accuracy number on a familiar benchmark, but that framing hides the decision problem faced by forensic reviewers. A useful detector must keep ranking, calibration, threshold behavior, and high-confidence triage stable when the generator source or image processing pipeline changes. This project therefore treats real-vs-generated image detection as a source-heldout evaluation problem. The current evidence starts with a same-domain anchor where `combined_v3` and ResNet-18 both reach 0.8246 accuracy on Ishu, while physics-guided ResNet fusion improves to 0.8450 accuracy and 0.9177 AUC. The transfer results then separate ranking strength from deployable decisions: frozen CLIP is the strongest Ishu to MS COCOAI ranker at 0.8641 AUC, while source-aware reverse tuned fusion gives a more usable operating point at 0.7632 accuracy with a 0.5175 target fake-call rate.

**Draft assembly note:** Open with ranking/calibration/threshold shift; cite physics-guided same-domain gain and CLIP transfer frontier.

**Assets to place or cite:**
- Supporting artifact: `reports/paper_section_drafts_2026_06_14.md`
- Table fragment: `reports/assets/latex_tables/same_domain_anchor.tex`
- Table fragment: `reports/assets/latex_tables/transfer_frontier.tex`

## Related work and problem framing

Prior work motivates three design choices in this benchmark. First, universal fake-detector and cross-generator benchmark studies motivate frozen foundation encoders and source-heldout testing (`universal_fake_detectors_2023`, `genimage_2023`). Second, multi-expert and spectral detectors motivate keeping physical/signal, frequency, reconstruction, and foundation branches separate enough to inspect (`aide_chameleon_2025`, `spectral_any_resolution_2025`). Third, compression-bias and high-resolution work warn against over-interpreting gains that disappear under JPEG, resize, crop, or tiling changes (`fake_or_jpeg_2024`, `no_pixel_left_behind_2025`). The physics branch is framed with the same caution: photometric and reflectance-based forensics motivate physical cues, but this repo currently uses a single-image physical/signal proxy rather than calibrated multi-light photometric stereo (`photometric_faces_2023`, `light2lie_2026`).

**Draft assembly note:** Compress to forensic benchmark families; preserve the single-image physical-proxy caveat.

**Assets to place or cite:**
- Supporting artifact: `references.bib`
- Supporting artifact: `reports/literature_map_2026_06_14.md`

## Data, audits, and metrics

The current submission package centers on two audited evaluation directions. Ishu AI-vs-real provides repeated same-domain splits for comparing `combined_v3`, ResNet-18, physics-guided fusion, frozen encoders, and saved-score fusion. Source-balanced MS COCOAI provides generator source labels for cross-domain and source-aware validation with SD3, SD2.1, SDXL, DALL-E 3, and MidJourney-style generated subsets represented in the exported metadata. Dataset commands, audits, and artifact manifests are part of the public repo so that results can be traced back to specific exports rather than opaque benchmark names. The paper should state that raw datasets and large model artifacts are external, while reports, prediction-derived tables, figures, manifests, and reproduction commands are checked in. The source-holdout stress table adds a generator-level audit: for the paper-facing capped policy, `sd3` is currently the weakest held-out generator, with 0.7961 recall and 0.2039 fake-miss rate.

**Draft assembly note:** Explain Ishu, source-balanced MS COCOAI, source labels, calibration, fake-call rate, and triage coverage.

**Assets to place or cite:**
- Table fragment: `reports/assets/latex_tables/source_holdout_stress.tex`
- Supporting artifact: `reports/source_holdout_generator_stress_2026_06_14.md`

## Methods

The compared methods are intentionally heterogeneous. `combined_v3` measures single-image physical/signal features such as pseudo-normal consistency, noise residuals, JPEG/block cues, frequency balance, and chroma consistency. ResNet-18 is the conventional neural baseline. The physics-guided model fuses a ResNet image embedding with standardized `combined_v3` features, which is the practical physics-informed route for single-image datasets. Frozen ConvNeXt, DINOv2, and CLIP encoders test whether broad pretrained representations transfer better than task-trained detectors. SCP-Fusion combines saved branch scores and evaluates source-aware calibration, branch constraints, and threshold policies, reporting AUC, accuracy, Brier score, ECE, fake-call rate, and partial triage coverage instead of a single headline metric.

**Draft assembly note:** Describe method families rather than every script; keep SCP-Fusion as a diagnostic protocol.

**Assets to place or cite:**
- Supporting artifact: `reports/method_family_comparison_2026_06_14.md`
- Table fragment: `reports/assets/latex_tables/method_family_comparison.tex`

## Results

Same-domain Ishu results show why the physical branch is worth keeping: physics-guided fusion improves from the tied `combined_v3`/ResNet-18 accuracy of 0.8246 to 0.8450 accuracy and 0.9177 AUC. Cross-domain transfer changes the ranking: frozen CLIP reaches 0.6363 accuracy and 0.8641 AUC on Ishu to source-balanced MS COCOAI, ahead of all-foundation SCP-Fusion at 0.6163 accuracy and 0.7995 AUC. The triage result is more operationally useful than a forced threshold: CLIP decides 0.4747 of target images at 0.9261 decided-case accuracy. In the reverse direction, source-capped tuned fusion reaches 0.7632 accuracy and 0.8361 AUC, and the bounded native-tiled conventional branch diagnostic improves that to 0.7749 accuracy and 0.8472 AUC. Robustness remains mixed: social-style 720p processing is comparatively stable at 0.7602 accuracy and 0.8506 AUC, while JPEG30 and blur expose weaker operating points at 0.7076 and 0.7105 accuracy. Held-out-generator stress adds a source-specific failure handle: `sd3` is the weakest capped source-holdout generator at 0.7961 recall and 0.2039 fake-miss rate. The reconstruction ablation is deliberately caveated: `reconstruction_v2` changes AUC by +0.0215 same-domain but -0.0408 under Ishu-to-MS transfer versus `reconstruction_lite`. A tiled-DINO follow-up gives `tile_max` average deltas of +0.0139 accuracy and +0.0147 AUC across 4 stress probes, while `tile_mean` improves Brier on 4/4 and ECE on 3/4 probes. The calibration audit keeps operating modes separate: SCP-Fusion + CLIP leads Brier at 0.3112, while combined_v4 select-k60 leads ECE at 0.2663.

**Draft assembly note:** Use compact tables first; include only two or three figures if the official page limit gets tight.

**Assets to place or cite:**
- Table fragment: `reports/assets/latex_tables/transfer_frontier.tex`
- Table fragment: `reports/assets/latex_tables/reverse_operating_points.tex`
- Table fragment: `reports/assets/latex_tables/calibration_operating_modes.tex`
- Table fragment: `reports/assets/latex_tables/robustness_stress.tex`
- Table fragment: `reports/assets/latex_tables/reconstruction_ablation.tex`
- Figure candidate: `reports/assets/publication_score_fusion_clip_frontier.png`
- Figure candidate: `reports/assets/publication_reverse_operating_points.png`

## Limitations and reproducibility

The manuscript should preserve all 11 ready or caveated claim guardrails from the claim-evidence matrix. Most importantly, the physical branch is a single-image proxy rather than classic photometric stereo, SCP-Fusion does not universally beat frozen CLIP, native/foundation tiling is a bounded diagnostic rather than an official external high-resolution benchmark result, and robustness claims must name the transform being tested. The operating-mode guardrail is especially important: ranking, Brier, ECE, source-heldout decisions, and tiled-DINO stress select different winners, so comparisons must name the objective. The repo is public and contains generated result tables, LaTeX fragments, paper skeletons, lint reports, literature maps, draft BibTeX, and reproduction commands. The final paper still needs official venue templates, verified bibliography metadata, and, if time allows, broader source-balanced data or a true reconstruction branch.

**Draft assembly note:** State that raw datasets/models are external while reports, commands, tables, figures, and lints are checked in.

**Assets to place or cite:**
- Supporting artifact: `reports/reproducibility_checklist_2026_06_12.md`
- Supporting artifact: `reports/submission_package_lint_2026_06_14.md`

## Author Checklist

- Replace this markdown with the official venue template before submission.
- Verify bibliography metadata and image permissions.
- Keep all claims tied to the claim-evidence matrix and linted result tables.
- Preserve the single-image physical-proxy and no-SOTA guardrails.
