# DFRWS Qualitative Grid Selection

Run date: 2026-06-14

Decision: use the seed-29 SCP-Fusion false-negative grid as the DFRWS poster qualitative panel; keep the other grid as appendix or backup material.

This selection intentionally weights poster readability and forensic story clarity ahead of raw miss confidence. The seed-17 examples have slightly lower SCP-Fusion scores, but the seed-29 panel communicates the failure mode faster.

## Candidate Audit

| seed | recommendation | rows | sources | source_counts | mean_scp_fusion_score | mean_score_spread | poster_readability_score | artifact_story_score | poster_selection_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 29 | selected_for_dfrws_poster | 12 | 4 | dalle3=3, midjourney6=2, sd21=5, sd3=2 | 0.0232 | 0.0228 | 5.0000 | 5.0000 | 5.0000 |
| 17 | appendix_backup | 12 | 4 | dalle3=3, midjourney6=4, sd21=3, sd3=2 | 0.0103 | 0.0113 | 3.5000 | 3.5000 | 3.8000 |

## Poster Use

- Selected panel: `reports/assets/qualitative_seed29_scp_fusion_false_negatives.png`.
- Caption angle: source-heldout fusion can miss generated images even when visible text, signage, object labels, or compositing cues look suspicious to a human reviewer.
- Keep seed 17 as a reproducibility/appendix panel because it shows the same failure class under a different split seed.
- Do not imply these are representative of all misses; they are illustrative examples selected from checked-in false-negative manifests.

## Candidate Notes

- Seed 29: Best poster panel: readable text/sign failures, object-label oddities, compositing artifacts, and repeated low fake scores across source families make the failure mode obvious quickly.
- Seed 17: Good confidence-miss evidence, but several examples are visually subtle at poster scale and the panel reads more like broad error sampling than a single forensic story.
