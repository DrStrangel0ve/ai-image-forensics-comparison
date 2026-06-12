# Dataset Triage: 2026-06-12 Follow-Up

This pass searched for newer or more diverse real-vs-generated image sources and validated the most practical new candidate before running any benchmark scores.

## Search Notes

Kaggle CLI searches for recent small-to-medium datasets under queries like `AI generated image detection` and `real vs ai images` did not surface a newer ready-to-use real-vs-AI image-folder benchmark than `ishu15m/ai-vs-real-images`, which is already validated in this repo. One tiny Kaggle candidate, `ndijeinr/ai-generated-image-detection-val`, appeared as a validation-only package and needs manual inspection before cataloging.

Hugging Face checks found:

| candidate | status | notes |
| --- | --- | --- |
| [ARPAN2026/deepfake-detection-dataset-v3](https://huggingface.co/datasets/ARPAN2026/deepfake-detection-dataset-v3) | cataloged, exported, rejected for fair split scoring | Accessible parquet-backed face-forensics dataset with `fake=0`, `real=1`; audit found severe exact duplicate leakage. |
| [thu-coai/Syncred-Bench](https://huggingface.co/datasets/thu-coai/Syncred-Bench) | cataloged as manual candidate | Updated 2026-06-07, 2,100 rows across `fp_450` and `syncred_600` configs according to Dataset Viewer size metadata; promising for document/credential forensics, but label semantics need confirmation before generic real/fake scoring. |
| [ThreeLiu/Treasure](https://huggingface.co/datasets/ThreeLiu/Treasure) | gated/manual | Modern generated-image source with image-forensics tags, but Dataset Viewer returned 401 without access approval. |
| [Scam-AI/gpt-image-2](https://huggingface.co/datasets/Scam-AI/gpt-image-2) | already cataloged as gated/manual | GPT-image-2 public-output candidate, but Dataset Viewer still requires gated access. |

## ARPAN V3 Export Validation

The exporter needed a label-mapping fix before this dataset could be handled safely. Its Hugging Face class-label order is `fake=0`, `real=1`, while many existing exports use the opposite convention. `scripts/export_hf_image_dataset.py` now treats user-supplied `--real-label` and `--fake-label` values as overrides instead of appending them to defaults.

Export command:

```powershell
python scripts/export_hf_image_dataset.py `
  --dataset-key arpan_deepfake_detection_v3_2026 `
  --config default `
  --splits train test `
  --out-dir data/raw/arpan_deepfake_detection_v3_2026 `
  --image-column image `
  --label-column label `
  --real-label 1 `
  --fake-label 0
```

Export result:

| split | ai_generated | real |
| --- | ---: | ---: |
| train | 378 | 96 |
| test | 41 | 12 |

Audit command:

```powershell
python scripts/audit_image_dataset.py `
  --data-dir data/raw/arpan_deepfake_detection_v3_2026 `
  --out-dir runs/arpan_deepfake_detection_v3_2026_initial/audit
```

Audit summary:

| field | value |
| --- | ---: |
| images | 527 |
| unique SHA-256 hashes | 9 |
| exact duplicate groups | 9 |
| cross-split duplicate groups | 9 |
| cross-class duplicate groups | 0 |
| perceptual near-duplicate pairs | 0 |
| width/height | 256x256 |

Decision: do not report neural-vs-conventional accuracy on ARPAN V3's upstream split. Every exported image belongs to one of only nine exact-hash groups, and all duplicate groups cross train/test. Any benchmark on that split would mostly measure duplicate memorization.

## Project Changes From This Triage

- Added catalog entries for `arpan_deepfake_detection_v3_2026` and `syncred_bench_2026`.
- Fixed Hugging Face exporter label overrides so reversed class ids can be handled without ambiguous defaults.
- Added `--fail-on-leakage` and targeted leakage failure flags to `scripts/audit_image_dataset.py`, allowing batch dataset triage to stop after writing the audit report when leakage is found.

## Next Steps

The next worthwhile dataset pass is SynCred-Bench label-semantics validation or a gated-access check for Treasure/GPT-image-2. ARPAN V3 should stay in the catalog as a cautionary candidate, not as a fair benchmark source.
