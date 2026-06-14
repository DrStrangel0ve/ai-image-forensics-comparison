# AI Image Forensics Comparison

This project compares two ways to detect generated images:

1. A standard neural network classifier, defaulting to ResNet-18.
2. A photometric normal-consistency baseline that estimates pseudo normals from each image and checks whether lighting and surface gradients look physically coherent.
3. Conventional single-image forensic baselines that measure noise residuals, JPEG/block artifacts, ELA differences, FFT frequency balance, and chroma/noise consistency.
4. A physics-guided neural fusion model that gives ResNet-18 both pixels and standardized `combined_v3` forensic features.

The default dataset target is Kaggle's **CIFAKE: Real and AI-Generated Synthetic Images**:

```text
birdy654/cifake-real-and-ai-generated-synthetic-images
```

That dataset has real CIFAR-style images and Stable Diffusion generated images with train/test folders, which makes it a clean first benchmark. The code also supports any image-folder dataset with class names such as `REAL` and `FAKE`.

## Important Photometric Note

True photometric stereo needs multiple images of the same object from the same camera under different known lighting directions. Most real-vs-generated Kaggle datasets are single-image datasets, including CIFAKE.

So this repo implements a **single-image photometric proxy**:

- estimate local normals from luminance gradients,
- measure normal-field integrability,
- measure high-frequency shading artifacts,
- measure edge, saturation, and color-channel consistency,
- train a small logistic-regression classifier from those physical features.

That gives a fair, runnable physics-inspired baseline against the neural network. If you later add a calibrated multi-light dataset, the photometric module is the place to extend it into full photometric stereo.

The repo also includes a physics-guided neural model. It is not a classic PDE-style PINN; instead, it fuses a ResNet image embedding with a small MLP over photometric, residual, JPEG, frequency, and chroma features. This is the practical physics-informed route for the current single-image datasets.

## Research Direction

The current research plan is documented in [reports/research_deep_dive_2026_06_12.md](reports/research_deep_dive_2026_06_12.md). The proposed next model is **SCP-Fusion**: source-calibrated physical-spectral foundation fusion. It keeps the photometric/physics branch, adds CLIP/DINO-style foundation embeddings, expands `combined_v3` into multiscale spectral/noise/photometric `combined_v4`, and adds AEROBLADE-style reconstruction-error features for NTIRE/ImageCLEF-style robustness.

The active publication roadmap is documented in [reports/submission_plan_2026.md](reports/submission_plan_2026.md). The current targets are DFRWS-USA 2026 posters, IEEE WIFS 2026, and DFF-2026 at ACM Multimedia.

The current venue-facing scorecard is documented in [reports/submission_scorecard_2026_06_14.md](reports/submission_scorecard_2026_06_14.md), with a machine-readable summary in [reports/assets/submission_scorecard.csv](reports/assets/submission_scorecard.csv). The practical final-upload checklist is generated in [reports/submission_upload_checklist_2026_06_14.md](reports/submission_upload_checklist_2026_06_14.md). The current venue/challenge opportunity watchlist is generated in [reports/opportunity_watchlist_2026_06_14.md](reports/opportunity_watchlist_2026_06_14.md), with a machine-readable version in [reports/assets/opportunity_watchlist.csv](reports/assets/opportunity_watchlist.csv). NTIRE/ImageCLEF official-vs-proxy readiness is audited in [reports/external_benchmark_readiness_2026_06_14.md](reports/external_benchmark_readiness_2026_06_14.md), with machine-readable status and proxy rows in [reports/assets/external_benchmark_readiness.csv](reports/assets/external_benchmark_readiness.csv) and [reports/assets/external_benchmark_proxy_metrics.csv](reports/assets/external_benchmark_proxy_metrics.csv); [reports/external_benchmark_claim_lint_2026_06_14.md](reports/external_benchmark_claim_lint_2026_06_14.md) checks that this proxy evidence is not described as official challenge performance. The official-SOTA gap is summarized in [reports/sota_gap_report_2026_06_14.md](reports/sota_gap_report_2026_06_14.md), with machine-readable NTIRE-style proxy gaps in [reports/assets/sota_gap_report.csv](reports/assets/sota_gap_report.csv); the next experiments to close that gap are prioritized in [reports/sota_gap_closure_plan_2026_06_14.md](reports/sota_gap_closure_plan_2026_06_14.md), with machine-readable tasks in [reports/assets/sota_gap_closure_plan.csv](reports/assets/sota_gap_closure_plan.csv). The full packet manifest is documented in [reports/submission_packet_2026_06_13.md](reports/submission_packet_2026_06_13.md), with a machine-readable artifact index in [reports/assets/submission_packet_manifest.csv](reports/assets/submission_packet_manifest.csv). It validates the files to carry into DFRWS, WIFS, and DFF drafts and keeps the major claim caveats visible. A paper-facing related-work map is generated in [reports/literature_map_2026_06_14.md](reports/literature_map_2026_06_14.md), and draft BibTeX references are generated in [references.bib](references.bib) with a caveated report in [reports/references_bib_2026_06_14.md](reports/references_bib_2026_06_14.md). Editable abstract and paper-section wording is generated in [reports/submission_text_drafts_2026_06_14.md](reports/submission_text_drafts_2026_06_14.md), expanded WIFS/DFF prose section drafts are generated in [reports/paper_section_drafts_2026_06_14.md](reports/paper_section_drafts_2026_06_14.md), and [reports/paper_section_drafts_lint_2026_06_14.md](reports/paper_section_drafts_lint_2026_06_14.md) checks those prose drafts for required metrics, caveats, and overclaims. Compact copy-ready result tables are generated in [reports/submission_result_tables_2026_06_14.md](reports/submission_result_tables_2026_06_14.md), and [reports/submission_result_tables_lint_2026_06_14.md](reports/submission_result_tables_lint_2026_06_14.md) checks those tables against canonical metrics, robustness deltas, and claim-evidence IDs. The DFRWS poster assets are checked by [reports/dfrws_poster_package_lint_2026_06_14.md](reports/dfrws_poster_package_lint_2026_06_14.md), including key-number consistency, figure dimensions, and overclaim cautions. WIFS/DFF LaTeX table fragments are generated in [reports/submission_latex_tables_2026_06_14.md](reports/submission_latex_tables_2026_06_14.md), WIFS/DFF paper skeletons are generated in [reports/submission_paper_skeletons_2026_06_14.md](reports/submission_paper_skeletons_2026_06_14.md), [reports/paper_skeleton_lint_2026_06_14.md](reports/paper_skeleton_lint_2026_06_14.md) checks those skeletons for referenced assets and claim guardrails, and [reports/submission_package_lint_2026_06_14.md](reports/submission_package_lint_2026_06_14.md) checks the package for missing assets, abstract word counts, and unsafe overclaims.

The ordered publication-control runner is [scripts/run_publication_control_suite.py](scripts/run_publication_control_suite.py). Use `python scripts/run_publication_control_suite.py --dry-run` to refresh [reports/publication_control_suite_2026_06_14.md](reports/publication_control_suite_2026_06_14.md), or omit `--dry-run` to execute the checked-in publication builders and lints in dependency order.

The public reproducibility checklist is documented in [reports/reproducibility_checklist_2026_06_12.md](reports/reproducibility_checklist_2026_06_12.md). It lists what is included in the public repo, what must be downloaded externally, and the shortest commands for tests, ablations, and publication figure generation.

The completed `combined_v4` full-transfer gate is summarized in [reports/combined_v4_full_transfer_summary_2026_06_13.md](reports/combined_v4_full_transfer_summary_2026_06_13.md). Across seeds 7/17/29, raw v4 improves Ishu -> source-balanced MS COCOAI transfer accuracy but not transfer AUC or calibration, while select-k60 v4 improves transfer AUC/Brier/ECE at the cost of same-domain Ishu accuracy. `combined_v3` remains the main conventional baseline; `combined_v4_selectk60` is now a caveated transfer/calibration ablation.

## Setup

From this repo root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For an NVIDIA desktop such as an RTX 3060 Ti, install the CUDA PyTorch wheels after the normal requirements:

```powershell
python -m pip install --force-reinstall -r requirements-gpu-cu128.txt
@'
import torch
print(torch.__version__, torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
'@ | python -
```

## Kaggle Access

The Kaggle API expects credentials at:

```text
C:\Users\<you>\.kaggle\kaggle.json
```

Download CIFAKE:

```powershell
python scripts/download_kaggle.py `
  --dataset birdy654/cifake-real-and-ai-generated-synthetic-images `
  --out data/raw/cifake
```

## Dataset Catalog

List known dataset candidates:

```powershell
python scripts/list_datasets.py
```

Download one by key:

```powershell
python scripts/download_dataset.py ai_vs_real_2026
```

The catalog currently includes:

- `cifake`: CIFAKE real vs Stable Diffusion images.
- `ai_vs_real_2026`: practical-size 2026 Kaggle real-vs-AI image dataset.
- `chatgpt_gemini_deepfake_2026`: validated May 2026 Kaggle image-folder probe with ChatGPT/Gemini realistic images.
- `ishu_ai_vs_real_2026`: validated May 2026 Kaggle real-vs-AI benchmark across seven nested categories.
- `rhythm_ai_vs_real_2026`: another practical-size 2026 Kaggle real-vs-AI dataset.
- `itszubi_ai_vs_real_2026`: compact April 2026 Kaggle candidate, pending binary-label validation.
- `safemedia_ai_image_eval_2026`: recent Kaggle prompt/target sample for generated-image qualitative probes, not a ready binary benchmark.
- `stylegan3_faces_2026`: recent StyleGAN3 real-vs-fake face dataset.
- `ai_generated_vs_real_multigen`: large multi-generator Kaggle dataset.
- `ms_cocoai_2026`: Hugging Face research dataset with SD3, SD2.1, SDXL, DALL-E 3, and MidJourney v6 references.
- `arpan_deepfake_detection_v3_2026`: compact 2026 HF face-forensics dataset with reversed class ids (`fake=0`, `real=1`); export audit found severe exact duplicate leakage across its upstream train/test split.
- `syncred_bench_2026`: June 2026 HF document/credential forensics candidate with synthetic and false-positive subsets; validate label semantics before generic real/fake scoring.
- `realhd_2026`: external 2026 benchmark candidate.
- `genimage`: million-scale cross-generator benchmark candidate.
- `wildfake`: large wild-collected cross-generator benchmark candidate.
- `chameleon_aide_2025`: challenging ICLR 2025 detector stress-test candidate.
- `ntire_robust_aigen_2026`: Hugging Face training set for the NTIRE 2026 robust AI-generated image detection challenge.
- `project1_aigen_2026`: small Hugging Face 2026 candidate with labeled train/validation splits, pending manual import validation because HF Dataset Viewer first-row loading currently fails.
- `openfake_2026`: very large Hugging Face benchmark with core and Reddit in-the-wild splits.
- `parvesh_ai_vs_real`: practical-size Hugging Face real-vs-AI image candidate.
- `real_vs_ai_corpus`: very large merged Hugging Face corpus with source metadata.
- `scam_ai_gpt_image_2_2026`: recent generated-only GPT-image-2 candidate, pending manual/gated access validation.
- `aigi_now_2025`: 2025 Hugging Face generalization candidate, pending manual import validation.

## Export Hugging Face Image Datasets

Some newer datasets, such as Defactify/MS COCOAI, are hosted as Hugging Face datasets instead of image folders. Export a balanced subset into the repo's standard layout:

```powershell
python scripts/export_hf_image_dataset.py `
  --dataset-key ms_cocoai_2026 `
  --out-dir data/raw/ms_cocoai_2026_export `
  --splits train validation `
  --max-per-class-per-split 1000 `
  --shuffle-buffer 1000 `
  --streaming
```

The exporter writes folders such as `train/real`, `train/ai_generated`, `validation/real`, and `validation/ai_generated`, plus `metadata.csv`.

For datasets whose class ids use `fake=0` and `real=1`, override the label mapping explicitly:

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

## Audit Image-Folder Datasets

Before treating a new dataset as evidence, audit its layout, class counts, dimensions, exact duplicate leakage, and stricter perceptual near-duplicate candidates:

```powershell
python scripts/audit_image_dataset.py `
  --data-dir data/raw/chatgpt_gemini_deepfake_2026 `
  --out-dir runs/chatgpt_gemini_deepfake_2026_initial/audit `
  --fail-on-leakage
```

The audit writes `audit.json` and `report.md`. Near-duplicate scanning requires both average-hash and difference-hash distances to fall under the configured thresholds.

For generator-balanced Defactify/MS COCOAI validation exports, cap each generated source label independently:

```powershell
python scripts/export_hf_image_dataset.py `
  --dataset-key ms_cocoai_2026 `
  --out-dir data/raw/ms_cocoai_2026_validation_source_balanced_100 `
  --splits validation `
  --max-real-per-split 500 `
  --max-per-source-per-split 100 `
  --fake-source-label 1 --fake-source-label 2 --fake-source-label 3 `
  --fake-source-label 4 --fake-source-label 5 `
  --streaming
```

## Run One Full Benchmark

The benchmark runner can compare multiple methods on the same split:

```powershell
python scripts/run_benchmark.py `
  --dataset-key ai_vs_real_2026 `
  --out-dir runs/ai_vs_real_2026_full `
  --methods photometric noise combined combined_v2 combined_v3 neural `
  --feature-classifier logistic_regression `
  --feature-image-size 128 `
  --neural-model resnet18 `
  --pretrained `
  --epochs 5 `
  --batch-size 64 `
  --neural-image-size 128 `
  --num-workers 0 `
  --device cuda `
  --seed 7 `
  --val-fraction 0.2
```

Use `--seed` and `--val-fraction` on the wrapper when you want one deterministic split with the same feature and neural settings. For repeated split sweeps, use:

```powershell
python scripts/run_repeated_benchmark.py `
  --dataset-key ai_vs_real_2026 `
  --out-dir runs/ai_vs_real_2026_repeated `
  --seeds 7 17 29 `
  -- `
  --methods combined_v3 neural physics_guided `
  --feature-classifier logistic_regression `
  --feature-image-size 128 `
  --neural-model resnet18 `
  --pretrained `
  --epochs 5 `
  --batch-size 64 `
  --neural-image-size 128 `
  --num-workers 0 `
  --device cuda
```

Feature baselines can also train on deterministic image-sharing variants without creating a second dataset on disk:

```powershell
python scripts/run_feature_baseline.py `
  --data-dir data/raw/ms_cocoai_2026_subset_500 `
  --output-dir runs/ms_cocoai_2026_subset_500/feature_combined_v3_augmented_logistic_regression `
  --feature-set combined_v3 `
  --classifier logistic_regression `
  --image-size 128 `
  --train-augment-variants jpeg70 blur1 resize_half crop85
```

Frozen pretrained-image encoders are available as a stronger neural-feature baseline:

```powershell
python scripts/run_frozen_encoder_baseline.py `
  --data-dir data\raw\ishu_ai_vs_real_2026 `
  --output-dir runs\ishu_ai_vs_real_2026_frozen_encoder\convnext_tiny_seed7 `
  --encoder convnext_tiny `
  --pretrained `
  --classifier logistic_regression `
  --batch-size 32 `
  --num-workers 0 `
  --device cuda `
  --seed 7 `
  --val-fraction 0.2
```

Evaluate a saved frozen-encoder classifier on a new dataset:

```powershell
python scripts/evaluate_frozen_encoder_model.py `
  --model-dir runs\ishu_ai_vs_real_2026_frozen_encoder\convnext_tiny_seed7 `
  --target-dir data\raw\ms_cocoai_2026_validation_source_balanced_100 `
  --output-dir runs\ishu_to_ms_cocoai_source_balanced_seed7\convnext_tiny_frozen `
  --target-split all `
  --batch-size 32 `
  --num-workers 0 `
  --device cuda
```

Fuse saved prediction scores from multiple branches:

```powershell
python scripts/fuse_prediction_scores.py `
  --out-dir runs\score_fusion\ishu_seed7_to_ms_cocoai_all4 `
  --seed 7 `
  --train combined_v3=runs\ishu_ai_vs_real_2026_initial\feature_combined_v3_logistic_regression\predictions.csv `
  --train resnet18=runs\ishu_ai_vs_real_2026_initial\resnet18\predictions.csv `
  --train physics_guided=runs\ishu_ai_vs_real_2026_physics_guided_seed7\physics_guided_resnet18_combined_v3\predictions.csv `
  --train convnext_tiny_frozen=runs\ishu_ai_vs_real_2026_frozen_encoder\convnext_tiny_seed7\predictions.csv `
  --variant ms_cocoai:combined_v3=runs\ishu_to_ms_cocoai_source_balanced_seed7\combined_v3\predictions.csv `
  --variant ms_cocoai:resnet18=runs\ishu_to_ms_cocoai_source_balanced_seed7\resnet18\predictions.csv `
  --variant ms_cocoai:physics_guided=runs\ishu_to_ms_cocoai_source_balanced_seed7\physics_guided_resnet18_combined_v3\predictions.csv `
  --variant ms_cocoai:convnext_tiny_frozen=runs\ishu_to_ms_cocoai_source_balanced_seed7\convnext_tiny_frozen\predictions.csv
```

Neural baselines can use the same deterministic variants:

```powershell
python scripts/train_neural_net.py `
  --data-dir data/raw/ms_cocoai_2026_subset_500 `
  --output-dir runs/ms_cocoai_2026_subset_500/resnet18_augmented `
  --model resnet18 `
  --pretrained `
  --epochs 4 `
  --batch-size 64 `
  --image-size 128 `
  --device cuda `
  --train-augment-variants jpeg70 blur1 resize_half crop85
```

When running the full benchmark wrapper, pass neural variants with:

```powershell
python scripts/run_benchmark.py `
  --dataset-key ms_cocoai_2026 `
  --methods neural `
  --neural-train-augment-variants jpeg70 blur1 resize_half crop85
```

Saved physics-guided fusion models can be evaluated on a held-out split or another dataset with the same feature normalization learned during training:

```powershell
python scripts/evaluate_physics_guided_net.py `
  --model-dir runs/ishu_ai_vs_real_2026_physics_guided_seed29/physics_guided_resnet18_combined_v3 `
  --target-dir data/raw/ishu_ai_vs_real_2026 `
  --output-dir runs/ishu_ai_vs_real_2026_physics_guided_seed29/eval_repro_test `
  --image-size 128 `
  --feature-image-size 128 `
  --batch-size 64 `
  --num-workers 0 `
  --device cuda `
  --target-split test `
  --seed 29 `
  --val-fraction 0.2 `
  --skip-errors
```

## Run CIFAKE Manually

Train the neural network:

```powershell
python scripts/train_neural_net.py `
  --data-dir data/raw/cifake `
  --output-dir runs/resnet18 `
  --model resnet18 `
  --epochs 5 `
  --batch-size 128 `
  --image-size 96 `
  --device auto
```

Train the photometric normal-consistency baseline:

```powershell
python scripts/run_feature_baseline.py `
  --data-dir data/raw/cifake `
  --output-dir runs/photometric `
  --feature-set photometric `
  --image-size 128
```

Train the combined conventional baseline:

```powershell
python scripts/run_feature_baseline.py `
  --data-dir data/raw/cifake `
  --output-dir runs/combined_features `
  --feature-set combined `
  --classifier logistic_regression `
  --image-size 128
```

Compare them:

```powershell
python scripts/compare_methods.py `
  --neural runs/resnet18/metrics.json `
  --photometric runs/photometric/metrics.json `
  --out-dir runs/comparison
```

Evaluate a saved model on a different dataset:

```powershell
python scripts/evaluate_feature_model.py `
  --model-dir runs/ai_vs_real_2026_full/feature_combined_logistic_regression `
  --target-key rhythm_ai_vs_real_2026 `
  --output-dir runs/cross_dataset/ai_to_rhythm_feature_combined `
  --target-split all

python scripts/evaluate_neural_net.py `
  --model-dir runs/ai_vs_real_2026_full/resnet18 `
  --target-key rhythm_ai_vs_real_2026 `
  --output-dir runs/cross_dataset/ai_to_rhythm_resnet18 `
  --device cuda `
  --target-split all
```

Results are written to:

```text
runs/resnet18/metrics.json
runs/photometric/metrics.json
runs/comparison/report.md
runs/comparison/comparison.csv
```

## Quick Smoke Test

This creates a tiny synthetic dataset and runs both pipelines:

```powershell
python scripts/smoke_test.py
```

It is meant to validate the code path, not measure real accuracy.

## Tests

```powershell
pytest
```

## Using A Different Dataset

Any folder like this should work:

```text
dataset/
  train/
    REAL/
    FAKE/
  test/
    REAL/
    FAKE/
```

or:

```text
dataset/
  REAL/
  FAKE/
```

For a single folder without predefined splits, the scripts make a stratified train/test split.

## Expected Outputs

Each method writes:

- `metrics.json`: accuracy, precision, recall, F1, ROC AUC, confusion matrix.
- `predictions.csv`: per-image or per-sample generated-image score.
- model artifact: `model.pt` for the neural network, `photometric_model.joblib` for the photometric classifier.

The comparison script creates a markdown report and a CSV table so you can track which method wins on the same split.

## Initial CIFAKE Subset Result

An initial balanced CIFAKE subset run is checked into [reports/cifake_subset_initial.md](reports/cifake_subset_initial.md).
It used 2,000 train images and 1,000 test images per method, with a one-epoch ResNet-18 neural baseline.

## Initial 2026 Dataset Result

A full small-dataset run on `ai_vs_real_2026` is checked into [reports/ai_vs_real_2026_benchmark.md](reports/ai_vs_real_2026_benchmark.md).
It used CUDA on the local RTX 3060 Ti and compared photometric, noise, combined conventional features, and pretrained ResNet-18.

## Second 2026 Dataset Result

A second run on `rhythm_ai_vs_real_2026` is checked into [reports/rhythm_ai_vs_real_2026_benchmark.md](reports/rhythm_ai_vs_real_2026_benchmark.md).
It adds a more diverse category split across animals, city, food, nature, and people.

## Cross-Dataset Generalization Result

A zero-shot transfer run between the 2026 datasets is checked into [reports/cross_dataset_2026_generalization.md](reports/cross_dataset_2026_generalization.md).
The main result is that all methods drop sharply outside their source dataset, but physics-guided fusion now has the strongest Ishu/MS COCOAI transfer after source-domain threshold calibration.

A stricter source-heldout threshold diagnostic is checked into [reports/source_holdout_diagnostics_2026_06_12.md](reports/source_holdout_diagnostics_2026_06_12.md). It holds out each generated MS COCOAI source in turn and shows that naive source-threshold transfer often drives thresholds too low, creating high real-image false-positive rates. Physics-guided fusion still has the best three-seed mean, but SCP-Fusion needs better source-aware training rather than only better thresholding.

## Conventional V2/V3/V4 Probes

An experimental `combined_v2` conventional baseline is checked into [reports/conventional_v2_probe.md](reports/conventional_v2_probe.md).
It adds local noise-entropy and multiscale residual features inspired by newer robustness work, but keeps the original `combined` feature set for backward-compatible saved models.

`combined_v3` adds JPEG recompression sensitivity, residual 8x8 phase periodicity, RGB residual-correlation, and local residual-variance features on top of `combined_v2`.

`combined_v4` is checked into [reports/combined_v4_probe_2026_06_12.md](reports/combined_v4_probe_2026_06_12.md). It adds AEROBLADE-lite reconstruction residuals, multiring FFT statistics, chroma boundary/edge features, and stronger JPEG recompression probes. A tiny 80/40 Ishu smoke test confirms the feature set is finite and trainable, but it trails `combined_v3` on that limited split, so it is currently an ablation candidate rather than a claimed improvement.

A feature-selection follow-up is checked into [reports/combined_v4_selectk_probe_2026_06_12.md](reports/combined_v4_selectk_probe_2026_06_12.md), with summary CSVs in [reports/assets/combined_v4_selectk_probe](reports/assets/combined_v4_selectk_probe). On three bounded Ishu seeds, selected `combined_v4` improves mean AUC to 0.8219 versus 0.8033 for `combined_v3`; `k=80` has the best bounded accuracy at 0.7611, while histogram-gradient boosting underperforms on the small split. The regenerated summary assets include 95% bootstrap confidence intervals.

A medium-size grid rerun is checked into [reports/combined_v4_medium_selectk_probe_2026_06_12.md](reports/combined_v4_medium_selectk_probe_2026_06_12.md), with assets in [reports/assets/combined_v4_medium_selectk_probe](reports/assets/combined_v4_medium_selectk_probe). At 240 train / 114 validation images per seed, raw `combined_v4` has the best mean accuracy/AUC at 0.7544 / 0.8315, while select-k60 has the best Brier/ECE. The ablation summaries now report confidence intervals, making the narrow `combined_v4` lead easier to frame honestly.

A completed transfer gate is checked into [reports/combined_v4_full_transfer_summary_2026_06_13.md](reports/combined_v4_full_transfer_summary_2026_06_13.md), with seed/mean/delta CSVs in [reports/assets](reports/assets). The reproducibility command manifest remains checked in at [reports/assets/combined_v4_transfer_command_manifest.csv](reports/assets/combined_v4_transfer_command_manifest.csv). The outcome keeps `combined_v3` as the main conventional baseline and frames `combined_v4_selectk60` as a caveated transfer/calibration ablation.

A source-slice diagnostic is checked into [reports/combined_v4_source_slice_diagnostics_2026_06_13.md](reports/combined_v4_source_slice_diagnostics_2026_06_13.md). It decomposes the v4 gate by Ishu content category and MS COCOAI generator label: select-k60 gains are strongest for MidJourney v6 and DALL-E 3 transfer detection, while Ishu food, items, and animals categories explain most of the same-domain accuracy loss.

## Defactify / MS COCOAI Subset Result

A Hugging Face export and benchmark run on a 2,000-image Defactify/MS COCOAI subset is checked into [reports/ms_cocoai_2026_subset_benchmark.md](reports/ms_cocoai_2026_subset_benchmark.md).
The best standalone conventional method was `combined_v3` at 0.7290 accuracy, physics-guided ResNet-18 + `combined_v3` reached 0.7810 accuracy / 0.8812 AUC, and pretrained ResNet-18 remained strongest at 0.8160 accuracy / 0.8982 AUC. Source-label analysis showed SD3 was the hardest generator family for both methods.

## Defactify / MS COCOAI Source-Balanced Validation Result

A source-balanced validation rerun is checked into [reports/ms_cocoai_source_balanced_validation.md](reports/ms_cocoai_source_balanced_validation.md).
It keeps 500 real images and exactly 100 generated images from each Defactify source label. `combined_v3` reached 0.7320 accuracy, improving over `combined_v2` at 0.7090, while ResNet-18 reached 0.8160 accuracy on the same 1,000-image slice.

## Defactify / MS COCOAI Robustness Variant Result

A transformation robustness run is checked into [reports/ms_cocoai_robustness_variants.md](reports/ms_cocoai_robustness_variants.md).
JPEG recompression at quality 70 barely changed either detector, but Gaussian blur dropped `combined_v3` by 9.7 accuracy points while ResNet-18 dropped 1.7 points.
The MS COCOAI-trained fusion model was more robust than standalone `combined_v3` under blur and resize, dropping 1.9 and 1.2 accuracy points respectively, but still stayed below vanilla ResNet-18 on clean and transformed accuracy.

A threshold calibration follow-up is checked into [reports/ms_cocoai_threshold_calibration.md](reports/ms_cocoai_threshold_calibration.md).
Clean-threshold calibration recovers part of the blurred `combined_v3` drop, improving accuracy from 0.6350 to 0.6740, while an oracle threshold reaches 0.7030.

An augmented conventional follow-up is checked into [reports/ms_cocoai_augmented_conventional_robustness.md](reports/ms_cocoai_augmented_conventional_robustness.md).
Training `combined_v3` with JPEG, blur, resize, and crop variants kept clean accuracy at 0.7320 while improving blur accuracy from 0.6350 to 0.6980 and half-resize accuracy from 0.6940 to 0.7230.

An augmented neural follow-up is checked into [reports/ms_cocoai_augmented_neural_robustness.md](reports/ms_cocoai_augmented_neural_robustness.md).
Training ResNet-18 with the same deterministic variants improved its own blur and resize robustness deltas, but the unaugmented ResNet-18 still had better clean accuracy and AUC on this 4-epoch run.

A small ChatGPT/Gemini May 2026 Kaggle probe is checked into [reports/chatgpt_gemini_deepfake_2026_probe.md](reports/chatgpt_gemini_deepfake_2026_probe.md).
It validates the dataset layout, adds exact/perceptual duplicate auditing, and shows that in-dataset results are easy while MS COCOAI-to-ChatGPT/Gemini zero-shot transfer remains weak.

An Ishu AI-vs-real May 2026 benchmark is checked into [reports/ishu_ai_vs_real_2026_benchmark.md](reports/ishu_ai_vs_real_2026_benchmark.md).
Across seed-7, seed-17, and seed-29 deterministic splits, `combined_v3` and six-epoch ResNet-18 tied on mean accuracy to four decimals and were nearly tied on mean AUC. A follow-up physics-guided ResNet-18 fused with `combined_v3` features improved to 0.8450 mean accuracy and 0.9177 mean AUC. On three-seed Ishu to source-balanced MS COCOAI transfer, the fusion model averaged 0.6060 accuracy and 0.6637 AUC, ahead of ResNet-18 at 0.5800 / 0.6488, but seed 17 remained a ResNet win because fusion under-detected most generated MS COCOAI sources. A stronger-regularization seed-17 probe improved default transfer accuracy but lowered AUC and still trailed ResNet. In the reverse MS COCOAI-to-Ishu direction, fusion had the best AUC at 0.7089 and improved from 0.5873 to 0.6596 accuracy when thresholded from source-domain validation predictions.

A frozen ConvNeXt-Tiny encoder follow-up is checked into [reports/foundation_encoder_baseline_2026_06_12.md](reports/foundation_encoder_baseline_2026_06_12.md). It reached 0.8947 mean accuracy / 0.9589 mean AUC on Ishu and 0.6163 mean accuracy / 0.7139 mean AUC on Ishu to source-balanced MS COCOAI transfer, making it the first strong frozen-encoder ranking baseline. Its source-threshold accuracy trails the physics-guided fusion model, so the next SCP-Fusion step is to combine foundation ranking strength with physics-guided calibration behavior.
CLIP/DINO frozen-encoder support is checked into [reports/clip_dino_encoder_support_2026_06_13.md](reports/clip_dino_encoder_support_2026_06_13.md). The frozen-encoder runner now exposes `clip_vit_b_32`, `dinov2_vits14`, and `dinov2_vitb14` via Hugging Face `transformers`, making the research roadmap's foundation-embedding baselines runnable with the same logistic/MLP head and cross-dataset evaluator.
A bounded DINOv2-small smoke probe is checked into [reports/dinov2_frozen_encoder_smoke_2026_06_13.md](reports/dinov2_frozen_encoder_smoke_2026_06_13.md). With only 80 Ishu training images, DINOv2-small reaches 0.8225 AUC on the 40-image Ishu smoke validation split and 0.7059 AUC on the 1,000-image source-balanced MS COCOAI transfer target, while remaining conservative at the default threshold.
A full three-seed DINOv2-small follow-up is checked into [reports/dinov2_three_seed_foundation_baseline_2026_06_13.md](reports/dinov2_three_seed_foundation_baseline_2026_06_13.md). It reaches 0.7807 mean accuracy / 0.8615 mean AUC on Ishu and 0.6133 / 0.7063 on Ishu to source-balanced MS COCOAI transfer. ConvNeXt remains stronger, but DINOv2 has higher target fake recall, making it a good candidate fifth SCP-Fusion branch.
A five-branch SCP-Fusion + DINOv2 probe is checked into [reports/score_fusion_dinov2_probe_2026_06_13.md](reports/score_fusion_dinov2_probe_2026_06_13.md). Adding DINOv2 raises mean Ishu-to-MS-COCOAI AUC from 0.7282 to 0.7503, and the source-calibrated five-branch fusion has the best target accuracy/Brier/ECE in the current fusion comparison. In source-heldout triage, the DINOv2 fusion improves the strict 5% operating point to about 26% coverage with about 80% decided-case accuracy.
A CLIP ViT-B/32 foundation and fusion follow-up is checked into [reports/clip_vit_b32_foundation_and_fusion_2026_06_13.md](reports/clip_vit_b32_foundation_and_fusion_2026_06_13.md). CLIP is now the strongest standalone cross-domain ranker at 0.6363 accuracy / 0.8641 AUC on Ishu to source-balanced MS COCOAI. Adding CLIP lifts all-foundation SCP-Fusion to 0.7995 AUC, but standalone CLIP still wins source-heldout triage, reaching about 47% coverage at 92.6% decided-case accuracy under the strict 5% calibration-error budget.
A reverse foundation-transfer follow-up is checked into [reports/ms_cocoai_to_ishu_foundation_reverse_2026_06_13.md](reports/ms_cocoai_to_ishu_foundation_reverse_2026_06_13.md). Training frozen encoders on source-balanced MS COCOAI and testing on Ishu shows CLIP still has the best reverse AUC at 0.8243, while ConvNeXt has the best default/source-threshold accuracy at 0.6579 / 0.6784. This gives the submission story a bidirectional ranking-versus-calibration result instead of only an Ishu-to-MS-COCOAI result.
A tiled frozen-encoder evaluator is now available at [scripts/evaluate_tiled_frozen_encoder_model.py](scripts/evaluate_tiled_frozen_encoder_model.py). It scores saved CLIP/DINOv2/ConvNeXt classifiers on native-resolution global views plus deterministic tiles, then reports global, tile-mean, tile-max, and top-2 tile aggregation modes. The SOTA closure plan tracks this as the next benchmarkable foundation-branch experiment.
A reverse neural/fusion follow-up is checked into [reports/ms_cocoai_to_ishu_reverse_neural_fusion_2026_06_13.md](reports/ms_cocoai_to_ishu_reverse_neural_fusion_2026_06_13.md). All-branch score fusion now has the best reverse mean AUC at 0.8285, narrowly ahead of CLIP, but physics-guided ResNet-18 has the best default/source-threshold accuracy at 0.6871 / 0.6813 and the best Brier/ECE. That makes calibration-aware fusion the next real SCP-Fusion problem.
A reverse fusion-regularization probe is checked into [reports/ms_cocoai_to_ishu_reverse_fusion_regularization_2026_06_13.md](reports/ms_cocoai_to_ishu_reverse_fusion_regularization_2026_06_13.md). Branch dropout raises the reverse AUC frontier to 0.8406, while strong regularization (`C=0.03`) improves fusion Brier/ECE to 0.2213 / 0.2091. Physics-guided ResNet still has the best default operating point, but regularized fusion now gives a concrete path toward combining CLIP-like ranking with better calibration.
A held-out source-threshold fusion probe is checked into [reports/ms_cocoai_to_ishu_source_threshold_fusion_2026_06_13.md](reports/ms_cocoai_to_ishu_source_threshold_fusion_2026_06_13.md). It adds `--threshold-strategy source_accuracy` to the score-fusion runner and reserves 20% of aligned source rows for threshold selection. Strongly regularized all-branch fusion now reaches 0.6959 accuracy / 0.8291 AUC / 0.2188 Brier on MS-COCOAI-to-Ishu, beating the physics-guided default accuracy while still over-calling generated images.
A conservative threshold tie-break probe is checked into [reports/ms_cocoai_to_ishu_threshold_tiebreak_probe_2026_06_13.md](reports/ms_cocoai_to_ishu_threshold_tiebreak_probe_2026_06_13.md). It adds `--threshold-tiebreak higher` as a source-only forensic operating preference; the same `C=0.03` fusion now reaches 0.7193 accuracy / 0.8291 AUC while reducing the predicted fake rate from 0.7076 to 0.6608.
A source fake-rate cap probe is checked into [reports/ms_cocoai_to_ishu_threshold_cap_probe_2026_06_13.md](reports/ms_cocoai_to_ishu_threshold_cap_probe_2026_06_13.md). It adds `--threshold-max-positive-rate` for source-threshold selection; the best `0.48` cap reaches 0.7222 accuracy / 0.8291 AUC and lowers the target fake-call rate again to 0.6228.
The first saved-score SCP-Fusion v0 probe improves the Ishu-to-MS-COCOAI mean AUC again to 0.7282, but default accuracy remains lower than ConvNeXt alone because the fused scores are conservative on generated MS COCOAI images.

A branch-dropout SCP-Fusion probe is checked into [reports/score_fusion_branch_dropout_probe_2026_06_12.md](reports/score_fusion_branch_dropout_probe_2026_06_12.md). It adds optional branch-dropout training and coefficient export to `scripts/fuse_prediction_scores.py`; the first three-seed probe is a useful negative result, with tiny default-accuracy gain but worse AUC/Brier/ECE than SCP-Fusion v0.

A source-calibrated SCP-Fusion probe is checked into [reports/score_fusion_source_calibration_probe_2026_06_13.md](reports/score_fusion_source_calibration_probe_2026_06_13.md). It adds reusable temperature/Platt/isotonic calibration utilities plus an optional held-out source calibration split for score fusion. On Ishu-to-MS-COCOAI, class-balanced temperature scaling improves default accuracy to 0.6073 and Brier/ECE to 0.3123 / 0.2947, while AUC dips slightly to 0.7242.

A source-heldout calibration stress test for the score-fusion variants is checked into [reports/score_fusion_source_holdout_probe_2026_06_13.md](reports/score_fusion_source_holdout_probe_2026_06_13.md). Source-calibrated fusion has the best source-heldout Brier/ECE and fake detection, but SCP-Fusion v0 remains stronger for the 10% two-threshold triage operating point.

A utility-tuned triage follow-up is checked into [reports/score_fusion_triage_tuning_2026_06_13.md](reports/score_fusion_triage_tuning_2026_06_13.md). It searches raw versus temperature-balanced score modes and asymmetric source-heldout triage budgets before evaluating the held-out generator. SCP-Fusion v0 still has the best tuned triage utility, and raw scores are selected for every held-out fold, which sharpens the distinction between probability calibration and forensic triage utility.

A calibration diagnostics follow-up is checked into [reports/calibration_diagnostics_2026_06_12.md](reports/calibration_diagnostics_2026_06_12.md). It adds Brier score, expected calibration error, maximum calibration error, reliability-bin CSVs, and a reliability-curve figure. SCP-Fusion v0 has the best Ishu-to-MS-COCOAI Brier score and AUC, but all strong ranking models remain under-confident on the target domain.

A source-heldout post-hoc calibration follow-up is checked into [reports/source_holdout_calibration_2026_06_12.md](reports/source_holdout_calibration_2026_06_12.md). Balanced temperature scaling improves Brier/ECE for the strongest models without changing their default decisions, while flexible Platt/isotonic calibrators often overfit non-heldout generator priors and inflate real-image false positives.

A source-heldout triage-mode follow-up is checked into [reports/source_holdout_triage_2026_06_12.md](reports/source_holdout_triage_2026_06_12.md). It evaluates `likely real` / `uncertain` / `likely fake` thresholds selected on non-heldout generators. At a 5% calibration error budget, frozen ConvNeXt and SCP-Fusion decide on about 21-24% of target images with roughly 75% triage accuracy.

The source-heldout calibration and triage summary assets now include 95% deterministic bootstrap confidence intervals over held-out source/seed rows, making the tables more publication-ready without changing the saved prediction scores.

A publication-assets pass is checked into [reports/publication_assets_2026_06_12.md](reports/publication_assets_2026_06_12.md). It adds reproducible figure builders, poster/paper-ready diagnostic PNGs, captions, a pipeline diagram, and a DFRWS-style abstract draft, including the DINOv2 SCP-Fusion gain figure, CLIP frontier figure, and utility-tuned score-fusion triage figure.

A generated DFRWS poster brief is checked into [reports/dfrws_poster_brief_2026_06_13.md](reports/dfrws_poster_brief_2026_06_13.md), with compact key numbers in [reports/assets/dfrws_poster_key_numbers.csv](reports/assets/dfrws_poster_key_numbers.csv). It turns the current tables into a poster spine, figure package, claims to carry, and explicit overclaim warnings.

A first editable DFRWS poster draft is checked into [reports/dfrws_poster_draft_2026_06_13.md](reports/dfrws_poster_draft_2026_06_13.md), with the PowerPoint file at [reports/assets/dfrws_poster_draft_2026_06_13.pptx](reports/assets/dfrws_poster_draft_2026_06_13.pptx) and a PNG preview at [reports/assets/dfrws_poster_draft_2026_06_13.png](reports/assets/dfrws_poster_draft_2026_06_13.png).

A DFRWS poster-native figure pack is checked into [reports/dfrws_poster_native_figures_2026_06_13.md](reports/dfrws_poster_native_figures_2026_06_13.md). It adds large-label PNG/SVG transfer and robustness panels generated by [scripts/build_dfrws_poster_figures.py](scripts/build_dfrws_poster_figures.py) from the core result table.

An updated DFRWS poster draft v2 is checked into [reports/dfrws_poster_draft_v2_2026_06_13.md](reports/dfrws_poster_draft_v2_2026_06_13.md), with the PowerPoint file at [reports/assets/dfrws_poster_draft_v2_2026_06_13.pptx](reports/assets/dfrws_poster_draft_v2_2026_06_13.pptx) and a PNG preview at [reports/assets/dfrws_poster_draft_v2_2026_06_13.png](reports/assets/dfrws_poster_draft_v2_2026_06_13.png).

A submission-readiness snapshot is checked into [reports/submission_readiness_2026_06_13.md](reports/submission_readiness_2026_06_13.md). It verifies the DFRWS, WIFS, and DFF deadlines, maps the checked-in figure package to each target, drafts the DFRWS poster structure, and names reverse-direction foundation transfer as the next paper-critical experiment.

A generated claim-evidence matrix is checked into [reports/assets/claim_evidence_matrix.md](reports/assets/claim_evidence_matrix.md). It maps each poster/paper claim to exact result rows, marks claims as ready or caveated, and now cites the completed raw-v4/select-k60 transfer rows while keeping `combined_v4` framed as an ablation rather than a headline method.

A source-utility threshold strategy is checked into [reports/score_fusion_source_utility_threshold_2026_06_13.md](reports/score_fusion_source_utility_threshold_2026_06_13.md), and the reverse sweep is summarized in [reports/ms_cocoai_to_ishu_source_utility_threshold_2026_06_13.md](reports/ms_cocoai_to_ishu_source_utility_threshold_2026_06_13.md). Threshold-only source utility matches the previous `cap_0p48` operating point at 0.7222 accuracy / 0.8291 AUC, so the next SCP-Fusion step should move utility into fusion training or validation selection.

A source-utility model-selection follow-up is checked into [reports/ms_cocoai_to_ishu_model_utility_selection_2026_06_13.md](reports/ms_cocoai_to_ishu_model_utility_selection_2026_06_13.md). It selects among existing reverse fusion families using source-side forensic utility only. Unconstrained selection chooses high source-utility models that over-call fake images on Ishu, reaching only 0.6520 target accuracy with a 0.8216 target fake-call rate. A stricter 0.48 source fake-rate cap recovers 0.7193 accuracy, but still does not beat the fixed capped threshold family, so source-heldout generator validation or train-time utility regularization is the next real SCP-Fusion v1 step.

A source-heldout generator selection follow-up is checked into [reports/ms_cocoai_to_ishu_source_holdout_model_selection_2026_06_13.md](reports/ms_cocoai_to_ishu_source_holdout_model_selection_2026_06_13.md). It holds out each generated MS COCOAI source label while scoring source utility. Even this stricter selector chooses the same over-firing fusion heads without a fake-rate cap, reaching 0.6520 target accuracy with a 0.8216 target fake-call rate. With a 0.48 source fake-rate cap it recovers 0.7193 accuracy / 0.8291 AUC, reinforcing that SCP-Fusion v1 needs held-out-generator utility plus an explicit real-image false-positive or fake-call constraint.

A source-heldout tuned fusion follow-up is checked into [reports/ms_cocoai_to_ishu_source_holdout_tuned_fusion_2026_06_13.md](reports/ms_cocoai_to_ishu_source_holdout_tuned_fusion_2026_06_13.md). It trains score-fusion heads over regularization, branch-dropout, and source fake-rate caps, selecting the best worst-source utility under a 0.48 source fake-rate cap. This is the first reverse SCP-Fusion result to beat the fixed capped threshold family, reaching 0.7339 accuracy / 0.8341 AUC with a 0.6813 target fake-call rate.

A source fake-rate constraint sweep for tuned fusion is checked into [reports/ms_cocoai_to_ishu_tuned_fusion_constraint_sweep_2026_06_13.md](reports/ms_cocoai_to_ishu_tuned_fusion_constraint_sweep_2026_06_13.md). Tightening the source cap to 0.40 gives the best reverse SCP-Fusion operating point so far: 0.7632 accuracy / 0.8361 AUC with a 0.5175 target fake-call rate.

A JPEG70 robustness check for that tuned-fusion cap is checked into [reports/ms_cocoai_to_ishu_tuned_fusion_jpeg70_robustness_2026_06_13.md](reports/ms_cocoai_to_ishu_tuned_fusion_jpeg70_robustness_2026_06_13.md). The same source-selected `cap_0p4` policy reaches 0.7661 accuracy / 0.8485 AUC under JPEG recompression, with a 0.4678 target fake-call rate.

A broader target-transform check is now checked in for [blur1](reports/ms_cocoai_to_ishu_tuned_fusion_blur1_robustness_2026_06_13.md), [resize_half](reports/ms_cocoai_to_ishu_tuned_fusion_resize_half_robustness_2026_06_13.md), [crop85](reports/ms_cocoai_to_ishu_tuned_fusion_crop85_robustness_2026_06_13.md), [JPEG50](reports/ms_cocoai_to_ishu_tuned_fusion_jpeg50_robustness_2026_06_13.md), [JPEG30](reports/ms_cocoai_to_ishu_tuned_fusion_jpeg30_robustness_2026_06_13.md), [noise3](reports/ms_cocoai_to_ishu_tuned_fusion_noise3_robustness_2026_06_13.md), [screenshot](reports/ms_cocoai_to_ishu_tuned_fusion_screenshot_robustness_2026_06_13.md), and [social_square](reports/ms_cocoai_to_ishu_tuned_fusion_social_square_robustness_2026_06_13.md). The same source-selected `cap_0p4` policy is weakest under JPEG30, blur, and half-resolution resize at 0.7076 / 0.8167, 0.7105 / 0.7872, and 0.7164 / 0.7816 accuracy/AUC, while crop85 reaches 0.7251 / 0.8227, JPEG50 reaches 0.7515 / 0.8309, noise3 reaches 0.7690 / 0.8704, screenshot-style down/upscale plus JPEG reaches 0.7310 / 0.7965, and social-square crop/recompression reaches 0.7778 / 0.8474. This turns robustness from a single positive JPEG check into a clearer stress-test story.

A generated robustness failure ranking is checked into [reports/robustness_failure_ranking_2026_06_14.md](reports/robustness_failure_ranking_2026_06_14.md), with the machine-readable table at [reports/assets/robustness_failure_ranking.csv](reports/assets/robustness_failure_ranking.csv). It ranks half-resolution resize as the largest AUC stressor and JPEG30 as the largest default-accuracy stressor for the reverse tuned-fusion SCP-Fusion setting.

An opposite-direction threshold-objective diagnostic is checked into [reports/ishu_to_ms_threshold_objective_sweep_2026_06_13.md](reports/ishu_to_ms_threshold_objective_sweep_2026_06_13.md). On saved Ishu-to-MS all-foundation SCP-Fusion scores, a less real-FPR-punitive source utility improves accuracy from the fixed-threshold 0.6163 to 0.6470, but the target fake-call rate remains low at 0.1863; treat this as a sensitivity result, not a deployable selector.

A public reproducibility pass is checked into [reports/reproducibility_checklist_2026_06_12.md](reports/reproducibility_checklist_2026_06_12.md). It adds a reviewer-facing map of included reports/assets, intentionally excluded raw data/model artifacts, core reproduction commands, and known limitations.

A qualitative failure-case export is checked into [reports/qualitative_failure_cases_2026_06_12.md](reports/qualitative_failure_cases_2026_06_12.md). It adds reproducible false-positive, false-negative, and model-disagreement grids for the seed-17 Ishu-to-MS-COCOAI transfer run, with per-method score manifests.

An Ishu three-seed robustness follow-up is checked into [reports/ishu_physics_guided_robustness_3seed.md](reports/ishu_physics_guided_robustness_3seed.md).
Across 12 seed-plus-transform checks, physics-guided fusion won 10 by accuracy and 10 by AUC. Its mean transformed score was 0.8443 accuracy and 0.9189 AUC, ahead of ResNet-18 at 0.8231 / 0.8904 and `combined_v3` at 0.7924 / 0.8751.

A focused physics-guided comparison is checked into [reports/physics_guided_vs_resnet_2026_06_12.md](reports/physics_guided_vs_resnet_2026_06_12.md). It frames the result conservatively: physics-guided ResNet-18 is stronger on Ishu, robustness, and several transfer diagnostics, while vanilla ResNet-18 still wins MS COCOAI in-domain validation.

A dataset triage follow-up is checked into [reports/dataset_triage_2026_06_12.md](reports/dataset_triage_2026_06_12.md).
It adds ARPAN V3 and SynCred-Bench to the catalog, fixes Hugging Face label override handling, and rejects ARPAN V3's upstream split for fair scoring because exact duplicate groups cross train/test.
