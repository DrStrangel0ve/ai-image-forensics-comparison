# FREUID Reproducibility Checklist - 2026-07-13

## Candidate and Artifacts

- [x] Public specialist submitted as `54624136` (`0.25470`).
- [x] OOD rank ensemble submitted as `54627101` (`0.25799`).
- [x] Both candidates explicitly selected in Kaggle (`2/2`).
- [x] Private-test inference completed over all `134,997` released private images with the frozen runtime.
- [x] Private-complete public specialist submitted as `54673713` (`0.25470` public score; CSV SHA-256 `f2a873...68b2`).
- [x] Private-complete OOD rank ensemble submitted as `54673732` (`0.25799` public score; CSV SHA-256 `5ce966...df07`).
- [ ] Replace the two pre-private final selections with `54673713` and `54673732`, then verify Kaggle reports `2/2` selected.
- [x] Submission has exact `142,818`-row `id,label` format and passes score-aware lint.
- [x] OOD candidate changes zero hidden fallback rows relative to the public specialist.
- [x] ConvNeXt-224 and residual EfficientNet-384 checkpoints copied into the artifact staging directory.
- [x] SHA-256 and byte size recorded for every frozen model and validation summary.
- [x] Five-image CUDA inference completed from the staged artifact copies.
- [x] Sequential loading keeps only one neural checkpoint resident at a time.

## Validation

- [x] Random type/label split retained as a sanity check, not the primary model-selection claim.
- [x] Leave-MAURITIUS-out and leave-GUINEA-out checks identify easy domains.
- [x] Leave-EGYPT-out identifies the hard generalization domain.
- [x] Paired clean/JPEG/screenshot/social transforms use identical validation IDs.
- [x] Fusion weight selected by strict APCER then AuDET, not by public leaderboard probing.
- [x] Rank normalization replaces unstable cross-domain raw probability calibration.

## Runtime

- [x] `scripts/infer_freuid_checkpoint_ensemble.py` writes exact score-valued `id,label` output.
- [x] `docker/freuid/Dockerfile` uses the frozen two-checkpoint ensemble and recursive `/data` discovery.
- [x] One Docker image maps both selected submissions through `FREUID_VARIANT=public_specialist|ood_rank`.
- [x] Submitted CSV checksums and variant commands documented.
- [x] Inference requires no network and writes only to `/submissions`.
- [x] Focused runtime/model tests pass; full repository suite: `197 passed`.
- [ ] Docker build/no-network run pending external WSL2 virtualization fix (`HCS_E_HYPERV_NOT_INSTALLED`).
- [x] Record exact 1,000-image RTX 3080 Ti wall time (`211.4 s`) and A100-oriented runtime settings.
- [ ] Confirm six-hour compliance on the organizer's A100 40GB sandbox; local hardware cannot establish this directly.

## Freeze

- [x] Training, model architecture, preprocessing, and weights finalized before July 13 AoE.
- [ ] Push final code commit and publish the v3 artifact release.
- [x] Export and visually verify the updated two-page short-report PDF.
- [x] Build the final package zip and pass every non-release verifier check.
- [ ] Post exactly one official-template reply by July 15, 23:59 AoE after private inference/final packaging.
