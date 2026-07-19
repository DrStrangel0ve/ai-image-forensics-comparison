# FREUID Final Reminder Audit - 2026-07-16

## Organizer Update

Kaggle topic `726393`, posted by organizer Ivan Relic on July 15, asks teams to confirm three items before verification:

1. final test predictions are submitted and selected;
2. the reproducibility package is available; and
3. source code plus external models/data satisfy the challenge licensing rules.

The organizer also states that the leaderboard will freeze and top packages will be checked for reproducibility. No new model submission or post-freeze training is requested.

## Audit Result

- Final submission `54673713` is `COMPLETE` with public score `0.25470`.
- Final submission `54673732` is `COMPLETE` with public score `0.25799`.
- The Kaggle final-selection UI was previously verified as exactly `2/2`, containing only those two refs.
- The required top-level reply exists once on topic `718637`, posted by `arnavmalani` on July 14 at `07:29:32 UTC`.
- The public release `freuid-final-2026-07-13` still exposes the runtime archive, final package, and report under their recorded SHA-256 hashes.
- `scripts/verify_freuid_final_package.py` passes against the released package and score-aware submission lints.
- Repository code is MIT licensed. The frozen report records that both encoders start from public torchvision ImageNet-1K weights and that training uses only official FREUID images; no external document-image dataset is used.

## Independent Docker Verification

The local Windows host cannot start the Docker Linux engine because virtualization is unavailable, so the exact release archive was tested on a clean GitHub-hosted Linux runner instead.

- Workflow: `.github/workflows/verify-freuid-frozen-docker.yml`
- Run: <https://github.com/DrStrangel0ve/ai-image-forensics-comparison/actions/runs/29473966223>
- Result: `success`
- Frozen runtime archive SHA-256: `5cd9c9c643eef8e2f6aadb9670fa8b9f4294a89dbaab0005845c2e146337f5c9`
- Built image digest: `sha256:558766eb4a2a8e61fb35ecaf160022df4f7c04b87680403c0dfc308ee2982bea`
- `public_specialist`: completed under `docker run --network none` and wrote a valid two-row `id,label` score file.
- `ood_rank`: completed under `docker run --network none` and wrote a valid two-row `id,label` score file.

This is packaging verification only. It does not alter checkpoints, architecture, preprocessing, fusion weights, thresholds, Kaggle submissions, or final selections.

## Remaining Organizer-Side Check

The six-hour limit must still be measured by organizers on their single A100 40GB / 24-CPU sandbox. Our recorded 1,000-image RTX 3080 Ti benchmark and sequential checkpoint loading remain the evidence supplied for that review. Monitor Kaggle and the registered email address for an organizer request; do not post another reproducibility reply.
