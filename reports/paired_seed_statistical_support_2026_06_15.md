# Paired Seed Statistical Support

Run date: 2026-06-15

Status: **PASS** (4/4 checks passed).

This report compares candidate methods against baselines on matched random seeds and bootstraps the mean paired-seed delta. With only three seeds in most blocks, treat these intervals as stability support for paper wording, not as definitive hypothesis tests.

## High-Signal Takeaways

- Physics-guided ResNet vs vanilla ResNet-18 on Ishu: accuracy delta +0.0205 [+0.0088, +0.0263], 3/3 favorable seeds; AUC delta +0.0250 [+0.0160, +0.0391], 3/3 favorable seeds.
- SCP all-foundation fusion vs CLIP on Ishu -> MS COCOAI: AUC delta -0.0646 [-0.0749, -0.0475], 0/3 favorable seeds; Brier delta -0.0028 [-0.0216, +0.0094], 1/3 favorable seeds. This supports the current claim that CLIP remains the transfer-ranking anchor while fusion/calibration can change operating behavior.
- Select-k60 combined_v4 vs combined_v3 on Ishu -> MS COCOAI: AUC delta +0.0119 [-0.0233, +0.0365], 2/3 favorable seeds. This keeps v4 as a caveated transfer ablation rather than a same-domain replacement.
- Temperature-balanced all-branch fusion vs CLIP on MS COCOAI -> Ishu: accuracy delta +0.0351 [+0.0088, +0.0614], 3/3 favorable seeds. This is the strongest paper-facing support for objective-specific fusion in the reverse-transfer setting.

## Label Meanings

- `consistent_gain_ci_excludes_zero`: all or most paired deltas support the candidate and the bootstrap interval over seed deltas stays favorable.
- `mixed_seed_mean_gain`: the candidate has a favorable mean, but the seed-level story is mixed.
- `candidate_trails`: the paired mean favors the baseline.
- `diagnostic_shift`: the metric is an operating-rate shift, not a higher/lower-is-better claim.

## Checks

| check | status | detail |
| --- | --- | --- |
| all configured comparisons emitted rows | PASS | 39 metric rows emitted |
| every metric row has paired seeds | PASS | minimum paired seeds: 3 |
| directional bootstrap intervals are finite | PASS | finite bootstrap intervals over paired seed deltas |
| diagnostic operating-rate metrics are not labeled as wins | PASS | fake-call and predicted-positive rates stay descriptive |

## Paired Comparisons

| comparison | metric | direction | candidate mean | baseline mean | delta mean | delta 95% CI | wins | support_label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ishu same-domain: physics-guided ResNet vs vanilla ResNet-18 | accuracy | higher | 0.8450 | 0.8246 | +0.0205 | [+0.0088, +0.0263] | 3/3 | consistent_gain_ci_excludes_zero |
| Ishu same-domain: physics-guided ResNet vs vanilla ResNet-18 | AUC | higher | 0.9177 | 0.8927 | +0.0250 | [+0.0160, +0.0391] | 3/3 | consistent_gain_ci_excludes_zero |
| Ishu same-domain: physics-guided ResNet vs combined_v3 conventional baseline | accuracy | higher | 0.8450 | 0.8246 | +0.0205 | [-0.0175, +0.0439] | 2/3 | mixed_seed_mean_gain |
| Ishu same-domain: physics-guided ResNet vs combined_v3 conventional baseline | AUC | higher | 0.9177 | 0.8942 | +0.0235 | [-0.0129, +0.0551] | 2/3 | mixed_seed_mean_gain |
| Ishu same-domain: raw combined_v4 vs combined_v3 | accuracy | higher | 0.8129 | 0.8246 | -0.0117 | [-0.0263, +0.0000] | 0/3 | candidate_trails |
| Ishu same-domain: raw combined_v4 vs combined_v3 | AUC | higher | 0.8970 | 0.8942 | +0.0028 | [+0.0000, +0.0052] | 2/3 | mixed_seed_mean_gain |
| Ishu same-domain: raw combined_v4 vs combined_v3 | Brier | lower | 0.1307 | 0.1304 | +0.0003 | [-0.0027, +0.0019] | 1/3 | candidate_trails |
| Ishu same-domain: raw combined_v4 vs combined_v3 | ECE | lower | 0.0855 | 0.0764 | +0.0092 | [-0.0055, +0.0290] | 1/3 | candidate_trails |
| Ishu same-domain: raw combined_v4 vs combined_v3 | fake-call rate | diagnostic | 0.5088 | 0.5146 | -0.0058 | [-0.0263, +0.0175] |  | diagnostic_shift |
| Ishu -> MS COCOAI: select-k60 combined_v4 vs combined_v3 | accuracy | higher | 0.5553 | 0.5467 | +0.0087 | [+0.0000, +0.0160] | 2/3 | mixed_seed_mean_gain |
| Ishu -> MS COCOAI: select-k60 combined_v4 vs combined_v3 | AUC | higher | 0.5922 | 0.5803 | +0.0119 | [-0.0233, +0.0365] | 2/3 | mixed_seed_mean_gain |
| Ishu -> MS COCOAI: select-k60 combined_v4 vs combined_v3 | Brier | lower | 0.3261 | 0.3417 | -0.0156 | [-0.0281, +0.0019] | 2/3 | mixed_seed_mean_gain |
| Ishu -> MS COCOAI: select-k60 combined_v4 vs combined_v3 | ECE | lower | 0.2663 | 0.2911 | -0.0249 | [-0.0499, -0.0062] | 3/3 | consistent_gain_ci_excludes_zero |
| Ishu -> MS COCOAI: select-k60 combined_v4 vs combined_v3 | fake-call rate | diagnostic | 0.1813 | 0.1740 | +0.0073 | [-0.0120, +0.0260] |  | diagnostic_shift |
| Ishu -> MS COCOAI: SCP all-foundation fusion vs CLIP standalone | accuracy | higher | 0.6163 | 0.6363 | -0.0200 | [-0.0340, +0.0020] | 1/3 | candidate_trails |
| Ishu -> MS COCOAI: SCP all-foundation fusion vs CLIP standalone | AUC | higher | 0.7995 | 0.8641 | -0.0646 | [-0.0749, -0.0475] | 0/3 | candidate_trails |
| Ishu -> MS COCOAI: SCP all-foundation fusion vs CLIP standalone | Brier | lower | 0.3118 | 0.3146 | -0.0028 | [-0.0216, +0.0094] | 1/3 | candidate_trails |
| Ishu -> MS COCOAI: SCP all-foundation fusion vs CLIP standalone | ECE | lower | 0.3263 | 0.3326 | -0.0063 | [-0.0326, +0.0096] | 1/3 | candidate_trails |
| Ishu -> MS COCOAI: SCP all-foundation fusion vs CLIP standalone | predicted positive rate | diagnostic | 0.1323 | 0.1630 | -0.0307 | [-0.0480, -0.0080] |  | diagnostic_shift |
| Ishu -> MS COCOAI: source-calibrated all-foundation fusion vs CLIP standalone | accuracy | higher | 0.6267 | 0.6363 | -0.0097 | [-0.0150, -0.0010] | 0/3 | candidate_trails |
| Ishu -> MS COCOAI: source-calibrated all-foundation fusion vs CLIP standalone | AUC | higher | 0.7974 | 0.8641 | -0.0667 | [-0.0747, -0.0583] | 0/3 | candidate_trails |
| Ishu -> MS COCOAI: source-calibrated all-foundation fusion vs CLIP standalone | Brier | lower | 0.2985 | 0.3146 | -0.0161 | [-0.0219, -0.0096] | 3/3 | consistent_gain_ci_excludes_zero |
| Ishu -> MS COCOAI: source-calibrated all-foundation fusion vs CLIP standalone | ECE | lower | 0.3089 | 0.3326 | -0.0237 | [-0.0364, -0.0164] | 3/3 | consistent_gain_ci_excludes_zero |
| Ishu -> MS COCOAI: source-calibrated all-foundation fusion vs CLIP standalone | predicted positive rate | diagnostic | 0.1500 | 0.1630 | -0.0130 | [-0.0170, -0.0090] |  | diagnostic_shift |
| Ishu -> MS COCOAI: source-calibrated CLIP vs CLIP standalone | accuracy | higher | 0.6287 | 0.6363 | -0.0077 | [-0.0210, +0.0030] | 1/3 | candidate_trails |
| Ishu -> MS COCOAI: source-calibrated CLIP vs CLIP standalone | AUC | higher | 0.7839 | 0.8641 | -0.0802 | [-0.0850, -0.0725] | 0/3 | candidate_trails |
| Ishu -> MS COCOAI: source-calibrated CLIP vs CLIP standalone | Brier | lower | 0.3026 | 0.3146 | -0.0120 | [-0.0129, -0.0108] | 3/3 | consistent_gain_ci_excludes_zero |
| Ishu -> MS COCOAI: source-calibrated CLIP vs CLIP standalone | ECE | lower | 0.3114 | 0.3326 | -0.0212 | [-0.0267, -0.0158] | 3/3 | consistent_gain_ci_excludes_zero |
| Ishu -> MS COCOAI: source-calibrated CLIP vs CLIP standalone | predicted positive rate | diagnostic | 0.1520 | 0.1630 | -0.0110 | [-0.0190, -0.0050] |  | diagnostic_shift |
| MS COCOAI -> Ishu: physics-guided ResNet vs vanilla ResNet-18 | accuracy | higher | 0.6871 | 0.6404 | +0.0468 | [+0.0175, +0.1053] | 3/3 | consistent_gain_ci_excludes_zero |
| MS COCOAI -> Ishu: physics-guided ResNet vs vanilla ResNet-18 | AUC | higher | 0.7420 | 0.7005 | +0.0415 | [-0.0040, +0.0804] | 2/3 | mixed_seed_mean_gain |
| MS COCOAI -> Ishu: physics-guided ResNet vs vanilla ResNet-18 | Brier | lower | 0.2436 | 0.2877 | -0.0441 | [-0.0761, -0.0195] | 3/3 | consistent_gain_ci_excludes_zero |
| MS COCOAI -> Ishu: physics-guided ResNet vs vanilla ResNet-18 | ECE | lower | 0.1902 | 0.2671 | -0.0770 | [-0.1040, -0.0406] | 3/3 | consistent_gain_ci_excludes_zero |
| MS COCOAI -> Ishu: physics-guided ResNet vs vanilla ResNet-18 | fake-call rate | diagnostic | 0.5292 | 0.5819 | -0.0526 | [-0.1228, +0.0351] |  | diagnostic_shift |
| MS COCOAI -> Ishu: temperature-balanced score fusion vs CLIP | accuracy | higher | 0.6579 | 0.6228 | +0.0351 | [+0.0088, +0.0614] | 3/3 | consistent_gain_ci_excludes_zero |
| MS COCOAI -> Ishu: temperature-balanced score fusion vs CLIP | AUC | higher | 0.8285 | 0.8243 | +0.0043 | [-0.0362, +0.0668] | 1/3 | candidate_trails |
| MS COCOAI -> Ishu: temperature-balanced score fusion vs CLIP | Brier | lower | 0.3067 | 0.3317 | -0.0250 | [-0.0543, +0.0071] | 2/3 | mixed_seed_mean_gain |
| MS COCOAI -> Ishu: temperature-balanced score fusion vs CLIP | ECE | lower | 0.3304 | 0.3566 | -0.0262 | [-0.0427, -0.0007] | 3/3 | consistent_gain_ci_excludes_zero |
| MS COCOAI -> Ishu: temperature-balanced score fusion vs CLIP | fake-call rate | diagnostic | 0.8099 | 0.8626 | -0.0526 | [-0.0965, -0.0088] |  | diagnostic_shift |
