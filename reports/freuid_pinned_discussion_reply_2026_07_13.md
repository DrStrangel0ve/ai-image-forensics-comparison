Our final public code and frozen FREUID runtime are available here:

<https://github.com/DrStrangel0ve/ai-image-forensics-comparison>

Frozen release:

<https://github.com/DrStrangel0ve/ai-image-forensics-comparison/releases/tag/freuid-freeze-v3-2026-07-13>

Selected submissions:

- `54624136`: public-layout specialist, public score `0.25470`
- `54627101`: type-adversarial rank ensemble, public score `0.25799`

The final no-network runtime sequentially fuses a ConvNeXt-Tiny 224-pixel detector with a 384-pixel residual EfficientNet-B0 trained using capture augmentation and gradient-reversal document-type suppression. We selected the conservative 85/15 percentage-rank blend on leave-one-document-type-out validation, not on the public leaderboard. The release includes checkpoint hashes, validation summaries, Docker source, and the short technical report.
