# FREUID Pinned Discussion Reply Draft

Use this as the single pinned-thread reply after verifying the exact organizer instructions for the thread.

```text
Team / participant: Arnav Malani

Public repository:
https://github.com/DrStrangel0ve/ai-image-forensics-comparison

Frozen runtime artifacts:
https://github.com/DrStrangel0ve/ai-image-forensics-comparison/releases/tag/freuid-freeze-2026-07-10

Best public Kaggle submission:
54511333

Public score:
0.37009

Method summary:
The submitted system is a frozen two-branch score fusion:
score = 0.7 * combined_v4_hgb + 0.3 * convnext_tiny_logreg.

The combined_v4 branch uses photometric, residual/noise, JPEG/blocking, frequency, chroma, and reconstruction-lite image forensics features with a histogram-gradient-boosting classifier. The neural branch uses a frozen ImageNet-pretrained ConvNeXt-Tiny encoder with a logistic classifier over embeddings. The backbone is not fine-tuned.

Runtime:
The container expects test images mounted flat at /data and writes /submissions/submission.csv with id,label fraud scores, where id is the filename stem and label is a score in [0, 1].

Code-freeze note:
The repository contains the training/inference code and Docker scaffold. The frozen model/checkpoint artifacts are included in the GitHub release above and match the runtime recipe used for submission 54511333.
```
