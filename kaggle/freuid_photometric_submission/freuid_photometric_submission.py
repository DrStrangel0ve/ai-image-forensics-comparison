from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


COMPETITION_SLUG = "the-freuid-challenge-2026-ijcai-ecai"
REPO_URL = "https://github.com/DrStrangel0ve/ai-image-forensics-comparison.git"
WORK_DIR = Path("/kaggle/working")
INPUT_DIR = Path("/kaggle/input") / COMPETITION_SLUG
REPO_DIR = WORK_DIR / "ai-image-forensics-comparison"


def run(command: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=str(cwd) if cwd is not None else None, check=True)


def first_existing(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"None of these paths exist: {[str(candidate) for candidate in candidates]}")


def main() -> None:
    train_labels = first_existing(
        [
            INPUT_DIR / "train_labels.csv",
            INPUT_DIR / "small_files" / "train_labels.csv",
        ]
    )
    sample_submission = first_existing(
        [
            INPUT_DIR / "sample_submission.csv",
            INPUT_DIR / "small_files" / "sample_submission.csv",
        ]
    )

    if not REPO_DIR.exists():
        run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)])
    run([sys.executable, "-m", "pip", "install", "-q", "joblib", "pandas", "pillow", "scikit-learn", "tqdm"])

    train_csv = WORK_DIR / "freuid_train.csv"
    val_csv = WORK_DIR / "freuid_val.csv"
    split_manifest = WORK_DIR / "freuid_split_manifest.json"
    output_dir = WORK_DIR / "freuid_photometric_public"
    branch_predictions = output_dir / "test_predictions.csv"
    submission = WORK_DIR / "submission.csv"
    package_manifest = WORK_DIR / "submission_manifest.json"
    lint_manifest = WORK_DIR / "submission_lint.json"

    run(
        [
            sys.executable,
            "scripts/prepare_freuid_split.py",
            "--train-labels",
            str(train_labels),
            "--train-out",
            str(train_csv),
            "--val-out",
            str(val_csv),
            "--manifest-out",
            str(split_manifest),
            "--val-fraction",
            "0.2",
            "--seed",
            "7",
        ],
        cwd=REPO_DIR,
    )

    run(
        [
            sys.executable,
            "scripts/run_freuid_feature_baseline.py",
            "--train-csv",
            str(train_csv),
            "--val-csv",
            str(val_csv),
            "--test-csv",
            str(sample_submission),
            "--image-root",
            str(INPUT_DIR),
            "--output-dir",
            str(output_dir),
            "--feature-set",
            "photometric",
            "--classifier",
            "logistic_regression",
            "--image-size",
            "128",
            "--max-train-samples",
            "640",
            "--max-val-samples",
            "160",
            "--limit-balance-columns",
            "type",
            "label",
            "--test-predictions-out",
            str(branch_predictions),
        ],
        cwd=REPO_DIR,
    )

    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    threshold = float(metrics["threshold_for_1pct_bpcer"])
    print(
        "local_validation="
        f"accuracy={metrics['accuracy']:.6f} "
        f"auc={metrics['roc_auc']:.6f} "
        f"apcer={metrics['apcer_at_1pct_bpcer']:.6f} "
        f"audet_proxy={metrics['audet_proxy']:.6f} "
        f"threshold={threshold:.8f}",
        flush=True,
    )

    run(
        [
            sys.executable,
            "scripts/package_freuid_submission.py",
            "--sample-submission",
            str(sample_submission),
            "--predictions",
            str(branch_predictions),
            "--score-column",
            "fraud_score",
            "--threshold",
            f"{threshold:.12f}",
            "--out-path",
            str(submission),
            "--manifest-out",
            str(package_manifest),
        ],
        cwd=REPO_DIR,
    )
    run(
        [
            sys.executable,
            "scripts/lint_freuid_submission.py",
            "--sample-submission",
            str(sample_submission),
            "--submission",
            str(submission),
            "--manifest-out",
            str(lint_manifest),
        ],
        cwd=REPO_DIR,
    )

    print("submission_ready", submission, flush=True)
    print((WORK_DIR / "submission_lint.json").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
