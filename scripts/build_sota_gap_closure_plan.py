from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


RUN_DATE = "2026-06-14"

TASKS = [
    {
        "rank": 1,
        "task_id": "genimage_official_split_eval",
        "target": "GenImage official or paper-compatible splits",
        "status": "ready_when_data_available",
        "why": "This is the cleanest academic SOTA bridge: a public benchmark family with generator and degradation splits.",
        "primary_gap": "No official GenImage run; current evidence is Ishu/MS COCOAI proxy transfer only.",
        "commands": "Train frozen CLIP/SCP-Fusion branches on the chosen source split, then run `python scripts/evaluate_frozen_encoder_model.py --model-dir <clip_run> --target-dir <genimage_target> --output-dir <out>` and `python scripts/evaluate_feature_model.py --model-dir <combined_v3_run> --target-dir <genimage_target> --output-dir <out>`.",
        "exit_criterion": "Report per-generator and degraded-split accuracy/AUC for CLIP, DINOv2, combined_v3, physics-guided, and SCP-Fusion.",
        "paper_use": "Turns SOTA comparison from methodological to benchmark-backed.",
    },
    {
        "rank": 2,
        "task_id": "ntire_released_protocol_replay",
        "target": "NTIRE 2026 released train/validation protocol or NTIRE 2027",
        "status": "blocked_on_official_labels_or_next_cycle",
        "why": "NTIRE is the strongest robustness SOTA anchor; current best proxy AUC is below the official robust top score.",
        "primary_gap": "No official NTIRE score; current NTIRE-style proxy rows are marked `proxy_not_official`.",
        "commands": "Once labels/splits are available, run source-aware SCP-Fusion and robustness transforms with the same ROC-AUC reporting; keep official and proxy outputs in separate report IDs.",
        "exit_criterion": "Produce an official or released-split ROC-AUC table with clean, transformed, and average robust scores.",
        "paper_use": "Needed for any future NTIRE/SOTA leaderboard claim.",
    },
    {
        "rank": 3,
        "task_id": "high_res_tiled_foundation_eval",
        "target": "Ishu/MS COCOAI plus future GenImage/RealHD high-resolution slices",
        "status": "core_transform_stress_probes_complete",
        "why": "Native tiling now covers conventional and foundation branches, but official SOTA systems often exploit broader high-resolution local evidence and stronger transform coverage.",
        "primary_gap": "CLIP, DINOv2, and ConvNeXt reverse-transfer tiling are benchmarked and folded into a fixed reverse SCP-Fusion diagnostic; DINOv2 tiled fusion now has blur1, JPEG30, resize-half, and screenshot-style stress probes, but official high-resolution benchmark runs are still pending.",
        "commands": "Run `python scripts/build_tiled_foundation_comparison.py` and `python scripts/build_tiled_foundation_fusion_comparison.py`; for transformed targets, run `python scripts/evaluate_reverse_tiled_fusion_robustness.py --variant <variant> --tile-branch dinov2_vits14 --tile-detail <detail.csv>` after generating tiled target scores.",
        "exit_criterion": "Show whether the fused tiled-foundation gain over 0.8472 AUC transfers to official or paper-compatible high-resolution benchmark slices without target tuning.",
        "paper_use": "Strengthens SCP-Fusion as a forensic, high-resolution protocol instead of a resized-image classifier.",
    },
    {
        "rank": 4,
        "task_id": "reconstruction_branch_ablation",
        "target": "Ishu/MS COCOAI, then GenImage degraded splits",
        "status": "bounded_transfer_and_fusion_probe_complete_pretrained_branch_next",
        "why": "AEROBLADE/FIRE-style reconstruction signals are a major SOTA direction and are not yet a full branch here.",
        "primary_gap": "reconstruction_lite now has bounded same-domain, Ishu -> MS transfer, and two-branch fusion probes, but no pretrained autoencoder or diffusion-reconstruction detector branch is implemented yet.",
        "commands": "Use `python scripts/summarize_reconstruction_lite_transfer_probe.py` and `python scripts/summarize_reconstruction_lite_fusion_probe.py` for bounded diagnostics, then add cached autoencoder/diffusion-reconstruction residual features.",
        "exit_criterion": "Report whether reconstruction features beat combined_v3 or improve SCP-Fusion transfer/robustness under JPEG, blur, and resize.",
        "paper_use": "Makes the physics/conventional branch more competitive with current reconstruction-based forensics.",
    },
    {
        "rank": 5,
        "task_id": "source_diverse_training_scaleup",
        "target": "MS COCOAI plus recent generated-image datasets",
        "status": "data_next",
        "why": "SOTA robustness depends heavily on source diversity; current transfer results show strong threshold and generator-shift sensitivity.",
        "primary_gap": "Training sources are much narrower than NTIRE-style 42-generator protocols.",
        "commands": "Expand the dataset catalog, train repeated seeds with `python scripts/run_repeated_benchmark.py --data-dir <source> --out-dir <runs> --seeds 7 17 29 -- ...`, and evaluate leave-one-generator-out.",
        "exit_criterion": "Show improved heldout-source AUC/fake-call balance without target tuning.",
        "paper_use": "Separates model weakness from dataset/source-coverage weakness.",
    },
    {
        "rank": 6,
        "task_id": "imageclef_next_cycle_packaging",
        "target": "ImageCLEF 2027-style no-training/public-submission protocol",
        "status": "watchlist",
        "why": "ImageCLEF rewards robust packaging and clear operating points; our triage evidence is a natural fit.",
        "primary_gap": "No ImageCLEF 2026 task data or submission.",
        "commands": "Package the inference path as a single runner that emits per-image scores, labels, and confidence/triage decisions without training-time leakage.",
        "exit_criterion": "A dry-run submission folder can be produced from any image folder with a stable manifest and score CSV.",
        "paper_use": "Makes the repo competition-ready for the next open cycle.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a prioritized plan for closing the SOTA/benchmark gap."
    )
    parser.add_argument(
        "--sota-gap",
        default="reports/assets/sota_gap_report.csv",
        help="Machine-readable SOTA gap rows from build_sota_gap_report.py.",
    )
    parser.add_argument(
        "--out-path",
        default="reports/sota_gap_closure_plan_2026_06_14.md",
        help="Markdown closure plan to write.",
    )
    parser.add_argument(
        "--csv-out",
        default="reports/assets/sota_gap_closure_plan.csv",
        help="Machine-readable closure plan to write.",
    )
    return parser.parse_args()


def _markdown_escape(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).replace("\n", " ").replace("|", "\\|")


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame[columns].itertuples(index=False):
        lines.append("| " + " | ".join(_markdown_escape(value) for value in row) + " |")
    return "\n".join(lines)


def _gap_summary(gaps: pd.DataFrame) -> dict[str, object]:
    if gaps.empty:
        raise ValueError("SOTA gap CSV has no rows")
    required = {"finding_id", "proxy_auc", "official_sota_auc", "auc_gap_to_official_sota"}
    missing = required - set(gaps.columns)
    if missing:
        raise ValueError(f"SOTA gap CSV missing columns: {', '.join(sorted(missing))}")
    best = gaps.sort_values("proxy_auc", ascending=False).iloc[0]
    worst = gaps.sort_values("proxy_auc", ascending=True).iloc[0]
    return {
        "best_finding_id": best["finding_id"],
        "best_proxy_auc": float(best["proxy_auc"]),
        "official_sota_auc": float(best["official_sota_auc"]),
        "best_gap": float(best["auc_gap_to_official_sota"]),
        "worst_finding_id": worst["finding_id"],
        "worst_gap": float(worst["auc_gap_to_official_sota"]),
        "n_proxy_rows": int(len(gaps)),
    }


def build_closure_plan(sota_gap_path: Path) -> tuple[str, pd.DataFrame]:
    gaps = pd.read_csv(sota_gap_path)
    summary = _gap_summary(gaps)
    frame = pd.DataFrame(TASKS)

    status_summary = frame.groupby("status", sort=False).size().reset_index(name="count")
    lines = [
        "# SOTA Gap Closure Plan",
        "",
        f"Run date: {RUN_DATE}",
        "",
        "Generated by `scripts/build_sota_gap_closure_plan.py` from the guarded SOTA-gap report.",
        "",
        "This is an experiment plan, not a claim of leaderboard performance. It ranks the next work by how directly it can turn the current proxy evidence into official or paper-compatible benchmark evidence.",
        "",
        "## Current Gap Anchor",
        "",
        f"- Best NTIRE-style local proxy row: `{summary['best_finding_id']}` with AUC `{summary['best_proxy_auc']:.4f}`.",
        f"- Official NTIRE 2026 robust SOTA anchor: AUC `{summary['official_sota_auc']:.4f}`.",
        f"- Best proxy gap to that anchor: `{summary['best_gap']:.4f}` AUC.",
        f"- Stress rows tracked: `{summary['n_proxy_rows']}`; weakest tracked gap is `{summary['worst_gap']:.4f}` AUC from `{summary['worst_finding_id']}`.",
        "",
        "## Status Summary",
        "",
        _markdown_table(status_summary, ["status", "count"]),
        "",
        "## Prioritized Tasks",
        "",
        _markdown_table(
            frame,
            [
                "rank",
                "task_id",
                "target",
                "status",
                "why",
                "primary_gap",
                "exit_criterion",
                "paper_use",
            ],
        ),
        "",
        "## Command Notes",
        "",
        _markdown_table(frame, ["rank", "task_id", "commands"]),
        "",
        "## Immediate Recommendation",
        "",
        "1. Do `genimage_official_split_eval` first if the data can be downloaded cleanly; it is the fastest way to make the paper's SOTA section concrete.",
        "2. Stress-test the small fused tiled-foundation gain before promoting it beyond a diagnostic result.",
        "3. Keep NTIRE/ImageCLEF wording conservative until an official or released-split run exists.",
        "",
    ]
    return "\n".join(lines), frame


def main() -> None:
    args = parse_args()
    text, frame = build_closure_plan(Path(args.sota_gap))

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")

    csv_path = Path(args.csv_out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)

    print(out_path.resolve())
    print(csv_path.resolve())


if __name__ == "__main__":
    main()
