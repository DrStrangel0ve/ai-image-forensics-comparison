from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


RUN_DATE = "2026-06-14"

OPTIONS = [
    {
        "option_id": "freeze_current_scope",
        "decision": "selected",
        "paper_use": "WIFS main plan",
        "manuscript_value": 4.0,
        "deadline_fit": 5.0,
        "reproducibility_fit": 5.0,
        "effort": 1.0,
        "risk": 1.0,
        "rationale": (
            "Use the already checked source-heldout, robustness, reconstruction, and method-family evidence. "
            "This keeps the WIFS paper compact and avoids a last-minute experiment that could dilute the story."
        ),
    },
    {
        "option_id": "source_aware_v4_selection",
        "decision": "optional_spare_time",
        "paper_use": "appendix ablation only",
        "manuscript_value": 4.0,
        "deadline_fit": 4.0,
        "reproducibility_fit": 3.0,
        "effort": 2.0,
        "risk": 3.0,
        "rationale": (
            "Best optional add-on if a short GPU window opens, because the current combined_v4 caveat already "
            "points to source-aware feature selection as the most targeted follow-up."
        ),
    },
    {
        "option_id": "larger_source_split",
        "decision": "defer",
        "paper_use": "post-WIFS/DFF extension",
        "manuscript_value": 5.0,
        "deadline_fit": 2.0,
        "reproducibility_fit": 2.0,
        "effort": 5.0,
        "risk": 4.0,
        "rationale": (
            "Scientifically valuable, but it changes dataset scope and can disturb all existing tables close to deadline."
        ),
    },
    {
        "option_id": "true_tiled_neural_foundation",
        "decision": "defer",
        "paper_use": "post-WIFS/DFF extension",
        "manuscript_value": 5.0,
        "deadline_fit": 2.0,
        "reproducibility_fit": 2.0,
        "effort": 5.0,
        "risk": 4.0,
        "rationale": (
            "High-value method work, but it is implementation-heavy and should not block the current WIFS paper."
        ),
    },
]

EVIDENCE_ANCHORS = [
    {
        "anchor": "method_family_comparison",
        "path": "reports/method_family_comparison_2026_06_14.md",
        "use": "Condense the physical, neural, foundation, reconstruction, and fusion families into the method/results spine.",
    },
    {
        "anchor": "source_holdout_stress",
        "path": "reports/source_holdout_generator_stress_2026_06_14.md",
        "use": "Use held-out generator stress as the breadth evidence for source-shift risk.",
    },
    {
        "anchor": "robustness_stress",
        "path": "reports/assets/latex_tables/robustness_stress.tex",
        "use": "Use transform stress as the processing-robustness table.",
    },
    {
        "anchor": "reconstruction_ablation",
        "path": "reports/assets/latex_tables/reconstruction_ablation.tex",
        "use": "Use reconstruction_v2 as a caveated ablation, not a new lead method.",
    },
    {
        "anchor": "combined_v4_source_slices",
        "path": "reports/combined_v4_source_slice_diagnostics_2026_06_13.md",
        "use": "Use v4 source slices as dataset-bias explanation for why v4 stays in the appendix.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve the WIFS paper-critical breadth choice from checked-in evidence."
    )
    parser.add_argument("--repo-root", default=".", help="Repository root for checking anchor paths.")
    parser.add_argument(
        "--scorecard",
        default="reports/assets/submission_scorecard.csv",
        help="Venue scorecard CSV used to quote current WIFS readiness.",
    )
    parser.add_argument(
        "--out-path",
        default="reports/wifs_breadth_decision_2026_06_14.md",
        help="Markdown decision report to write.",
    )
    parser.add_argument(
        "--csv-out",
        default="reports/assets/wifs_breadth_decision.csv",
        help="Machine-readable decision table to write.",
    )
    return parser.parse_args()


def _format_float(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.1f}"


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


def _read_wifs_scorecard(scorecard_path: Path) -> dict[str, object]:
    frame = pd.read_csv(scorecard_path)
    row = frame[frame["venue_key"] == "WIFS"].iloc[0]
    return {
        "packet_status": row["packet_status"],
        "artifacts": f"{int(row['required_artifacts_present'])}/{int(row['required_artifacts_total'])}",
        "lint_reports": f"{int(row['lint_reports_passed'])}/{int(row['lint_reports_total'])}",
        "lint_checks": f"{int(row['lint_checks_passed'])}/{int(row['lint_checks_total'])}",
        "claims": (
            f"ready {int(row['ready_claims'])}, caveated {int(row['caveated_claims'])}, "
            f"needs {int(row['needs_more_evidence_claims'])}"
        ),
    }


def build_decision_report(repo_root: Path, scorecard_path: Path) -> tuple[str, pd.DataFrame]:
    options = pd.DataFrame(OPTIONS)
    options["decision_score"] = (
        options["manuscript_value"]
        + options["deadline_fit"]
        + options["reproducibility_fit"]
        - options["effort"]
        - options["risk"]
    )
    options = options.sort_values(["decision_score", "option_id"], ascending=[False, True])
    display = options.copy()
    for column in [
        "manuscript_value",
        "deadline_fit",
        "reproducibility_fit",
        "effort",
        "risk",
        "decision_score",
    ]:
        display[column] = display[column].map(_format_float)

    anchors = pd.DataFrame(EVIDENCE_ANCHORS)
    anchors["exists"] = anchors["path"].map(lambda relative: (repo_root / relative).exists())
    wifs_status = _read_wifs_scorecard(scorecard_path)

    lines = [
        "# WIFS Breadth Decision",
        "",
        f"Run date: {RUN_DATE}",
        "",
        "Decision: freeze the WIFS experimental scope and write the 6-page paper around the checked-in evidence. "
        "Do not make a new model run a WIFS blocker.",
        "",
        (
            f"Current WIFS readiness: packet `{wifs_status['packet_status']}`, artifacts "
            f"{wifs_status['artifacts']}, lint reports {wifs_status['lint_reports']}, "
            f"lint checks {wifs_status['lint_checks']}, claims {wifs_status['claims']}."
        ),
        "",
        "## Option Ranking",
        "",
        _markdown_table(
            display,
            [
                "option_id",
                "decision",
                "paper_use",
                "manuscript_value",
                "deadline_fit",
                "reproducibility_fit",
                "effort",
                "risk",
                "decision_score",
            ],
        ),
        "",
        "## Evidence Anchors",
        "",
        _markdown_table(anchors, ["anchor", "path", "exists", "use"]),
        "",
        "## Writing Rule",
        "",
        "- Lead WIFS with metric-family breadth: same-domain anchor, transfer frontier, reverse operating points, robustness stress, source-heldout stress, and reconstruction ablation.",
        "- Keep CLIP as the transfer ranking frontier and SCP-Fusion as a diagnostic protocol with calibrated operating points.",
        "- Keep combined_v4 and reconstruction_v2 as caveated ablations; do not promote either to the main method.",
        "- If spare compute appears, run source-aware v4 selection as an appendix-only add-on. Do not delay WIFS for a larger source split or true tiled neural/foundation branch.",
        "",
    ]
    return "\n".join(lines), options


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    text, options = build_decision_report(repo_root, repo_root / args.scorecard)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")

    csv_out = Path(args.csv_out)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    options.to_csv(csv_out, index=False)

    print(out_path.resolve())
    print(csv_out.resolve())


if __name__ == "__main__":
    main()
