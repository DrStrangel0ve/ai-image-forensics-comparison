from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd


DEFAULT_RUN_DATE = date.today()

OPPORTUNITIES = [
    {
        "opportunity_id": "dfrws_usa_2026_poster",
        "name": "DFRWS-USA 2026 poster/demo",
        "type": "poster/demo",
        "status": "open",
        "deadline": "2026-07-07",
        "fit": "Best near-term public demo target for the repo, poster, and reproducibility package.",
        "action": "Submit the DFRWS poster after using the selected seed-29 qualitative grid and exporting the final poster file.",
        "source_url": "https://dfrws.org/call-for-papers-is-open-for-dfrws-usa-2026/",
        "source_note": "Official DFRWS-USA CFP lists poster submissions on a rolling deadline until July 7, 2026.",
    },
    {
        "opportunity_id": "wifs_2026_paper",
        "name": "IEEE WIFS 2026 paper",
        "type": "full paper",
        "status": "open",
        "deadline": "2026-07-15",
        "fit": "Strong formal forensics venue if the paper stays compact and claim-conservative.",
        "action": "Draft the 6-page paper from the WIFS skeleton, result tables, and caveated section prose.",
        "source_url": "https://wifs2026.utt.fr/call-for-papers",
        "source_note": "Official WIFS CFP lists a July 15, 2026 deadline, 6-page limit including bibliography and figures, and single-blind review.",
    },
    {
        "opportunity_id": "dff_2026_workshop",
        "name": "DFF-2026 at ACM Multimedia",
        "type": "workshop paper",
        "status": "open",
        "deadline": "2026-07-16",
        "fit": "Best full-paper fit for SCP-Fusion as a diagnostic protocol with robustness and failure analysis.",
        "action": "Build the DFF paper around SCP-Fusion diagnostics, transform robustness, failure grids, and source-slice caveats.",
        "source_url": "https://iplab.dmi.unict.it/mfs/acm-dff-ws-2026/",
        "source_note": "Official DFF workshop page says workshop paper submission is open until July 16 and uses OpenReview.",
    },
    {
        "opportunity_id": "dfrws_apac_2026_poster",
        "name": "DFRWS-APAC 2026 poster",
        "type": "poster",
        "status": "future_open",
        "deadline": "2026-09-30",
        "fit": "Good fallback/extension poster target after DFRWS-USA, especially for a polished follow-up demo.",
        "action": "Re-use the DFRWS-USA poster package after WIFS/DFF paper feedback, if a second poster venue is useful.",
        "source_url": "https://dfrws.org/conferences/dfrws-apac-2026/",
        "source_note": "Official DFRWS-APAC page lists poster submissions through September 30, 2026 with rolling notification until the program is full.",
    },
    {
        "opportunity_id": "ntire_2026_robust_aigc",
        "name": "NTIRE 2026 Robust AI-Generated Image Detection in the Wild",
        "type": "benchmark/challenge",
        "status": "closed_benchmark",
        "deadline": "2026-06-04",
        "fit": "Best robustness benchmark target for future SCP-Fusion v1/v2 work: many generators and real-world transforms.",
        "action": "Use the challenge report and CodaBench page as a benchmark target; monitor for dataset/code release and NTIRE 2027.",
        "source_url": "https://arxiv.org/abs/2604.11487",
        "source_note": "Challenge report describes 108,750 real images, 185,750 generated images, 42 generators, 36 transforms, and ROC-AUC evaluation.",
    },
    {
        "opportunity_id": "imageclef_2026_deepfake",
        "name": "ImageCLEF 2026 Deepfake Detection and Generation",
        "type": "benchmark/challenge",
        "status": "closed_benchmark",
        "deadline": "2026-05-07",
        "fit": "Useful benchmark style reference for source/generalization evaluation, but the 2026 run submission window is closed.",
        "action": "Use its detection/generation setup as a design reference; watch for 2027 registration.",
        "source_url": "https://www.imageclef.org/2026/deepfake-detection-and-generation",
        "source_note": "Official ImageCLEF page lists detection task end/run submission deadline as May 7, 2026.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a current publication/challenge opportunity watchlist for the project."
    )
    parser.add_argument(
        "--out-path",
        default="reports/opportunity_watchlist_2026_06_14.md",
        help="Markdown watchlist report to write.",
    )
    parser.add_argument(
        "--csv-out",
        default="reports/assets/opportunity_watchlist.csv",
        help="Machine-readable watchlist CSV to write.",
    )
    parser.add_argument(
        "--run-date",
        default=DEFAULT_RUN_DATE.isoformat(),
        help="Date used for deadline countdowns, in YYYY-MM-DD format. Defaults to today's local date.",
    )
    return parser.parse_args()


def _days_until(deadline: str, run_date: date) -> int:
    return (date.fromisoformat(deadline) - run_date).days


def build_watchlist(run_date: date = DEFAULT_RUN_DATE) -> tuple[str, pd.DataFrame]:
    frame = pd.DataFrame(OPPORTUNITIES)
    frame.insert(
        4,
        "days_until_deadline",
        frame["deadline"].map(lambda deadline: _days_until(str(deadline), run_date)),
    )
    frame["urgency"] = frame.apply(_urgency, axis=1)
    return _write_markdown(frame, run_date), frame


def _urgency(row: pd.Series) -> str:
    if row["status"] == "closed_benchmark":
        return "benchmark_only"
    days = int(row["days_until_deadline"])
    if days <= 14:
        return "submit_now"
    if days <= 35:
        return "active_window"
    return "watch"


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


def _write_markdown(frame: pd.DataFrame, run_date: date) -> str:
    open_frame = frame[frame["status"].isin(["open", "future_open"])].copy()
    benchmark_frame = frame[frame["status"] == "closed_benchmark"].copy()
    summary = (
        frame.groupby(["status", "urgency"], sort=False)
        .size()
        .reset_index(name="count")
    )
    lines = [
        "# Opportunity Watchlist",
        "",
        f"Run date: {run_date.isoformat()}",
        "",
        "Generated by `scripts/build_opportunity_watchlist.py` from manually verified official venue/challenge pages.",
        "",
        "This watchlist separates currently useful submission targets from closed benchmark tracks that should shape the next SCP-Fusion experiments.",
        "",
        "## Summary",
        "",
        _markdown_table(summary, ["status", "urgency", "count"]),
        "",
        "## Open Or Future Submission Targets",
        "",
        _markdown_table(
            open_frame,
            [
                "opportunity_id",
                "name",
                "type",
                "deadline",
                "days_until_deadline",
                "urgency",
                "fit",
                "action",
                "source_url",
            ],
        ),
        "",
        "## Closed Benchmark Tracks To Mine",
        "",
        _markdown_table(
            benchmark_frame,
            [
                "opportunity_id",
                "name",
                "deadline",
                "fit",
                "action",
                "source_url",
                "source_note",
            ],
        ),
        "",
        "## Project Implications",
        "",
        "- Submit/polish order remains DFRWS poster first, WIFS compact paper second, DFF broader workshop paper third.",
        "- NTIRE 2026 is no longer an active submission target, but it is the strongest external benchmark shape for robustness: many generators, many transforms, ROC-AUC reporting.",
        "- ImageCLEF 2026 is closed, but its detection/generation coupling is a useful evaluation model for future source-heldout and adversarial-generator testing.",
        "- No current open challenge found in this pass is a better immediate target than the three checked-in venues; the project should keep monitoring NTIRE/ImageCLEF-style 2027 calls.",
        "",
        "## Verification Notes",
        "",
        "- Deadline and source text should be rechecked before any actual upload.",
        "- `closed_benchmark` entries are not submission opportunities; they are dataset/protocol targets for future experiments.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    text, frame = build_watchlist(date.fromisoformat(args.run_date))
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
