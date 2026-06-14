from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_sota_gap_closure_plan_prioritizes_official_benchmark_runs(tmp_path: Path) -> None:
    out_path = tmp_path / "sota_gap_closure_plan.md"
    csv_out = tmp_path / "sota_gap_closure_plan.csv"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_sota_gap_closure_plan.py"),
            "--out-path",
            str(out_path),
            "--csv-out",
            str(csv_out),
        ],
        cwd=ROOT,
        check=True,
    )

    text = out_path.read_text(encoding="utf-8")
    plan = pd.read_csv(csv_out)

    assert "SOTA Gap Closure Plan" in text
    assert "not a claim of leaderboard performance" in text
    assert "Best NTIRE-style local proxy row" in text
    assert "genimage_official_split_eval" in text
    assert "high_res_tiled_foundation_eval" in text
    assert "evaluate_frozen_encoder_model.py" in text
    assert plan.iloc[0]["task_id"] == "genimage_official_split_eval"
    assert plan["rank"].is_monotonic_increasing
    assert "ntire_released_protocol_replay" in set(plan["task_id"])
    assert (
        plan.loc[
            plan["task_id"] == "high_res_tiled_foundation_eval", "status"
        ].item()
        == "core_transform_stress_probes_complete"
    )
    assert set(plan["status"]) >= {
        "bounded_transfer_probe_complete_pretrained_branch_next",
        "ready_when_data_available",
    }
