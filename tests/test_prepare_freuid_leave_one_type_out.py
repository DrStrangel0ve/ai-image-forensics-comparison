from __future__ import annotations

from pathlib import Path

import pandas as pd


def test_leave_one_type_out_cli(tmp_path: Path) -> None:
    rows = []
    for document_type in ("A/DL", "B/ID", "C/DL"):
        for label in (0, 1):
            rows.append({"id": f"{document_type}_{label}", "label": label, "type": document_type})
    input_csv = tmp_path / "input.csv"
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    pd.DataFrame(rows).to_csv(input_csv, index=False)

    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "prepare_freuid_leave_one_type_out.py"),
            "--input-csv",
            str(input_csv),
            "--holdout-type",
            "B/ID",
            "--train-out",
            str(train_csv),
            "--val-out",
            str(val_csv),
        ],
        cwd=root,
        check=True,
    )

    train = pd.read_csv(train_csv)
    val = pd.read_csv(val_csv)
    assert set(train["type"]) == {"A/DL", "C/DL"}
    assert set(val["type"]) == {"B/ID"}
