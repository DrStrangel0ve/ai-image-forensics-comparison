from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utils import read_json


@dataclass(frozen=True)
class DatasetEntry:
    key: str
    title: str
    source: str
    ref: str
    local_dir: Path
    updated: str | None
    size_mb: int | None
    notes: str


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_catalog(path: str | Path | None = None) -> dict[str, DatasetEntry]:
    catalog_path = Path(path) if path else project_root() / "configs" / "datasets.json"
    raw: dict[str, Any] = read_json(catalog_path)
    entries: dict[str, DatasetEntry] = {}
    for key, value in raw.items():
        entries[key] = DatasetEntry(
            key=key,
            title=str(value["title"]),
            source=str(value["source"]),
            ref=str(value["ref"]),
            local_dir=project_root() / str(value["local_dir"]),
            updated=value.get("updated"),
            size_mb=value.get("size_mb"),
            notes=str(value.get("notes", "")),
        )
    return entries
