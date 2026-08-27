from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from .setup import StaticConfig


@dataclass(frozen=True)
class OutputPaths:
    root: Path
    figures: Path
    screenshots: Path
    tables: Path

    @classmethod
    def from_config(cls, config: StaticConfig) -> "OutputPaths":
        root = config.artifact_path
        return cls(
            root=root,
            figures=root / "figures",
            screenshots=root / "screenshots",
            tables=root / "tables",
        )

    def prepare(self) -> None:
        for directory in (self.root, self.figures, self.screenshots, self.tables):
            directory.mkdir(parents=True, exist_ok=True)


def save_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"Nie można zapisać pustej tabeli: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)


def save_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
