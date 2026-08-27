from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd

from .setup import StaticConfig


@dataclass(frozen=True)
class OutputPaths:
    root: Path
    figures: Path
    tables: Path
    predictions: Path

    @classmethod
    def from_config(cls, config: StaticConfig) -> "OutputPaths":
        root = config.output_directory
        return cls(
            root=root,
            figures=root / "figures",
            tables=root / "tables",
            predictions=root / "predictions",
        )

    def prepare(self) -> None:
        for directory in (self.root, self.figures, self.tables, self.predictions):
            directory.mkdir(parents=True, exist_ok=True)


def save_frame(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False)


def save_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
