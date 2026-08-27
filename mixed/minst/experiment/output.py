from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd


@dataclass(frozen=True)
class OutputPaths:
    root: Path
    figures: Path
    checkpoints: Path

    @classmethod
    def from_root(cls, root: Path) -> "OutputPaths":
        return cls(root=root, figures=root / "figures", checkpoints=root / "checkpoints")

    def prepare(self) -> None:
        for directory in (self.root, self.figures, self.checkpoints):
            directory.mkdir(parents=True, exist_ok=True)


def save_frame(path: Path, rows) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def save_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
