from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np

from .evaluate import EvaluationResult
from .setup import REPOSITORY_ROOT, StaticConfig


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPOSITORY_ROOT.resolve()))


def build_manifest(
    genomes: dict[int, np.ndarray],
    result: EvaluationResult,
    config: StaticConfig,
    *,
    best_training_fitness: dict[int, float] | None,
    generated_from_training: bool,
    figure_paths: Sequence[Path],
    artifact_files: Sequence[str],
) -> dict[str, object]:
    max_seconds = config.max_seconds_by_phase
    return {
        "generated_from_training": generated_from_training,
        "best_training_fitness": (
            {str(phase): value for phase, value in best_training_fitness.items()}
            if best_training_fitness
            else None
        ),
        "training_protocol": config.protocol(),
        "evaluation_protocol": {
            "scenarios": list(config.evaluation_scenarios),
            "lives": config.lives,
            "fps": config.fps,
            "max_seconds_by_phase": max_seconds,
            "max_steps_by_phase": {
                str(phase): seconds * config.fps
                for phase, seconds in max_seconds.items()
            },
            "phase_frame_skip": {
                str(phase.phase): phase.frame_skip for phase in config.phases
            },
            "scenario_note": (
                "Scenarios 0-11 form the final deterministic evaluation set; "
                "some offsets also occur in the training set."
            ),
        },
        "scenario8": asdict(result.representative),
        "genomes": {
            str(phase): {
                "path": relative(config.genome_path(phase)),
                "sha256": hashlib.sha256(
                    config.genome_path(phase).read_bytes()
                ).hexdigest(),
                "parameters": int(genomes[phase].size),
            }
            for phase in (1, 2, 3)
        },
        "figures": [relative(path) for path in figure_paths],
        "screenshots": {
            key: relative(Path(path)) for key, path in result.screenshots.items()
        },
        "artifacts": list(artifact_files),
    }
