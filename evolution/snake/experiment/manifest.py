from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np

from ..snake_constants import config as game_config
from .evaluate import EvaluationResult
from .setup import StaticConfig


def build_manifest(
    genome: np.ndarray,
    genome_bytes: bytes,
    result: EvaluationResult,
    config: StaticConfig,
    *,
    best_training_fitness: float | None,
    generated_from_training: bool,
    artifact_files: Sequence[str],
) -> dict[str, object]:
    scores = np.asarray([row["score"] for row in result.rows], dtype=float)
    steps = np.asarray([row["steps"] for row in result.rows], dtype=float)
    return {
        "genome": config.genome_path.name,
        "sha256": hashlib.sha256(genome_bytes).hexdigest(),
        "parameters": int(genome.size),
        "generated_from_training": generated_from_training,
        "best_training_fitness": best_training_fitness,
        "training_protocol": config.protocol(),
        "evaluation_runs": len(result.rows),
        "evaluation_seeds": list(config.evaluation_seeds),
        "max_steps": config.eval_max_steps,
        "environment": {
            "grid_width": game_config.GRID_WIDTH,
            "grid_height": game_config.GRID_HEIGHT,
            "max_steps_without_food": game_config.MAX_STEPS_WITHOUT_FOOD,
        },
        "score_mean": float(scores.mean()),
        "score_median": float(np.median(scores)),
        "score_min": int(scores.min()),
        "score_max": int(scores.max()),
        "steps_mean": float(steps.mean()),
        "representative_seed": config.representative_seed,
        "representative": result.representative,
        "artifacts": list(artifact_files),
    }
