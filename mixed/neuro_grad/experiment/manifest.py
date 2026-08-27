from __future__ import annotations

import platform
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .evaluate import ExperimentFrames
from .jobs import standard_jobs
from .setup import StaticConfig


def build_manifest(
    config: StaticConfig,
    frames: ExperimentFrames,
    *,
    duration_seconds: float | None,
    generated_from_training: bool,
    figure_paths: Sequence[Path],
    artifact_files: Sequence[str],
) -> dict[str, object]:
    jobs = standard_jobs(config)
    return {
        "generated_from_training": generated_from_training,
        "duration_s": duration_seconds,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "configuration": config.protocol(),
        "jobs": [asdict(job) for job in jobs],
        "changed_objective": {
            "pretrain_epochs": config.objective_pretrain_epochs,
            "finetune_epochs": config.objective_finetune_epochs,
            "threshold_lambda": config.objective_threshold_lambda,
            "threshold_tau": config.objective_threshold_tau,
            "alpha": config.objective_alpha,
            "mc_trajectories": config.objective_mc_trajectories,
            "seeds": config.objective_seeds,
        },
        "seed_protocol": {
            "standard_data_seed": "10000 + replicate",
            "standard_initialization_seed": "20000 + replicate",
            "standard_mc_seed": "1000000 + 100 * replicate + trajectory",
            "changed_objective_data_seed": "30000 + replicate",
            "changed_objective_initialization_seed": "40000 + replicate",
            "changed_objective_mc_seed": "2000000 + 100 * replicate + trajectory",
        },
        "test_protocol": {
            "curve_points": 1000,
            "surface_grid": [60, 60],
            "noise_std": 0.0,
            "coordinate_interval_curve": [-0.995, 0.995],
            "coordinate_interval_surface": [-0.985, 0.985],
        },
        "run_count": len(frames.runs),
        "paired_run_count": len(frames.paired),
        "figures": [str(path.relative_to(config.output_directory)) for path in figure_paths],
        "artifacts": list(artifact_files),
    }
