from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


NEURO_GRAD_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Job:
    scenario: str
    variant: str
    experiment_type: str = "curve"
    target_function: str = "sin_pi"
    noise_std: float = 0.0
    depth: int = 1
    width: int = 40
    epochs: int = 2000
    alpha: float = 0.01
    seeds: int = 12
    mc_trajectories: int = 5
    n_points: int = 200
    grid_size: int = 25


@dataclass(frozen=True)
class StaticConfig:
    """Pełny, niezmienny protokół eksperymentu regresji."""

    alpha: float = 0.01
    output_directory_name: str = "regression_20260820"
    trace_interval: int = 40

    objective_pretrain_epochs: int = 1200
    objective_finetune_epochs: int = 800
    objective_alpha: float = 0.001
    objective_threshold_lambda: float = 0.15
    objective_threshold_tau: float = 0.05
    objective_seeds: int = 12
    objective_mc_trajectories: int = 5

    @property
    def output_directory(self) -> Path:
        return NEURO_GRAD_DIR / "output" / self.output_directory_name

    def protocol(self) -> dict[str, object]:
        return asdict(self)
