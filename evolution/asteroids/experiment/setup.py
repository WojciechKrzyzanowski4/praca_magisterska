from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


ASTEROIDS_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ASTEROIDS_DIR.parents[1]


@dataclass(frozen=True)
class PhaseConfig:
    phase: int
    name: str
    generations: int
    frame_skip: int


@dataclass(frozen=True)
class StaticConfig:
    """Pełny, niezmienny protokół trójfazowego eksperymentu Asteroids."""

    population_size: int = 80
    generations: int = 50
    cx_prob: float = 0.65
    mut_prob: float = 0.03
    mutation_sigma: float = 0.20
    mutation_sigma_min: float = 0.05
    tournament_size: int = 3
    eval_max_steps: int = 60 * 120
    random_seed: int = 123
    elites: int = 2
    reevaluate_topk: int = 5
    training_scenarios: tuple[int, ...] = (0, 2, 4, 6, 8, 10, 12, 14)

    hidden_size: int = 48
    evaluation_scenarios: tuple[int, ...] = tuple(range(12))
    representative_scenario: int = 8
    lives: int = 1
    fps: int = 60
    artifact_directory: str = "asteroids_final_20260824"

    @property
    def phases(self) -> tuple[PhaseConfig, ...]:
        return (
            PhaseConfig(1, "P1-DODGE", 150, 3),
            PhaseConfig(2, "P2-AIM", 150, 2),
            PhaseConfig(3, "P3-KILL", 50, 2),
        )

    @property
    def max_seconds_by_phase(self) -> dict[int, int]:
        return {1: 120, 2: 120, 3: 300}

    @property
    def output_directory(self) -> Path:
        return ASTEROIDS_DIR / "output"

    @property
    def artifact_path(self) -> Path:
        return self.output_directory / self.artifact_directory

    def genome_path(self, phase: int) -> Path:
        if phase not in (1, 2, 3):
            raise ValueError("phase must be 1, 2, or 3")
        return self.output_directory / f"asteroid_killer_phase{phase}.npy"

    def phase(self, phase: int) -> PhaseConfig:
        return self.phases[phase - 1]

    def protocol(self) -> dict[str, object]:
        values = asdict(self)
        values["training_scenarios"] = list(self.training_scenarios)
        values["evaluation_scenarios"] = list(self.evaluation_scenarios)
        values["phases"] = [asdict(phase) for phase in self.phases]
        values["max_seconds_by_phase"] = self.max_seconds_by_phase
        return values
