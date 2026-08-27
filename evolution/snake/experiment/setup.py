from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


SNAKE_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class StaticConfig:
    """Pełny, niezmienny protokół eksperymentu Snake."""

    population_size: int = 40
    generations: int = 100
    cx_prob: float = 0.65
    mut_prob: float = 0.30
    mutation_sigma: float = 0.20
    tournament_size: int = 3
    eval_max_steps: int = 30 * 24 * 20
    eval_episodes: int = 3
    random_seed: int = 123
    elites: int = 2
    reevaluate_topk: int = 5

    hidden_size: int = 48
    evaluation_seeds: tuple[int, ...] = tuple(range(20))
    representative_seed: int = 11
    capture_scores: tuple[int, ...] = (0, 25, 50, 75, 100)
    artifact_directory: str = "snake_final_20260824"

    @property
    def output_directory(self) -> Path:
        return SNAKE_DIR / "output"

    @property
    def genome_path(self) -> Path:
        return self.output_directory / "snake_agent.npy"

    @property
    def artifact_path(self) -> Path:
        return self.output_directory / self.artifact_directory

    def protocol(self) -> dict[str, object]:
        values = asdict(self)
        values["evaluation_seeds"] = list(self.evaluation_seeds)
        values["capture_scores"] = list(self.capture_scores)
        return values
