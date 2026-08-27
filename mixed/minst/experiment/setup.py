from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


MINST_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class BenchmarkConfig:
    seeds: tuple[int, ...] = (11, 29, 47)
    train_size: int = 3000
    validation_size: int = 1000
    test_size: int = 2000
    gd_epochs: int = 3
    gd_extra_epochs: int = 1
    batch_size: int = 64
    learning_rate: float = 1e-3
    evolution_generations: int = 8
    population_size: int = 8
    elite_count: int = 2
    samples_per_class: int = 20
    random_mutation_std: float = 0.01
    warm_mutation_std: float = 0.003
    output_directory_name: str = "mnist_20260822"

    @property
    def output_directory(self) -> Path:
        return MINST_DIR / "output" / self.output_directory_name

    def protocol(self) -> dict[str, object]:
        values = asdict(self)
        values["seeds"] = list(self.seeds)
        return values


@dataclass(frozen=True)
class ExtendedConfig:
    seed: int = 42
    train_size: int = 5000
    monitor_size: int = 1000
    gd_epochs: int = 3
    batch_size: int = 64
    learning_rate: float = 1e-3
    evolution_generations: int = 150
    population_size: int = 30
    elite_count: int = 2
    mutation_std: float = 0.02
    mutation_scope: str = "all"
    samples_per_class: int = 40
    diversity_weight: float = 0.30
    output_directory_name: str = "mnist_extended_20260827"

    @property
    def output_directory(self) -> Path:
        return MINST_DIR / "output" / self.output_directory_name

    def protocol(self) -> dict[str, object]:
        return asdict(self)
