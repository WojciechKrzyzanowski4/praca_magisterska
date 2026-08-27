from __future__ import annotations

import json
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame

from ..asteroid_trainer import DEAPCurriculum, PhaseSpec, save_genome
from .evaluate import evaluate_genomes, scenario_rows
from .manifest import build_manifest
from .output import OutputPaths, save_csv, save_json
from .plots import make_evaluation_plots, make_training_plot
from .setup import StaticConfig


def train_curriculum(
    config: StaticConfig,
) -> tuple[
    dict[int, np.ndarray],
    dict[int, float],
    dict[int, list[dict[str, int | float]]],
]:
    genomes: dict[int, np.ndarray] = {}
    fitness: dict[int, float] = {}
    histories: dict[int, list[dict[str, int | float]]] = {}
    seed_genome = None

    for phase_config in config.phases:
        trainer = DEAPCurriculum(config, hidden_size=config.hidden_size)
        phase = PhaseSpec(
            name=phase_config.name,
            step_phase=phase_config.phase,
            generations=phase_config.generations,
            frame_skip=phase_config.frame_skip,
        )
        _, best_fitness, best_genome = trainer.evolve_phase(
            phase,
            seed_genome=seed_genome,
            checkpoint_path=None,
        )
        save_genome(str(config.genome_path(phase_config.phase)), best_genome)
        genomes[phase_config.phase] = best_genome
        fitness[phase_config.phase] = best_fitness
        histories[phase_config.phase] = trainer.history
        seed_genome = best_genome

    return genomes, fitness, histories


def generate_artifacts(
    genomes: dict[int, np.ndarray],
    config: StaticConfig,
    *,
    histories: dict[int, list[dict[str, int | float]]] | None = None,
    best_training_fitness: dict[int, float] | None = None,
    generated_from_training: bool = False,
) -> dict[str, object]:
    paths = OutputPaths.from_config(config)
    paths.prepare()
    result = evaluate_genomes(genomes, config, paths)

    save_csv(paths.tables / "asteroids_scenarios.csv", scenario_rows(result))
    save_csv(paths.tables / "asteroids_summary.csv", result.summary)
    save_csv(
        paths.tables / "asteroids_scenario8_timeline.csv",
        result.representative_timeline,
    )
    figure_paths = make_evaluation_plots(result, paths)

    if histories:
        for phase, history in histories.items():
            save_csv(paths.tables / f"asteroids_training_phase{phase}.csv", history)
        training_plot = make_training_plot(histories, paths)
        if training_plot is not None:
            figure_paths.append(training_plot)
    else:
        for phase in (1, 2, 3):
            stale_path = paths.tables / f"asteroids_training_phase{phase}.csv"
            if stale_path.exists():
                stale_path.unlink()
        stale_plot = paths.figures / "asteroids_trening.png"
        if stale_plot.exists():
            stale_plot.unlink()

    artifact_files = sorted(
        str(path.relative_to(paths.root))
        for path in paths.root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    manifest = build_manifest(
        genomes,
        result,
        config,
        best_training_fitness=best_training_fitness,
        generated_from_training=generated_from_training,
        figure_paths=figure_paths,
        artifact_files=artifact_files,
    )
    save_json(paths.root / "manifest.json", manifest)
    return manifest


def evaluate_saved_genomes() -> dict[str, object]:
    """Odtwarza ocenę i artefakty bez ponownego treningu curriculum."""

    config = StaticConfig()
    genomes = {
        phase: np.load(config.genome_path(phase)) for phase in (1, 2, 3)
    }
    return generate_artifacts(genomes, config)


def run_experiment() -> dict[str, object]:
    """Trenuje kolejno fazy 1, 2 i 3, a następnie wykonuje pełną ocenę."""

    config = StaticConfig()
    config.output_directory.mkdir(parents=True, exist_ok=True)
    genomes, fitness, histories = train_curriculum(config)
    return generate_artifacts(
        genomes,
        config,
        histories=histories,
        best_training_fitness=fitness,
        generated_from_training=True,
    )


def main() -> None:
    try:
        manifest = run_experiment()
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
