from __future__ import annotations

import json
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame

from ..snake_trainer import DEAPTrainer, save_genome
from .evaluate import evaluate_genome
from .manifest import build_manifest
from .output import OutputPaths, save_csv, save_json
from .plots import (
    make_contact_sheet,
    make_evaluation_plots,
    make_training_plot,
)
from .setup import StaticConfig


def train_model(config: StaticConfig) -> tuple[np.ndarray, float, list[dict[str, int | float]]]:
    trainer = DEAPTrainer(config, hidden_size=config.hidden_size)
    _, best_fitness, best_genome = trainer.evolve(seed_genome=None)
    save_genome(str(config.genome_path), best_genome)
    return best_genome, best_fitness, trainer.history


def generate_artifacts(
    genome: np.ndarray,
    config: StaticConfig,
    *,
    history: list[dict[str, int | float]] | None = None,
    best_training_fitness: float | None = None,
    generated_from_training: bool = False,
) -> dict[str, object]:
    paths = OutputPaths.from_config(config)
    paths.prepare()
    result = evaluate_genome(genome, config, paths)

    save_csv(paths.tables / "snake_evaluation.csv", result.rows)
    save_csv(paths.tables / "snake_timeline.csv", result.timeline)
    make_contact_sheet(paths, config)
    make_evaluation_plots(result.rows, result.timeline, paths.figures)

    if history:
        save_csv(paths.tables / "snake_training.csv", history)
        make_training_plot(history, paths.figures)
    else:
        # Ocena istniejącego genomu nie może odziedziczyć plików po treningu.
        for stale_path in (
            paths.tables / "snake_training.csv",
            paths.figures / "snake_trening.png",
        ):
            if stale_path.exists():
                stale_path.unlink()

    artifact_files = sorted(
        str(path.relative_to(paths.root))
        for path in paths.root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    genome_bytes = config.genome_path.read_bytes()
    manifest = build_manifest(
        genome,
        genome_bytes,
        result,
        config,
        best_training_fitness=best_training_fitness,
        generated_from_training=generated_from_training,
        artifact_files=artifact_files,
    )
    save_json(paths.root / "manifest.json", manifest)
    return manifest


def evaluate_saved_genome() -> dict[str, object]:
    """Odtwarza ocenę i artefakty bez ponownego treningu."""

    config = StaticConfig()
    genome = np.load(config.genome_path)
    return generate_artifacts(genome, config)


def run_experiment() -> dict[str, object]:
    """Wykonuje kompletny statyczny eksperyment: trening, ocenę i raport."""

    config = StaticConfig()
    config.output_directory.mkdir(parents=True, exist_ok=True)
    genome, best_fitness, history = train_model(config)
    return generate_artifacts(
        genome,
        config,
        history=history,
        best_training_fitness=best_fitness,
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
