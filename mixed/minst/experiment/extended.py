from __future__ import annotations

from pathlib import Path

import numpy as np

from ..minst_evo import balanced_training_subset, evaluate_evolution_candidate
from ..minst_model import load_model, save_model
from .manifest import build_extended_manifest
from .methods import evaluate_model, train_evolution, train_gradient
from .output import OutputPaths, save_frame, save_json
from .plots import plot_extended
from .setup import ExtendedConfig
from .splits import prepare_extended_bundle


def checkpoint_paths(paths: OutputPaths) -> dict[str, Path]:
    return {
        "GD": paths.checkpoints / "GD.npz",
        "EVO-random": paths.checkpoints / "EVO_random.npz",
    }


def finalize_extended(
    config: ExtendedConfig,
    paths: OutputPaths,
    *,
    generated_from_training: bool,
) -> dict[str, object]:
    figures = plot_extended(paths)
    manifest = build_extended_manifest(
        config,
        paths,
        generated_from_training=generated_from_training,
        figures=figures,
    )
    save_json(paths.root / "manifest.json", manifest)
    return manifest


def evaluate_saved_extended() -> dict[str, object]:
    config = ExtendedConfig()
    paths = OutputPaths.from_root(config.output_directory)
    paths.prepare()
    models = {
        method: load_model(path) for method, path in checkpoint_paths(paths).items()
    }
    bundle = prepare_extended_bundle(config)
    rows = extended_summary(models, bundle, config)
    save_frame(paths.root / "summary.csv", rows)
    return finalize_extended(config, paths, generated_from_training=False)


def extended_summary(models, bundle, config: ExtendedConfig) -> list[dict[str, object]]:
    gd_result = evaluate_model(models["GD"], bundle, config.batch_size)
    evo_result = evaluate_model(models["EVO-random"], bundle, config.batch_size)
    images, labels = balanced_training_subset(
        bundle,
        config.samples_per_class,
        np.random.default_rng(config.seed),
    )
    objective, objective_loss, objective_accuracy = evaluate_evolution_candidate(
        models["EVO-random"],
        images,
        labels,
        batch_size=config.batch_size,
        diversity_weight=config.diversity_weight,
    )
    return [
        {
            "method": "GD",
            "monitor_accuracy": gd_result["test_accuracy"],
            "monitor_loss": gd_result["test_loss"],
            "objective_subset_accuracy": np.nan,
            "objective_subset_loss": np.nan,
            "objective_value": np.nan,
        },
        {
            "method": "EVO-random",
            "monitor_accuracy": evo_result["test_accuracy"],
            "monitor_loss": evo_result["test_loss"],
            "objective_subset_accuracy": objective_accuracy,
            "objective_subset_loss": objective_loss,
            "objective_value": objective,
        },
    ]


def run_extended() -> dict[str, object]:
    config = ExtendedConfig()
    paths = OutputPaths.from_root(config.output_directory)
    paths.prepare()
    bundle = prepare_extended_bundle(config)
    gd_model, gd_trace, _ = train_gradient(
        bundle,
        seed=config.seed,
        epochs=config.gd_epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
    )
    evo_model, evo_trace, _ = train_evolution(
        bundle,
        seed=config.seed,
        config=config,
        initial_model=None,
        mutation_scope=config.mutation_scope,
        mutation_std=config.mutation_std,
        diversity_weight=config.diversity_weight,
    )
    models = {"GD": gd_model, "EVO-random": evo_model}
    for method, path in checkpoint_paths(paths).items():
        save_model(models[method], path)
    traces = [
        {"method": method, **row}
        for method, trace in (("GD", gd_trace), ("EVO-random", evo_trace))
        for row in trace
    ]
    save_frame(paths.root / "traces.csv", traces)
    save_frame(paths.root / "summary.csv", extended_summary(models, bundle, config))
    return finalize_extended(config, paths, generated_from_training=True)
