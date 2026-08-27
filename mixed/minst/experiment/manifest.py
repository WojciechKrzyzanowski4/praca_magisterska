from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

from ..minst_model import SimpleCNN
from .output import OutputPaths
from .setup import BenchmarkConfig, ExtendedConfig, MINST_DIR


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_counts() -> tuple[int, int]:
    model = SimpleCNN()
    return (
        int(sum(parameter.data.size for parameter in model.parameters())),
        int(sum(parameter.data.size for parameter in model.classifier_parameters())),
    )


def checkpoint_manifest(paths: OutputPaths) -> dict[str, dict[str, object]]:
    return {
        path.stem: {
            "path": str(path.relative_to(MINST_DIR)),
            "sha256": sha256(path),
        }
        for path in sorted(paths.checkpoints.glob("*.npz"))
    }


def build_benchmark_manifest(
    config: BenchmarkConfig,
    paths: OutputPaths,
    *,
    generated_from_training: bool,
    figures: Sequence[Path],
) -> dict[str, object]:
    parameters, classifier_parameters = model_counts()
    return {
        **config.protocol(),
        "generated_from_training": generated_from_training,
        "parameter_count": parameters,
        "classifier_parameter_count": classifier_parameters,
        "data_protocol": {
            "source": "mixed/minst/data/mnist.npz",
            "split": "stratified without replacement",
            "training_pool": "official MNIST training split",
            "validation_pool": "official MNIST training split excluding selected training images",
            "test_pool": "official MNIST test split",
        },
        "robustness_protocol": {
            "noise_std": [0.0, 0.15, 0.30],
            "noise_seed_rule": "experiment_seed + 1000",
        },
        "checkpoints": checkpoint_manifest(paths),
        "figures": [str(path.relative_to(paths.root)) for path in figures],
    }


def build_extended_manifest(
    config: ExtendedConfig,
    paths: OutputPaths,
    *,
    generated_from_training: bool,
    figures: Sequence[Path],
) -> dict[str, object]:
    checkpoints = checkpoint_manifest(paths)
    return {
        "generated_from_training": generated_from_training,
        "configuration": config.protocol(),
        "data_protocol": {
            "training": "first 5000 images of the official MNIST training split",
            "monitoring": "first 1000 images of the official MNIST test split",
            "objective_subset": "40 images per class selected without replacement with seed 42",
        },
        "individual_evaluations": config.evolution_generations * config.population_size,
        "model_files": {
            "GD": checkpoints["GD"],
            "EVO-random": checkpoints["EVO_random"],
        },
        "figures": [str(path.relative_to(paths.root)) for path in figures],
    }
