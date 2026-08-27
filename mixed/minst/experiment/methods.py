from __future__ import annotations

import time

import numpy as np

from networks.loss import CrossEntropyLoss
from networks.optimizer import Adam

from ..minst_dataset import MNISTBundle
from ..minst_evo import EvolutionConfig, evolve_model
from ..minst_grad import evaluate, evaluate_loss, train_one_epoch
from ..minst_model import SimpleCNN, predict_images
from .setup import BenchmarkConfig, ExtendedConfig


def clone_model(model: SimpleCNN) -> SimpleCNN:
    clone = SimpleCNN()
    for source, destination in zip(model.parameters(), clone.parameters()):
        destination.data[...] = source.data
    return clone


def train_gradient(
    bundle: MNISTBundle,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    initial_model: SimpleCNN | None = None,
) -> tuple[SimpleCNN, list[dict[str, float]], float]:
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    model = clone_model(initial_model) if initial_model is not None else SimpleCNN()
    optimizer = Adam(model.parameters(), learning_rate=learning_rate)
    loss_function = CrossEntropyLoss()
    trace: list[dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(
            model,
            bundle.x_train,
            bundle.y_train,
            optimizer,
            loss_function,
            batch_size,
            rng,
        )
        trace.append(
            {
                "step": epoch,
                "loss": float(loss),
                "validation_accuracy": float(
                    evaluate(
                        model,
                        bundle.x_test,
                        bundle.y_test,
                        batch_size=batch_size,
                    )
                ),
                "elapsed": time.perf_counter() - started,
            }
        )
    return model, trace, time.perf_counter() - started


def train_evolution(
    bundle: MNISTBundle,
    *,
    seed: int,
    config: BenchmarkConfig | ExtendedConfig,
    initial_model: SimpleCNN | None,
    mutation_scope: str,
    mutation_std: float,
    diversity_weight: float = 0.1,
) -> tuple[SimpleCNN, list[dict[str, float]], float]:
    trace: list[dict[str, float]] = []
    started = time.perf_counter()

    def callback(step, loss, _train_accuracy, validation_accuracy, _model):
        trace.append(
            {
                "step": int(step),
                "loss": float(loss),
                "validation_accuracy": float(validation_accuracy),
                "elapsed": time.perf_counter() - started,
            }
        )

    evo_config = EvolutionConfig(
        generations=config.evolution_generations,
        population_size=config.population_size,
        elite_count=config.elite_count,
        mutation_std=mutation_std,
        mutation_scope=mutation_scope,
        samples_per_class=config.samples_per_class,
        diversity_weight=diversity_weight,
        batch_size=config.batch_size,
        seed=seed,
    )
    model, _ = evolve_model(
        bundle,
        base_model=initial_model,
        config=evo_config,
        callback=callback,
    )
    return model, trace, time.perf_counter() - started


def confusion_matrix(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    matrix = np.zeros((10, 10), dtype=np.int64)
    np.add.at(matrix, (labels.astype(int), predictions.astype(int)), 1)
    return matrix


def evaluate_model(
    model: SimpleCNN,
    bundle: MNISTBundle,
    batch_size: int,
) -> dict[str, object]:
    predictions, confidences = predict_images(model, bundle.x_test)
    matrix = confusion_matrix(bundle.y_test, predictions)
    class_accuracy = np.divide(
        np.diag(matrix),
        matrix.sum(axis=1),
        out=np.zeros(10),
        where=matrix.sum(axis=1) > 0,
    )
    return {
        "test_accuracy": float(np.mean(predictions == bundle.y_test)),
        "test_loss": float(
            evaluate_loss(model, bundle.x_test, bundle.y_test, batch_size)
        ),
        "mean_confidence": float(np.mean(confidences)),
        "class_accuracy": class_accuracy,
        "confusion": matrix,
        "predictions": predictions,
        "confidences": confidences,
    }


def robustness_rows(
    model: SimpleCNN,
    bundle: MNISTBundle,
    method: str,
    seed: int,
) -> list[dict[str, float | int | str]]:
    rng = np.random.default_rng(seed + 1000)
    rows = []
    for noise_std in (0.0, 0.15, 0.30):
        noisy = np.clip(
            bundle.x_test + rng.normal(0.0, noise_std, bundle.x_test.shape),
            0.0,
            1.0,
        ).astype(np.float32)
        predictions, _ = predict_images(model, noisy)
        rows.append(
            {
                "seed": seed,
                "method": method,
                "noise_std": noise_std,
                "accuracy": float(np.mean(predictions == bundle.y_test)),
            }
        )
    return rows
