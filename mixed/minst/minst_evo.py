from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from networks.loss import CrossEntropyLoss

from .minst_dataset import MNISTBundle
from .minst_grad import evaluate, iterate_minibatches
from .minst_model import (
    SimpleCNN,
    load_model,
    probabilities_from_logits,
    save_model,
)


@dataclass
class EvolutionConfig:
    generations: int = 12
    population_size: int = 12
    elite_count: int = 2
    mutation_std: float = 0.01
    mutation_scope: str = "classifier"
    samples_per_class: int = 40
    diversity_weight: float = 0.25
    batch_size: int = 64
    seed: int = 42


@dataclass
class EvolutionHistory:
    train_losses: list[float] = field(default_factory=list)
    test_accuracies: list[float] = field(default_factory=list)


def clone_model(model):
    clone = SimpleCNN()
    for source, destination in zip(model.parameters(), clone.parameters()):
        destination.data[...] = source.data
    return clone


def mutate_model(model, mutation_std=0.02, rng=None, *, scope="all"):
    if mutation_std < 0:
        raise ValueError("mutation_std must be non-negative")
    if scope not in {"all", "classifier"}:
        raise ValueError("scope must be 'all' or 'classifier'")

    rng = rng or np.random.default_rng()
    parameters = (
        model.parameters()
        if scope == "all"
        else model.classifier_parameters()
    )
    for parameter in parameters:
        parameter.data[...] += rng.normal(
            0.0,
            mutation_std,
            size=parameter.data.shape,
        )
    return model


def balanced_training_subset(bundle, samples_per_class, rng):
    if samples_per_class < 1:
        raise ValueError("samples_per_class must be positive")

    selected_indices = []
    for label in range(10):
        candidates = np.flatnonzero(bundle.y_train == label)
        if len(candidates) == 0:
            continue
        count = min(samples_per_class, len(candidates))
        selected_indices.extend(
            rng.choice(candidates, size=count, replace=False)
        )
    selected_indices = np.asarray(selected_indices, dtype=np.int64)
    rng.shuffle(selected_indices)
    return (
        bundle.x_train[selected_indices],
        bundle.y_train[selected_indices],
    )


def evaluate_evolution_candidate(
    model,
    images,
    labels,
    *,
    batch_size,
    diversity_weight,
):
    loss_function = CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    probability_sum = np.zeros(10)
    rng = np.random.default_rng(0)

    for batch_images, batch_labels in iterate_minibatches(
        images,
        labels,
        batch_size,
        rng,
        shuffle=False,
    ):
        logits = model.forward(batch_images)
        loss = loss_function.forward(logits, batch_labels)
        probabilities = probabilities_from_logits(logits)
        predictions = np.argmax(probabilities, axis=1)

        total_loss += float(loss) * len(batch_labels)
        total_correct += int(np.sum(predictions == batch_labels))
        total_count += len(batch_labels)
        probability_sum += np.sum(probabilities, axis=0)

    mean_loss = total_loss / max(1, total_count)
    accuracy = total_correct / max(1, total_count)
    mean_probabilities = probability_sum / max(1, total_count)
    uniform = np.full(10, 0.1)
    diversity_penalty = np.sum(
        mean_probabilities
        * np.log((mean_probabilities + 1e-12) / uniform)
    )
    objective = mean_loss + diversity_weight * float(diversity_penalty)
    return objective, mean_loss, accuracy


def evolve_model(bundle: MNISTBundle, base_model=None, config=None, callback=None):
    config = config or EvolutionConfig()
    if config.generations < 1:
        raise ValueError("generations must be positive")
    if config.population_size < 1:
        raise ValueError("population_size must be positive")
    if not 1 <= config.elite_count <= config.population_size:
        raise ValueError("elite_count must be within the population")

    np.random.seed(config.seed)
    rng = np.random.default_rng(config.seed)
    base_model = clone_model(base_model) if base_model is not None else SimpleCNN()
    train_images, train_labels = balanced_training_subset(
        bundle,
        config.samples_per_class,
        rng,
    )
    population = [clone_model(base_model)]
    while len(population) < config.population_size:
        population.append(
            mutate_model(
                clone_model(base_model),
                config.mutation_std,
                rng,
                scope=config.mutation_scope,
            )
        )

    best_model = clone_model(base_model)
    best_objective = float("inf")
    history = EvolutionHistory()

    for generation in range(1, config.generations + 1):
        scored = []
        for individual in population:
            objective, loss, accuracy = evaluate_evolution_candidate(
                individual,
                train_images,
                train_labels,
                batch_size=config.batch_size,
                diversity_weight=config.diversity_weight,
            )
            scored.append((objective, loss, accuracy, individual))
        scored.sort(key=lambda result: result[0])

        objective, loss, train_accuracy, generation_best = scored[0]
        if objective < best_objective:
            best_objective = objective
            best_model = clone_model(generation_best)
        test_accuracy = evaluate(
            generation_best,
            bundle.x_test,
            bundle.y_test,
            batch_size=config.batch_size,
        )
        history.train_losses.append(float(loss))
        history.test_accuracies.append(test_accuracy)
        if callback is not None:
            callback(
                generation,
                loss,
                train_accuracy,
                test_accuracy,
                generation_best,
            )

        elites = [result[3] for result in scored[:config.elite_count]]
        next_population = [clone_model(elite) for elite in elites]
        while len(next_population) < config.population_size:
            parent = elites[int(rng.integers(0, len(elites)))]
            next_population.append(
                mutate_model(
                    clone_model(parent),
                    config.mutation_std,
                    rng,
                    scope=config.mutation_scope,
                )
            )
        population = next_population

    return best_model, history


def load_evo_model(checkpoint_path: str, device: str | None = None):
    if device not in (None, "auto", "cpu"):
        raise ValueError("The NumPy implementation supports CPU execution only")
    return load_model(checkpoint_path), "cpu"


__all__ = [
    "EvolutionConfig",
    "EvolutionHistory",
    "balanced_training_subset",
    "clone_model",
    "evaluate_evolution_candidate",
    "evolve_model",
    "load_evo_model",
    "mutate_model",
    "save_model",
]
