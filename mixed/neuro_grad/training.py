from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from networks.loss import MSELoss
from networks.optimizer import SGD

from .data import RegressionData
from .model import FullyConnectedRegressor


@dataclass(frozen=True)
class ExperimentConfig:
    n_points: int = 200
    x_min: float = -1.0
    x_max: float = 1.0
    noise_std: float = 0.0
    target_function: str = "sin_pi"
    seed: int = 42
    depth: int = 1
    width: int = 40
    input_size: int = 1
    output_size: int = 1
    epochs: int = 2000
    learning_rate: float = 5e-3
    mc_trajectories: int = 10
    save_every: int = 20
    device: str = "cpu"


@dataclass
class TrainingTrace:
    epochs: np.ndarray
    losses: np.ndarray
    params: np.ndarray
    predictions: np.ndarray
    training_time: float


ProgressCallback = Callable[[str, int, TrainingTrace], None]


def make_seeded_model_factory(
    config: ExperimentConfig,
    *,
    seed_offset: int = 0,
) -> Callable[[], FullyConnectedRegressor]:
    """Return a factory that recreates the same initialized model each time."""
    random_state = np.random.get_state()
    np.random.seed(config.seed + seed_offset)
    try:
        base_model = FullyConnectedRegressor(
            config.depth,
            config.width,
            input_size=config.input_size,
            output_size=config.output_size,
        )
    finally:
        np.random.set_state(random_state)

    def factory() -> FullyConnectedRegressor:
        return copy.deepcopy(base_model)

    return factory


def _prediction_numpy(
    model: FullyConnectedRegressor,
    x: np.ndarray,
) -> np.ndarray:
    return np.asarray(model.forward(x), dtype=np.float32)


def _trace(
    epochs: list[int],
    losses: list[float],
    params: list[np.ndarray],
    predictions: list[np.ndarray],
    training_time: float,
) -> TrainingTrace:
    return TrainingTrace(
        epochs=np.asarray(epochs, dtype=np.int64),
        losses=np.asarray(losses, dtype=np.float64),
        params=np.asarray(params, dtype=np.float32),
        predictions=np.asarray(predictions, dtype=np.float32),
        training_time=float(training_time),
    )


def train_clipped_gd(
    model: FullyConnectedRegressor,
    data: RegressionData,
    config: ExperimentConfig,
    on_update: ProgressCallback | None = None,
) -> TrainingTrace:
    loss_function = MSELoss()
    optimizer = SGD(model.parameters(), learning_rate=config.learning_rate)

    epochs: list[int] = []
    losses: list[float] = []
    params: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    start_time = time.time()

    def store(epoch: int, loss_value: float) -> TrainingTrace:
        epochs.append(epoch)
        losses.append(float(loss_value))
        params.append(model.flat_parameters().astype(np.float32))
        predictions.append(_prediction_numpy(model, data.x))
        partial = _trace(epochs, losses, params, predictions, time.time() - start_time)
        if on_update is not None:
            on_update("gd", epoch, partial)
        return partial

    initial_loss = loss_function(model.forward(data.x), data.y)
    current_trace = store(0, float(initial_loss))

    for epoch in range(1, config.epochs + 1):
        optimizer.zero_grad()
        prediction = model.forward(data.x)
        loss_function(prediction, data.y)
        model.backward(loss_function.backward())

        squared_sum = sum(
            float(np.sum(parameter.grad ** 2))
            for parameter in model.parameters()
        )
        gradient_norm = max(math.sqrt(squared_sum), 1e-12)
        for parameter in model.parameters():
            parameter.grad[...] /= gradient_norm
        optimizer.step()

        if epoch % config.save_every == 0 or epoch == config.epochs:
            stored_loss = loss_function(model.forward(data.x), data.y)
            current_trace = store(epoch, float(stored_loss))

    current_trace.training_time = time.time() - start_time
    return current_trace


def train_zero_temp_mc(
    model_factory: Callable[[], FullyConnectedRegressor],
    data: RegressionData,
    config: ExperimentConfig,
    on_update: ProgressCallback | None = None,
) -> list[TrainingTrace]:
    traces: list[TrainingTrace] = []
    sigma = config.learning_rate * math.sqrt(2.0 * math.pi)

    for trajectory_idx in range(config.mc_trajectories):
        model = model_factory()
        loss_function = MSELoss()
        mutation_rng = np.random.default_rng(config.seed + 1_000_000 + trajectory_idx)

        epochs: list[int] = []
        losses: list[float] = []
        params: list[np.ndarray] = []
        predictions: list[np.ndarray] = []
        start_time = time.time()

        def store(epoch: int, loss_value: float) -> TrainingTrace:
            epochs.append(epoch)
            losses.append(float(loss_value))
            params.append(model.flat_parameters().astype(np.float32))
            predictions.append(_prediction_numpy(model, data.x))
            partial = _trace(epochs, losses, params, predictions, time.time() - start_time)
            if on_update is not None:
                on_update(f"mc_{trajectory_idx}", epoch, partial)
            return partial

        old_loss = float(loss_function(model.forward(data.x), data.y))
        current_trace = store(0, old_loss)

        for epoch in range(1, config.epochs + 1):
            old_state = [
                parameter.data.copy()
                for parameter in model.parameters()
            ]

            for parameter in model.parameters():
                parameter.data[...] += mutation_rng.normal(
                    loc=0.0,
                    scale=sigma,
                    size=parameter.data.shape,
                )
            new_loss = float(loss_function(model.forward(data.x), data.y))

            if new_loss <= old_loss:
                old_loss = new_loss
            else:
                for parameter, old_data in zip(model.parameters(), old_state):
                    parameter.data[...] = old_data
                new_loss = old_loss

            if epoch % config.save_every == 0 or epoch == config.epochs:
                current_trace = store(epoch, new_loss)

        current_trace.training_time = time.time() - start_time
        traces.append(current_trace)

    return traces
