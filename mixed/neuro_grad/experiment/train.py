from __future__ import annotations

import math
import time

import numpy as np

from networks.loss import MSELoss
from networks.optimizer import SGD

from ..data import RegressionData, generate_data
from ..model import FullyConnectedRegressor
from .setup import Job


def parameter_count(model: FullyConnectedRegressor) -> int:
    return int(sum(parameter.data.size for parameter in model.parameters()))


def make_model(job: Job, init_seed: int) -> FullyConnectedRegressor:
    state = np.random.get_state()
    np.random.seed(init_seed)
    try:
        return FullyConnectedRegressor(
            depth=job.depth,
            width=job.width,
            input_size=2 if job.experiment_type == "surface" else 1,
            output_size=1,
        )
    finally:
        np.random.set_state(state)


def make_datasets(job: Job, data_seed: int) -> tuple[RegressionData, RegressionData]:
    train = generate_data(
        n_points=job.n_points,
        noise_std=job.noise_std,
        seed=data_seed,
        target_function=job.target_function,
        experiment_type=job.experiment_type,
        grid_size=job.grid_size,
    )
    test = generate_data(
        n_points=1000,
        x_min=-0.995 if job.experiment_type == "curve" else -0.985,
        x_max=0.995 if job.experiment_type == "curve" else 0.985,
        noise_std=0.0,
        seed=data_seed + 500_000,
        target_function=job.target_function,
        experiment_type=job.experiment_type,
        grid_size=60,
    )
    return train, test


def exact_objective(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    threshold_lambda: float = 0.0,
    threshold_tau: float = 0.05,
) -> tuple[float, float, float]:
    errors = np.asarray(prediction) - np.asarray(target)
    mse = float(np.mean(errors**2))
    violation_rate = float(np.mean(np.abs(errors) > threshold_tau))
    return mse + threshold_lambda * violation_rate, mse, violation_rate


def metrics(
    model: FullyConnectedRegressor,
    train: RegressionData,
    test: RegressionData,
    *,
    threshold_lambda: float = 0.0,
    threshold_tau: float = 0.05,
) -> dict[str, float]:
    train_prediction = np.asarray(model.forward(train.x))
    objective, train_mse, violation_rate = exact_objective(
        train_prediction,
        train.y,
        threshold_lambda=threshold_lambda,
        threshold_tau=threshold_tau,
    )
    clean_mse = float(np.mean((train_prediction - train.clean_y) ** 2))
    test_prediction = np.asarray(model.forward(test.x))
    test_mse = float(np.mean((test_prediction - test.clean_y) ** 2))
    return {
        "objective": objective,
        "train_mse": train_mse,
        "clean_mse": clean_mse,
        "test_mse": test_mse,
        "violation_rate": violation_rate,
    }


def trace_row(
    *,
    run_id: str,
    method: str,
    step: int,
    elapsed: float,
    objective_evaluations: int,
    values: dict[str, float],
    acceptance_rate: float | None,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "method": method,
        "step": step,
        "elapsed_time_s": elapsed,
        "objective_evaluations": objective_evaluations,
        **values,
        "acceptance_rate": acceptance_rate,
    }


def train_gd(
    model: FullyConnectedRegressor,
    train: RegressionData,
    test: RegressionData,
    *,
    epochs: int,
    alpha: float,
    run_id: str,
    save_every: int = 40,
    objective_mode: str = "mse",
    threshold_lambda: float = 0.0,
    threshold_tau: float = 0.05,
    surrogate_temperature: float = 0.025,
) -> tuple[FullyConnectedRegressor, list[dict[str, object]], float]:
    optimizer = SGD(model.parameters(), learning_rate=alpha)
    loss_function = MSELoss()
    rows: list[dict[str, object]] = []
    start = time.perf_counter()
    initial = metrics(
        model,
        train,
        test,
        threshold_lambda=threshold_lambda,
        threshold_tau=threshold_tau,
    )
    rows.append(
        trace_row(
            run_id=run_id,
            method="GD" if objective_mode == "mse" else "GD-surrogate",
            step=0,
            elapsed=0.0,
            objective_evaluations=0,
            values=initial,
            acceptance_rate=None,
        )
    )

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        prediction = np.asarray(model.forward(train.x))
        loss_function.forward(prediction, train.y)
        gradient = loss_function.backward()
        if objective_mode == "threshold_surrogate":
            error = prediction - train.y
            logits = np.clip(
                (np.abs(error) - threshold_tau) / surrogate_temperature,
                -50.0,
                50.0,
            )
            soft_indicator = 1.0 / (1.0 + np.exp(-logits))
            soft_derivative = soft_indicator * (1.0 - soft_indicator)
            gradient = gradient + (
                threshold_lambda
                * soft_derivative
                * np.sign(error)
                / (surrogate_temperature * error.size)
            )

        model.backward(gradient)
        squared_sum = sum(
            float(np.sum(parameter.grad**2)) for parameter in model.parameters()
        )
        norm = max(math.sqrt(squared_sum), 1e-12)
        for parameter in model.parameters():
            parameter.grad[...] /= norm
        optimizer.step()

        if epoch % save_every == 0 or epoch == epochs:
            values = metrics(
                model,
                train,
                test,
                threshold_lambda=threshold_lambda,
                threshold_tau=threshold_tau,
            )
            rows.append(
                trace_row(
                    run_id=run_id,
                    method="GD" if objective_mode == "mse" else "GD-surrogate",
                    step=epoch,
                    elapsed=time.perf_counter() - start,
                    objective_evaluations=epoch,
                    values=values,
                    acceptance_rate=None,
                )
            )
    return model, rows, time.perf_counter() - start


def train_mc(
    model: FullyConnectedRegressor,
    train: RegressionData,
    test: RegressionData,
    *,
    epochs: int,
    mutation_sigma: float,
    mutation_seed: int,
    run_id: str,
    save_every: int = 40,
    threshold_lambda: float = 0.0,
    threshold_tau: float = 0.05,
) -> tuple[FullyConnectedRegressor, list[dict[str, object]], float, int]:
    rng = np.random.default_rng(mutation_seed)
    rows: list[dict[str, object]] = []
    start = time.perf_counter()
    accepted = 0
    current_prediction = np.asarray(model.forward(train.x))
    old_objective, _, _ = exact_objective(
        current_prediction,
        train.y,
        threshold_lambda=threshold_lambda,
        threshold_tau=threshold_tau,
    )
    initial = metrics(
        model,
        train,
        test,
        threshold_lambda=threshold_lambda,
        threshold_tau=threshold_tau,
    )
    rows.append(
        trace_row(
            run_id=run_id,
            method="MC",
            step=0,
            elapsed=0.0,
            objective_evaluations=0,
            values=initial,
            acceptance_rate=0.0,
        )
    )

    for epoch in range(1, epochs + 1):
        old_state = [parameter.data.copy() for parameter in model.parameters()]
        for parameter in model.parameters():
            parameter.data[...] += rng.normal(
                0.0, mutation_sigma, size=parameter.data.shape
            )
        new_prediction = np.asarray(model.forward(train.x))
        new_objective, _, _ = exact_objective(
            new_prediction,
            train.y,
            threshold_lambda=threshold_lambda,
            threshold_tau=threshold_tau,
        )
        if new_objective <= old_objective:
            old_objective = new_objective
            accepted += 1
        else:
            for parameter, state in zip(model.parameters(), old_state):
                parameter.data[...] = state

        if epoch % save_every == 0 or epoch == epochs:
            values = metrics(
                model,
                train,
                test,
                threshold_lambda=threshold_lambda,
                threshold_tau=threshold_tau,
            )
            rows.append(
                trace_row(
                    run_id=run_id,
                    method="MC",
                    step=epoch,
                    elapsed=time.perf_counter() - start,
                    objective_evaluations=epoch,
                    values=values,
                    acceptance_rate=accepted / epoch,
                )
            )
    return model, rows, time.perf_counter() - start, accepted
