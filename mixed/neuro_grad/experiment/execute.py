from __future__ import annotations

import copy
import math

import numpy as np

from ..data import RegressionData
from ..model import FullyConnectedRegressor
from .jobs import changed_objective_job
from .output import OutputPaths
from .setup import Job, StaticConfig
from .train import make_datasets, make_model, metrics, parameter_count, train_gd, train_mc


def run_record(
    *,
    job: Job,
    run_id: str,
    method: str,
    replicate: int,
    data_seed: int,
    init_seed: int,
    optimizer_seed: int,
    model: FullyConnectedRegressor,
    train: RegressionData,
    test: RegressionData,
    training_time: float,
    objective_evaluations: int,
    accepted_mutations: int | None,
    threshold_lambda: float = 0.0,
    threshold_tau: float = 0.05,
) -> dict[str, object]:
    values = metrics(
        model,
        train,
        test,
        threshold_lambda=threshold_lambda,
        threshold_tau=threshold_tau,
    )
    return {
        "run_id": run_id,
        "scenario": job.scenario,
        "variant": job.variant,
        "method": method,
        "replicate": replicate,
        "data_seed": data_seed,
        "init_seed": init_seed,
        "optimizer_seed": optimizer_seed,
        "experiment_type": job.experiment_type,
        "target_function": job.target_function,
        "noise_std": job.noise_std,
        "depth": job.depth,
        "width": job.width,
        "parameter_count": parameter_count(model),
        "alpha": job.alpha,
        "mutation_sigma": job.alpha * math.sqrt(2.0 * math.pi),
        "epochs": objective_evaluations,
        "objective_evaluations": objective_evaluations,
        "forward_passes": objective_evaluations,
        "backward_passes": objective_evaluations if method.startswith("GD") else 0,
        "accepted_mutations": accepted_mutations,
        "acceptance_rate": (
            accepted_mutations / objective_evaluations
            if accepted_mutations is not None and objective_evaluations
            else None
        ),
        **values,
        "training_time_s": training_time,
        "status": "ok",
    }


def save_predictions(
    paths: OutputPaths,
    *,
    key: str,
    train: RegressionData,
    test: RegressionData,
    gd_model: FullyConnectedRegressor,
    mc_models: list[FullyConnectedRegressor],
) -> None:
    mc_predictions = np.stack(
        [np.asarray(model.forward(test.x)) for model in mc_models], axis=0
    )
    np.savez_compressed(
        paths.predictions / f"{key}.npz",
        train_x=train.x,
        train_y=train.y,
        train_clean_y=train.clean_y,
        test_x=test.x,
        test_clean_y=test.clean_y,
        grid_shape=np.asarray(test.grid_shape or (), dtype=np.int64),
        gd_prediction=np.asarray(gd_model.forward(test.x)),
        mc_predictions=mc_predictions,
        mc_mean_prediction=np.mean(mc_predictions, axis=0),
    )


def run_standard_job(
    job: Job,
    paths: OutputPaths,
    *,
    trace_interval: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    runs: list[dict[str, object]] = []
    traces: list[dict[str, object]] = []
    for replicate in range(job.seeds):
        data_seed = 10_000 + replicate
        init_seed = 20_000 + replicate
        train, test = make_datasets(job, data_seed)
        base_model = make_model(job, init_seed)

        gd_id = f"{job.scenario}-{job.variant}-r{replicate:02d}-gd"
        gd_model, gd_trace, gd_time = train_gd(
            copy.deepcopy(base_model),
            train,
            test,
            epochs=job.epochs,
            alpha=job.alpha,
            run_id=gd_id,
            save_every=trace_interval,
        )
        traces.extend(
            {
                **row,
                "scenario": job.scenario,
                "variant": job.variant,
                "replicate": replicate,
            }
            for row in gd_trace
        )
        runs.append(
            run_record(
                job=job,
                run_id=gd_id,
                method="GD",
                replicate=replicate,
                data_seed=data_seed,
                init_seed=init_seed,
                optimizer_seed=init_seed,
                model=gd_model,
                train=train,
                test=test,
                training_time=gd_time,
                objective_evaluations=job.epochs,
                accepted_mutations=None,
            )
        )

        mc_models: list[FullyConnectedRegressor] = []
        for trajectory in range(job.mc_trajectories):
            mutation_seed = 1_000_000 + replicate * 100 + trajectory
            mc_id = (
                f"{job.scenario}-{job.variant}-r{replicate:02d}-mc{trajectory:02d}"
            )
            mc_model, mc_trace, mc_time, accepted = train_mc(
                copy.deepcopy(base_model),
                train,
                test,
                epochs=job.epochs,
                mutation_sigma=job.alpha * math.sqrt(2.0 * math.pi),
                mutation_seed=mutation_seed,
                run_id=mc_id,
                save_every=trace_interval,
            )
            mc_models.append(mc_model)
            traces.extend(
                {
                    **row,
                    "scenario": job.scenario,
                    "variant": job.variant,
                    "replicate": replicate,
                }
                for row in mc_trace
            )
            runs.append(
                run_record(
                    job=job,
                    run_id=mc_id,
                    method="MC",
                    replicate=replicate,
                    data_seed=data_seed,
                    init_seed=init_seed,
                    optimizer_seed=mutation_seed,
                    model=mc_model,
                    train=train,
                    test=test,
                    training_time=mc_time,
                    objective_evaluations=job.epochs,
                    accepted_mutations=accepted,
                )
            )

        if replicate == 0 and job.scenario in {"baseline", "function"}:
            save_predictions(
                paths,
                key=f"{job.scenario}_{job.variant}",
                train=train,
                test=test,
                gd_model=gd_model,
                mc_models=mc_models,
            )
    return runs, traces


def run_changed_objective(
    config: StaticConfig,
    paths: OutputPaths,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    job = changed_objective_job(config)
    runs: list[dict[str, object]] = []
    traces: list[dict[str, object]] = []

    for replicate in range(job.seeds):
        data_seed = 30_000 + replicate
        init_seed = 40_000 + replicate
        train, test = make_datasets(job, data_seed)
        base_model = make_model(job, init_seed)
        checkpoint, _, _ = train_gd(
            base_model,
            train,
            test,
            epochs=config.objective_pretrain_epochs,
            alpha=config.alpha,
            run_id=f"objective-pretrain-r{replicate:02d}",
            save_every=config.objective_pretrain_epochs,
        )

        variants = (
            ("GD-MSE", "mse", 0.0),
            (
                "GD-surrogate",
                "threshold_surrogate",
                config.objective_threshold_lambda,
            ),
        )
        final_models: dict[str, FullyConnectedRegressor] = {}
        for method, mode, optimization_lambda in variants:
            run_id = f"objective-threshold-r{replicate:02d}-{method.lower()}"
            model, trace, duration = train_gd(
                copy.deepcopy(checkpoint),
                train,
                test,
                epochs=job.epochs,
                alpha=job.alpha,
                run_id=run_id,
                save_every=config.trace_interval,
                objective_mode=mode,
                threshold_lambda=optimization_lambda,
                threshold_tau=config.objective_threshold_tau,
            )
            final_models[method] = model
            traces.extend(
                {
                    **row,
                    "scenario": job.scenario,
                    "variant": job.variant,
                    "replicate": replicate,
                }
                for row in trace
            )
            runs.append(
                run_record(
                    job=job,
                    run_id=run_id,
                    method=method,
                    replicate=replicate,
                    data_seed=data_seed,
                    init_seed=init_seed,
                    optimizer_seed=init_seed,
                    model=model,
                    train=train,
                    test=test,
                    training_time=duration,
                    objective_evaluations=job.epochs,
                    accepted_mutations=None,
                    threshold_lambda=config.objective_threshold_lambda,
                    threshold_tau=config.objective_threshold_tau,
                )
            )

        mc_models: list[FullyConnectedRegressor] = []
        for trajectory in range(job.mc_trajectories):
            mutation_seed = 2_000_000 + replicate * 100 + trajectory
            run_id = f"objective-threshold-r{replicate:02d}-mc{trajectory:02d}"
            model, trace, duration, accepted = train_mc(
                copy.deepcopy(checkpoint),
                train,
                test,
                epochs=job.epochs,
                mutation_sigma=job.alpha * math.sqrt(2.0 * math.pi),
                mutation_seed=mutation_seed,
                run_id=run_id,
                save_every=config.trace_interval,
                threshold_lambda=config.objective_threshold_lambda,
                threshold_tau=config.objective_threshold_tau,
            )
            mc_models.append(model)
            traces.extend(
                {
                    **row,
                    "scenario": job.scenario,
                    "variant": job.variant,
                    "replicate": replicate,
                }
                for row in trace
            )
            runs.append(
                run_record(
                    job=job,
                    run_id=run_id,
                    method="MC",
                    replicate=replicate,
                    data_seed=data_seed,
                    init_seed=init_seed,
                    optimizer_seed=mutation_seed,
                    model=model,
                    train=train,
                    test=test,
                    training_time=duration,
                    objective_evaluations=job.epochs,
                    accepted_mutations=accepted,
                    threshold_lambda=config.objective_threshold_lambda,
                    threshold_tau=config.objective_threshold_tau,
                )
            )

        if replicate == 0:
            np.savez_compressed(
                paths.predictions / "objective_threshold.npz",
                train_x=train.x,
                train_y=train.y,
                test_x=test.x,
                test_clean_y=test.clean_y,
                gd_mse_prediction=final_models["GD-MSE"].forward(test.x),
                gd_surrogate_prediction=final_models["GD-surrogate"].forward(test.x),
                mc_mean_prediction=np.mean(
                    np.stack([model.forward(test.x) for model in mc_models]), axis=0
                ),
            )
    return runs, traces
