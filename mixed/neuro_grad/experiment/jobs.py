from __future__ import annotations

from .setup import Job, StaticConfig


def standard_jobs(config: StaticConfig) -> tuple[Job, ...]:
    alpha = config.alpha
    jobs: list[Job] = [
        Job("baseline", "sin_pi", alpha=alpha, seeds=20, mc_trajectories=10),
    ]
    for name in ("sin_pi", "gaussian_bump", "wave_trend"):
        jobs.append(Job("function", name, target_function=name, alpha=alpha))
    for name in ("bowl", "mixed_surface"):
        jobs.append(
            Job(
                "function",
                name,
                experiment_type="surface",
                target_function=name,
                alpha=alpha,
                n_points=625,
                grid_size=25,
            )
        )
    jobs.extend(
        (
            Job("model", "small", depth=1, width=10, alpha=alpha),
            Job("model", "baseline", depth=1, width=40, alpha=alpha),
            Job("model", "large", depth=2, width=40, alpha=alpha),
            Job("noise", "0.00", noise_std=0.0, alpha=alpha),
            Job("noise", "0.10", noise_std=0.1, alpha=alpha),
            Job("noise", "0.25", noise_std=0.25, alpha=alpha),
        )
    )
    return tuple(jobs)


def changed_objective_job(config: StaticConfig) -> Job:
    return Job(
        scenario="objective",
        variant="threshold",
        epochs=config.objective_finetune_epochs,
        alpha=config.objective_alpha,
        seeds=config.objective_seeds,
        mc_trajectories=config.objective_mc_trajectories,
    )
