from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluate import ExperimentFrames
from .output import OutputPaths


COLORS = {"GD": "#f59e0b", "MC": "#06b6d4"}


def _comparison_plot(
    summary: pd.DataFrame,
    *,
    scenario: str,
    variants: tuple[str, ...],
    labels: tuple[str, ...],
    path: Path,
    logarithmic: bool = False,
) -> Path:
    subset = summary[
        (summary.scenario == scenario) & summary.method.isin(("GD", "MC"))
    ]
    x = np.arange(len(variants))
    width = 0.36
    figure, axis = plt.subplots(figsize=(11.5, 5.2))
    for offset, method in ((-width / 2, "GD"), (width / 2, "MC")):
        indexed = subset[subset.method == method].set_index("variant")
        values = [float(indexed.loc[variant, "test_mse_mean"]) for variant in variants]
        errors = [float(indexed.loc[variant, "test_mse_std"]) for variant in variants]
        axis.bar(
            x + offset,
            values,
            width,
            yerr=errors,
            capsize=3,
            label=method,
            color=COLORS[method],
        )
    axis.set_xticks(x, labels)
    axis.set_ylabel("MSE na zbiorze testowym")
    if logarithmic:
        axis.set_yscale("log")
    axis.grid(axis="y", alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_function_comparison(frames: ExperimentFrames, paths: OutputPaths) -> Path:
    return _comparison_plot(
        frames.summary,
        scenario="function",
        variants=("sin_pi", "gaussian_bump", "wave_trend", "bowl", "mixed_surface"),
        labels=("sin(πx)", "maksimum Gaussa", "fala + trend", "paraboloida", "mieszana 2D"),
        path=paths.figures / "regresja_wyniki_funkcje.png",
        logarithmic=True,
    )


def plot_model_comparison(frames: ExperimentFrames, paths: OutputPaths) -> Path:
    return _comparison_plot(
        frames.summary,
        scenario="model",
        variants=("small", "baseline", "large"),
        labels=("1×10", "1×40", "2×40"),
        path=paths.figures / "regresja_wyniki_skalowanie.png",
        logarithmic=True,
    )


def plot_noise_comparison(frames: ExperimentFrames, paths: OutputPaths) -> Path:
    return _comparison_plot(
        frames.summary,
        scenario="noise",
        variants=("0.00", "0.10", "0.25"),
        labels=("0,00", "0,10", "0,25"),
        path=paths.figures / "regresja_wyniki_szum.png",
        logarithmic=True,
    )


def plot_cost_quality(frames: ExperimentFrames, paths: OutputPaths) -> Path:
    subset = frames.paired[
        (frames.paired.scenario == "baseline")
        & frames.paired.method.isin(("GD", "MC"))
    ]
    figure, axis = plt.subplots(figsize=(8.6, 5.4))
    for method in ("GD", "MC"):
        rows = subset[subset.method == method]
        axis.scatter(
            rows.training_time_s,
            rows.test_mse,
            color=COLORS[method],
            alpha=0.8,
            s=48,
            label=method,
        )
    axis.set_xlabel("Czas treningu [s]")
    axis.set_ylabel("MSE na zbiorze testowym")
    axis.set_yscale("log")
    axis.grid(alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    figure.tight_layout()
    path = paths.figures / "regresja_wyniki_koszt_jakosc.png"
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_objective_change(frames: ExperimentFrames, paths: OutputPaths) -> Path:
    subset = frames.summary[frames.summary.scenario == "objective"].set_index("method")
    methods = ("GD-MSE", "GD-surrogate", "MC")
    labels = ("GD — MSE", "GD — surogat", "MC")
    colors = ("#f59e0b", "#ef4444", "#06b6d4")
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    axes[0].bar(labels, [subset.loc[m, "test_mse_mean"] for m in methods], color=colors)
    axes[0].set_ylabel("MSE na zbiorze testowym")
    axes[0].set_yscale("log")
    axes[1].bar(
        labels,
        [subset.loc[m, "violation_rate_mean"] for m in methods],
        color=colors,
    )
    axes[1].set_ylabel("Odsetek przekroczeń progu")
    for axis in axes:
        axis.tick_params(axis="x", rotation=12)
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    path = paths.figures / "regresja_wyniki_zmiana_celu.png"
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_representative_traces(frames: ExperimentFrames, paths: OutputPaths) -> Path:
    cases = (
        ("baseline", "sin_pi", "sin(πx)"),
        ("function", "wave_trend", "fala + trend"),
        ("function", "mixed_surface", "powierzchnia mieszana"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.3))
    for axis, (scenario, variant, title) in zip(axes, cases):
        subset = frames.traces[
            (frames.traces.scenario == scenario)
            & (frames.traces.variant == variant)
            & frames.traces.method.isin(("GD", "MC"))
        ]
        grouped = (
            subset.groupby(["method", "objective_evaluations"], as_index=False)
            .test_mse.mean()
        )
        for method in ("GD", "MC"):
            rows = grouped[grouped.method == method]
            axis.plot(
                rows.objective_evaluations,
                rows.test_mse,
                color=COLORS[method],
                label=method,
                linewidth=2,
            )
        axis.set_title(title)
        axis.set_xlabel("Ewaluacje funkcji celu")
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Średnie MSE testowe")
    axes[-1].legend(frameon=False)
    figure.tight_layout()
    path = paths.figures / "regresja_wyniki_przebieg_trzy_przypadki.png"
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_curve_fit(prediction_path: Path, output_path: Path) -> Path:
    with np.load(prediction_path) as data:
        x = data["test_x"].reshape(-1)
        target = data["test_clean_y"].reshape(-1)
        gd = data["gd_prediction"].reshape(-1)
        mc = data["mc_predictions"][:, :, 0]
        order = np.argsort(x)
        lower = np.quantile(mc, 0.1, axis=0)
        upper = np.quantile(mc, 0.9, axis=0)
        mean = np.mean(mc, axis=0)
        train_x = data["train_x"].reshape(-1)
        train_y = data["train_y"].reshape(-1)

    figure, axis = plt.subplots(figsize=(9.4, 5.2))
    axis.scatter(train_x, train_y, s=12, color="#94a3b8", alpha=0.45, label="Dane")
    axis.plot(x[order], target[order], color="#16a34a", linewidth=2.4, label="Funkcja")
    axis.plot(x[order], gd[order], color=COLORS["GD"], linewidth=2, label="GD")
    axis.fill_between(x[order], lower[order], upper[order], color=COLORS["MC"], alpha=0.18)
    axis.plot(x[order], mean[order], color=COLORS["MC"], linewidth=2, label="MC — średnia")
    axis.grid(alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncol=4)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_surface_fit(prediction_path: Path, output_path: Path) -> Path:
    with np.load(prediction_path) as data:
        shape = tuple(int(value) for value in data["grid_shape"])
        x = data["test_x"][:, 0].reshape(shape)
        y = data["test_x"][:, 1].reshape(shape)
        surfaces = (
            data["test_clean_y"].reshape(shape),
            data["gd_prediction"].reshape(shape),
            data["mc_mean_prediction"].reshape(shape),
        )
    minimum = min(float(surface.min()) for surface in surfaces)
    maximum = max(float(surface.max()) for surface in surfaces)
    figure = plt.figure(figsize=(14.2, 4.3))
    grid = figure.add_gridspec(1, 4, width_ratios=(1, 1, 1, 0.045), wspace=0.18)
    axes = [figure.add_subplot(grid[0, index]) for index in range(3)]
    color_axis = figure.add_subplot(grid[0, 3])
    image = None
    for axis, surface, title in zip(axes, surfaces, ("Funkcja", "GD", "MC — średnia")):
        image = axis.pcolormesh(x, y, surface, shading="auto", cmap="viridis", vmin=minimum, vmax=maximum)
        axis.set_title(title)
        axis.set_xlabel("x₁")
        axis.set_aspect("equal")
    axes[0].set_ylabel("x₂")
    figure.colorbar(image, cax=color_axis, label="Wartość")
    figure.subplots_adjust(left=0.06, right=0.96, bottom=0.13, top=0.9)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output_path


def make_plots(frames: ExperimentFrames, paths: OutputPaths) -> list[Path]:
    figures = [
        plot_function_comparison(frames, paths),
        plot_model_comparison(frames, paths),
        plot_noise_comparison(frames, paths),
        plot_cost_quality(frames, paths),
        plot_objective_change(frames, paths),
        plot_representative_traces(frames, paths),
    ]
    fitting_cases = (
        ("baseline_sin_pi.npz", "regresja_dopasowanie_sin_pi.png", "curve"),
        ("function_wave_trend.npz", "regresja_dopasowanie_wave_trend.png", "curve"),
        ("function_mixed_surface.npz", "regresja_dopasowanie_mixed_surface.png", "surface"),
    )
    for prediction_name, figure_name, kind in fitting_cases:
        prediction_path = paths.predictions / prediction_name
        if not prediction_path.exists():
            continue
        output_path = paths.figures / figure_name
        figures.append(
            plot_curve_fit(prediction_path, output_path)
            if kind == "curve"
            else plot_surface_fit(prediction_path, output_path)
        )
    return figures
