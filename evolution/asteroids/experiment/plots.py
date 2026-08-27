from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

from ..asteroid_constants import config as game_config
from .evaluate import EvaluationResult, ScenarioResult
from .output import OutputPaths


PHASE_NAMES = {1: "Faza 1", 2: "Faza 2", 3: "Faza 3"}


def plot_phase_summary(
    summary_rows: list[dict[str, int | float]], paths: OutputPaths
) -> Path:
    phases = [int(row["phase"]) for row in summary_rows]
    labels = [PHASE_NAMES[phase] for phase in phases]
    colors = ["#3b82f6", "#8b5cf6", "#16a34a"]
    figure, axes = plt.subplots(1, 3, figsize=(12.8, 4.1))
    metrics = [
        ("mean_seconds", "Średni czas [s]", "Przeżycie"),
        ("mean_alignment", "Średni cosinus kąta", "Ustawienie na cel"),
        ("mean_kills", "Średnia liczba", "Zniszczone asteroidy"),
    ]
    for axis, (key, ylabel, title) in zip(axes, metrics):
        values = [float(row[key]) for row in summary_rows]
        bars = axis.bar(labels, values, color=colors, width=0.62)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    figure.tight_layout()
    path = paths.figures / "asteroids_wyniki_fazy.png"
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_phase3_scenarios(
    results: list[ScenarioResult], paths: OutputPaths
) -> Path:
    scenarios = [result.scenario for result in results]
    colors = ["#16a34a" if result.won else "#64748b" for result in results]
    figure, axes = plt.subplots(2, 1, figsize=(11.5, 6.8), sharex=True)

    bars = axes[0].bar(scenarios, [result.waves for result in results], color=colors)
    axes[0].set_ylabel("Ukończone fale")
    axes[0].set_ylim(0, 5.65)
    axes[0].set_yticks(range(6))
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].spines[["top", "right"]].set_visible(False)
    for bar, result in zip(bars, results):
        if result.won:
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                5.08,
                "WYGRANA",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    axes[1].bar(scenarios, [result.seconds for result in results], color=colors)
    axes[1].set_ylabel("Czas epizodu [s]")
    axes[1].set_xlabel("Scenariusz")
    axes[1].set_xticks(scenarios)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    path = paths.figures / "asteroids_wyniki_scenariusze.png"
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_representative_timeline(
    records: list[dict[str, int | float]], paths: OutputPaths
) -> tuple[Path, Path]:
    time = [float(row["second"]) for row in records]
    figure, axes = plt.subplots(2, 1, figsize=(11.5, 6.8), sharex=True)
    axes[0].plot(
        time,
        [row["kills"] for row in records],
        label="Zniszczenia",
        color="#16a34a",
        linewidth=2,
    )
    axes[0].plot(
        time,
        [row["shots"] for row in records],
        label="Strzały",
        color="#3b82f6",
        linewidth=1.6,
    )
    axes[0].set_ylabel("Liczba skumulowana")
    axes[0].legend(frameon=False, ncol=2)
    axes[0].grid(alpha=0.25)
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].plot(
        time,
        [row["remaining_asteroids"] for row in records],
        color="#8b5cf6",
        linewidth=1.8,
    )
    axes[1].set_ylabel("Asteroidy na planszy")
    axes[1].set_xlabel("Czas [s]")
    axes[1].grid(alpha=0.25)
    axes[1].spines[["top", "right"]].set_visible(False)
    previous_wave = 0
    for row in records:
        wave = int(row["waves"])
        if wave > previous_wave:
            axes[1].axvline(float(row["second"]), color="#64748b", alpha=0.35)
            axes[1].text(
                float(row["second"]),
                axes[1].get_ylim()[1] * 0.92,
                f"F{wave}",
                fontsize=8,
                ha="center",
            )
            previous_wave = wave
    figure.tight_layout()
    timeline_path = paths.figures / "asteroids_wyniki_przebieg_zwyciestwa.png"
    figure.savefig(timeline_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    x = np.asarray([row["ship_x"] for row in records])
    y = np.asarray([row["ship_y"] for row in records])
    elapsed = np.asarray(time)
    figure, axis = plt.subplots(figsize=(9.6, 6.7))
    scatter = axis.scatter(x, y, c=elapsed, cmap="viridis", s=18, linewidths=0)
    axis.plot(x, y, color="#64748b", alpha=0.35, linewidth=0.8)
    axis.scatter([x[0]], [y[0]], color="#2563eb", s=70, marker="o", label="Start")
    axis.scatter([x[-1]], [y[-1]], color="#16a34a", s=90, marker="*", label="Koniec")
    axis.set_xlim(0, game_config.WIDTH)
    axis.set_ylim(game_config.HEIGHT, 0)
    axis.set_aspect("equal")
    axis.set_xlabel("Położenie X")
    axis.set_ylabel("Położenie Y")
    axis.grid(alpha=0.18)
    axis.legend(frameon=False)
    figure.colorbar(scatter, ax=axis, label="Czas [s]")
    figure.tight_layout()
    trajectory_path = paths.figures / "asteroids_wyniki_trajektoria.png"
    figure.savefig(trajectory_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return timeline_path, trajectory_path


def make_contact_sheet(result: EvaluationResult, paths: OutputPaths) -> Path:
    entries = [
        ("phase1", "Faza 1"),
        ("phase2", "Faza 2"),
        ("phase3_action", "Faza 3"),
        ("phase3_victory", "Wygrana"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 9.2))
    for axis, (key, label) in zip(axes.flat, entries):
        axis.imshow(plt.imread(result.screenshots[key]))
        axis.set_title(label)
        axis.axis("off")
    figure.tight_layout(pad=0.8)
    path = paths.figures / "asteroids_wyniki_klatki.png"
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="black")
    plt.close(figure)
    return path


def make_evaluation_plots(result: EvaluationResult, paths: OutputPaths) -> list[Path]:
    phase3 = [row for row in result.scenarios if row.phase == 3]
    timeline, trajectory = plot_representative_timeline(
        result.representative_timeline, paths
    )
    return [
        plot_phase_summary(result.summary, paths),
        plot_phase3_scenarios(phase3, paths),
        timeline,
        trajectory,
        make_contact_sheet(result, paths),
    ]


def make_training_plot(
    histories: dict[int, list[dict[str, int | float]]], paths: OutputPaths
) -> Path | None:
    if not histories:
        return None
    figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.2))
    for phase, axis in zip((1, 2, 3), axes):
        history = histories[phase]
        generations = [row["generation"] for row in history]
        axis.plot(
            generations,
            [row["best_fitness"] for row in history],
            label="Najlepsze",
        )
        axis.plot(
            generations,
            [row["mean_fitness"] for row in history],
            label="Średnie",
        )
        axis.set_title(PHASE_NAMES[phase])
        axis.set_xlabel("Generacja")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Dopasowanie")
    axes[-1].legend(frameon=False)
    figure.tight_layout()
    path = paths.figures / "asteroids_trening.png"
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return path
