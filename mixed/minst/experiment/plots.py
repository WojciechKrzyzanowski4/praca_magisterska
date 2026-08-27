from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .output import OutputPaths


METHODS = ("GD", "EVO-random", "GD+GD", "GD+EVO")
COLORS = {
    "GD": "#f59e0b",
    "EVO-random": "#8b5cf6",
    "GD+GD": "#ef4444",
    "GD+EVO": "#06b6d4",
}


def plot_benchmark(paths: OutputPaths) -> list[Path]:
    summary = pd.read_csv(paths.root / "summary.csv")
    traces = pd.read_csv(paths.root / "traces.csv")
    robustness = pd.read_csv(paths.root / "robustness.csv")
    classes = pd.read_csv(paths.root / "class_accuracy.csv")
    plt.style.use("seaborn-v0_8-whitegrid")
    figures: list[Path] = []

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for method in METHODS:
        subset = traces[traces.method == method]
        if subset.empty:
            continue
        grouped = subset.groupby("progress")
        x = np.asarray(sorted(subset.progress.unique()))
        axes[0].plot(x, grouped.loss.median().reindex(x), marker="o", color=COLORS[method], label=method)
        axes[1].plot(x, grouped.validation_accuracy.median().reindex(x), marker="o", color=COLORS[method], label=method)
    axes[0].set(xlabel="Postęp etapu", ylabel="Cross-entropy")
    axes[1].set(xlabel="Postęp etapu", ylabel="Dokładność", ylim=(0, 1))
    axes[1].legend(ncol=2, fontsize=8)
    figure.tight_layout()
    path = paths.figures / "mnist_wyniki_przebieg.png"
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    figures.append(path)

    figure, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    positions = np.arange(len(METHODS))
    for axis, metric, title in zip(
        axes,
        ("test_accuracy", "test_loss", "training_time"),
        ("Dokładność testowa", "Cross-entropy testowe", "Czas optymalizacji [s]"),
    ):
        values = [summary.loc[summary.method == method, metric].to_numpy() for method in METHODS]
        means = [value.mean() for value in values]
        errors = [value.std(ddof=1) if len(value) > 1 else 0 for value in values]
        axis.bar(positions, means, yerr=errors, capsize=4, color=[COLORS[m] for m in METHODS])
        axis.set_xticks(positions, METHODS, rotation=20, ha="right")
        axis.set_title(title)
        if metric == "test_accuracy":
            axis.set_ylim(0, 1)
    figure.tight_layout()
    path = paths.figures / "mnist_wyniki_porownanie_metod.png"
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    figures.append(path)

    figure, axis = plt.subplots(figsize=(10, 4.8))
    width = 0.19
    digits = np.arange(10)
    for index, method in enumerate(METHODS):
        values = classes[classes.method == method].groupby("digit").accuracy.mean()
        axis.bar(digits + (index - 1.5) * width, values.reindex(digits), width, label=method, color=COLORS[method])
    axis.set(xlabel="Cyfra", ylabel="Dokładność", xticks=digits, ylim=(0, 1))
    axis.legend(ncol=4, fontsize=8)
    figure.tight_layout()
    path = paths.figures / "mnist_wyniki_klasy.png"
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    figures.append(path)

    figure, axis = plt.subplots(figsize=(8.5, 4.5))
    for method in METHODS:
        values = robustness[robustness.method == method].groupby("noise_std").accuracy
        x = np.asarray(sorted(robustness.noise_std.unique()))
        axis.errorbar(
            x,
            values.mean().reindex(x),
            yerr=values.std().reindex(x).fillna(0),
            marker="o",
            capsize=4,
            linewidth=2,
            color=COLORS[method],
            label=method,
        )
    axis.set(xlabel="Odchylenie standardowe szumu", ylabel="Dokładność", ylim=(0, 1))
    axis.legend(ncol=2)
    figure.tight_layout()
    path = paths.figures / "mnist_wyniki_szum.png"
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    figures.append(path)

    representative = int(summary.seed.unique()[0])
    matrices = np.load(paths.root / f"confusion_seed_{representative}.npz")
    figure, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharex=True, sharey=True)
    image = None
    for axis, method in zip(axes, METHODS):
        image = axis.imshow(matrices[method], cmap="Blues", vmin=0)
        axis.set(title=method, xlabel="Predykcja", ylabel="Etykieta")
        axis.set_xticks(range(0, 10, 2))
        axis.set_yticks(range(0, 10, 2))
    figure.colorbar(image, ax=axes, shrink=0.78)
    figure.subplots_adjust(left=0.05, right=0.95, bottom=0.14, top=0.9, wspace=0.22)
    path = paths.figures / "mnist_wyniki_macierze.png"
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    figures.append(path)
    return figures


def plot_extended(paths: OutputPaths) -> list[Path]:
    summary = pd.read_csv(paths.root / "summary.csv")
    methods = summary.method.tolist()
    figure, axes = plt.subplots(1, 2, figsize=(9.8, 4.4))
    colors = [COLORS.get(method, "#64748b") for method in methods]
    axes[0].bar(methods, summary.monitor_accuracy, color=colors)
    axes[0].set_ylabel("Dokładność na zbiorze monitorującym")
    axes[0].set_ylim(0, 1)
    axes[1].bar(methods, summary.monitor_loss, color=colors)
    axes[1].set_ylabel("Cross-entropy")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    path = paths.figures / "mnist_wyniki_przebieg_rozszerzony.png"
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return [path]
