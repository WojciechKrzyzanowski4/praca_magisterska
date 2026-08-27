from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .output import OutputPaths
from .setup import StaticConfig


def make_contact_sheet(paths: OutputPaths, config: StaticConfig) -> None:
    screenshots = [
        paths.screenshots / f"snake_score_{score:03d}.png"
        for score in config.capture_scores
    ]
    labels = ["Start"] + [f"{score} punktów" for score in config.capture_scores[1:]]
    images = [Image.open(path).convert("RGB") for path in screenshots]
    width, height = images[0].size
    margin, label_height = 28, 34
    rows = (len(images) + 1) // 2
    canvas = Image.new(
        "RGB",
        (width * 2 + margin * 3, height * rows + label_height * rows + margin * (rows + 1)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=18)
    for index, (source, label) in enumerate(zip(images, labels)):
        row, column = divmod(index, 2)
        x = margin + column * (width + margin)
        y = margin + row * (height + label_height + margin)
        draw.text((x + width / 2, y), label, fill="black", font=font, anchor="ma")
        canvas.paste(source, (x, y + label_height))
    canvas.save(paths.figures / "snake_wyniki_klatki.png", quality=95)


def make_evaluation_plots(
    rows: list[dict[str, int | float]],
    timeline: list[dict[str, int]],
    output_directory: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    seeds = [int(row["seed"]) for row in rows]
    scores = [int(row["score"]) for row in rows]
    mean_score = float(np.mean(scores))

    figure, axis = plt.subplots(figsize=(11.5, 5.2))
    bars = axis.bar(seeds, scores, color="#16a34a")
    axis.axhline(
        mean_score,
        color="#2563eb",
        linewidth=2,
        label=f"Średnia: {mean_score:.1f}",
    )
    axis.set_xlabel("Przebieg")
    axis.set_ylabel("Zdobyte punkty")
    axis.set_xticks(seeds)
    axis.legend(frameon=False)
    axis.bar_label(bars, fontsize=7, padding=2)
    figure.tight_layout()
    figure.savefig(output_directory / "snake_wyniki_przebiegi.png", dpi=220, bbox_inches="tight")
    plt.close(figure)

    x = [row["step"] for row in timeline]
    y = [row["score"] for row in timeline]
    figure, axis = plt.subplots(figsize=(11.5, 5.2))
    axis.step(x, y, where="post", color="#16a34a", linewidth=2)
    axis.fill_between(x, y, step="post", alpha=0.12, color="#16a34a")
    axis.set_xlabel("Krok rozgrywki")
    axis.set_ylabel("Wynik")
    axis.set_xlim(left=0)
    axis.set_ylim(bottom=0)
    figure.tight_layout()
    figure.savefig(output_directory / "snake_wyniki_przebieg.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def make_training_plot(
    history: list[dict[str, int | float]],
    output_directory: Path,
) -> None:
    if not history:
        return
    generations = [row["generation"] for row in history]
    figure, axis = plt.subplots(figsize=(11.5, 5.2))
    axis.plot(generations, [row["best_fitness"] for row in history], label="Najlepsze dopasowanie")
    axis.plot(generations, [row["mean_fitness"] for row in history], label="Średnie dopasowanie")
    axis.set_xlabel("Generacja")
    axis.set_ylabel("Dopasowanie")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_directory / "snake_trening.png", dpi=220, bbox_inches="tight")
    plt.close(figure)
