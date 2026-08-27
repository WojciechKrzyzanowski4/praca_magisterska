from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pygame

from ..snake_constants import config as game_config
from ..snake_game import Game
from ..snake_policy import FullyConnectedPolicy
from .output import OutputPaths
from .setup import StaticConfig


@dataclass(frozen=True)
class EvaluationResult:
    rows: list[dict[str, int | float]]
    timeline: list[dict[str, int]]
    representative: dict[str, int]


def build_policy(genome: np.ndarray, hidden_size: int) -> FullyConnectedPolicy:
    probe = Game(headless=True, rng=random.Random(0))
    policy = FullyConnectedPolicy(probe.get_state().size, hidden_size=hidden_size)
    if genome.size != policy.total_param_count:
        raise ValueError(
            f"Genom ma {genome.size} parametrów, oczekiwano {policy.total_param_count}"
        )
    return policy


def choose_action(
    policy: FullyConnectedPolicy,
    genome: np.ndarray,
    state: np.ndarray,
) -> dict[str, bool]:
    return policy.output_to_control(policy.forward(genome, state))


def render_game(game: Game, path: Path) -> None:
    surface = pygame.Surface((game_config.WIDTH, game_config.HEIGHT))
    surface.fill(game_config.COLOR_BG)
    if game.food.pos is not None:
        game.food.draw(surface)
    game.snake.draw(surface)
    font = pygame.font.SysFont("consolas", 20)
    hud = font.render(f"Score: {game.score}", True, game_config.COLOR_FG)
    surface.blit(hud, (12, 10))
    pygame.image.save(surface, str(path))


def evaluate_seeds(
    genome: np.ndarray,
    policy: FullyConnectedPolicy,
    config: StaticConfig,
) -> list[dict[str, int | float]]:
    rows: list[dict[str, int | float]] = []
    for seed in config.evaluation_seeds:
        game = Game(headless=True, rng=random.Random(seed))
        steps = 0
        reward = 0.0
        while not game.finished and steps < config.eval_max_steps:
            control = choose_action(policy, genome, game.get_state())
            step_reward, _ = game.step(control)
            reward += step_reward
            steps += 1
        rows.append(
            {
                "seed": seed,
                "score": game.score,
                "steps": steps,
                "length": len(game.snake.body),
                "reward": reward,
            }
        )
    return rows


def evaluate_representative_run(
    genome: np.ndarray,
    policy: FullyConnectedPolicy,
    config: StaticConfig,
    paths: OutputPaths,
) -> tuple[list[dict[str, int]], dict[str, int]]:
    game = Game(headless=True, rng=random.Random(config.representative_seed))
    timeline: list[dict[str, int]] = []
    captures = set(config.capture_scores)
    saved: set[int] = set()
    previous_score = -1
    steps = 0

    while not game.finished and steps < config.eval_max_steps:
        if game.score in captures and game.score not in saved:
            render_game(game, paths.screenshots / f"snake_score_{game.score:03d}.png")
            saved.add(game.score)
        control = choose_action(policy, genome, game.get_state())
        game.step(control)
        steps += 1
        if game.score != previous_score:
            timeline.append(
                {
                    "step": steps,
                    "score": game.score,
                    "length": len(game.snake.body),
                }
            )
            previous_score = game.score

    missing = captures.difference(saved)
    if missing:
        raise RuntimeError(f"Przebieg reprezentatywny nie osiągnął wyników: {sorted(missing)}")

    render_game(game, paths.screenshots / "snake_final_frame.png")
    final = {
        "score": game.score,
        "steps": steps,
        "length": len(game.snake.body),
    }
    return timeline, final


def evaluate_genome(
    genome: np.ndarray,
    config: StaticConfig,
    paths: OutputPaths,
) -> EvaluationResult:
    policy = build_policy(genome, config.hidden_size)
    rows = evaluate_seeds(genome, policy, config)
    timeline, representative = evaluate_representative_run(
        genome, policy, config, paths
    )
    return EvaluationResult(rows, timeline, representative)
