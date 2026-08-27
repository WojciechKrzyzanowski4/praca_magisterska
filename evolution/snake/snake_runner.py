from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np

from .snake_constants import config
from .snake_game import Game
from .snake_policy import FullyConnectedPolicy
from .snake_trainer import DEAPConfig, DEAPTrainer
from .snake_utils import ensure_output_dir


def train_model(
    cfg: Optional[DEAPConfig] = None,
    hidden_size: int = 48,
    seed_path: Optional[str] = None,
    out_path: Optional[str] = None,
) -> str:
    ensure_output_dir()
    cfg = cfg or DEAPConfig()

    seed_genome = None
    if seed_path is not None:
        if not os.path.isfile(seed_path):
            raise FileNotFoundError(f"Requested seed genome not found: {seed_path}")
        seed_genome = np.load(seed_path)

    trainer = DEAPTrainer(cfg, hidden_size=hidden_size)
    _, best_fitness, best_genome = trainer.evolve(seed_genome=seed_genome)
    out_path = out_path or config.MODEL_FILE
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.save(out_path, best_genome)
    print(f"[SAVE] Training finished. Best fitness = {best_fitness:.3f}")
    print(f"[SAVE] Wrote best genome to: {out_path}")
    return out_path


def visualize_result(
    genome_path: Optional[str] = None,
    hidden_size: int = 48,
    fps: int = config.FPS,
    max_steps: int = config.VIS_MAX_STEPS,
    seed: Optional[int] = None,
) -> None:
    genome_path = genome_path or config.MODEL_FILE
    if not os.path.isfile(genome_path):
        raise FileNotFoundError(f"Genome not found at {genome_path}. Train the model first.")

    flat = np.load(genome_path)
    game = Game(headless=False, rng=random.Random(seed) if seed is not None else None)
    state_size = game.get_state().size
    policy = FullyConnectedPolicy(state_size, hidden_size=hidden_size)
    assert flat.size == policy.total_param_count, \
        f"Genome size {flat.size} != expected {policy.total_param_count} (state={state_size}, hidden={hidden_size})"

    total_reward = 0.0
    steps = 0

    while not game.finished and steps < max_steps:
        game._handle_events()
        state_vec = game.get_state()
        logits = policy.forward(flat, state_vec)
        control = policy.output_to_control(logits)
        reward, _ = game.step(control)
        total_reward += float(reward or 0.0)
        steps += 1
        game._draw()
        game.clock.tick(fps)

    game._draw()
    print(f"[VIS] reward={total_reward:.3f} | score={game.score} | steps={steps}")

def play_game() -> None:
    game = Game(headless=False)
    game.run()
