from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

from .asteroid_constants import config
from .asteroid_game import Game
from .asteroid_policy import FullyConnectedPolicy
from .asteroid_trainer import DEAPConfig, DEAPCurriculum, default_phase_spec
from .asteroid_utils import ensure_output_dir, phase_step_method


def train_single_phase(
    phase: int,
    cfg: Optional[DEAPConfig] = None,
    hidden_size: int = 48,
    seed_path: Optional[str] = None,
    out_path: Optional[str] = None,
    override_generations: Optional[int] = None,
) -> str:
    """
    Train (or continue training) a single phase. If seed_path is not provided,
    phases 2/3 auto-chain from previous phase files in config.PHASE_FILE.
    """
    ensure_output_dir()
    cfg = cfg or DEAPConfig()
    spec = default_phase_spec(phase)
    if override_generations is not None:
        spec.generations = int(override_generations)

    if seed_path is None:
        if phase == 2:
            seed_path = config.PHASE_FILE[1]
        elif phase == 3:
            seed_path = config.PHASE_FILE[2]

    seed_genome = None
    if seed_path:
        if not os.path.isfile(seed_path):
            raise FileNotFoundError(
                f"Requested seed for phase {phase} not found: {seed_path}. "
                f"Run phase {phase - 1} first or provide --seed explicitly."
            )
        seed_genome = np.load(seed_path)

    out_path = out_path or config.PHASE_FILE[phase]
    checkpoint_root, _ = os.path.splitext(out_path)
    checkpoint_path = f"{checkpoint_root}_checkpoint.npy"

    runner = DEAPCurriculum(cfg, hidden_size=hidden_size)
    _, best_f, best_genome = runner.evolve_phase(
        spec,
        seed_genome=seed_genome,
        checkpoint_path=checkpoint_path,
    )
    print(f"[SAVE] Phase {phase} finished. Best fitness = {best_f:.3f}")

    np.save(out_path, best_genome)
    print(f"[SAVE] Wrote best genome to: {out_path}")
    return out_path


def train_model(
    cfg: Optional[DEAPConfig] = None,
    hidden_size: int = 48,
) -> None:
    """Train 1 → 2 → 3 sequentially, auto-chaining best genomes."""
    cfg = cfg or DEAPConfig()
    for p in (1, 2, 3):
        train_single_phase(p, cfg=cfg, hidden_size=hidden_size)


def _load_genome_or_fail(path: str) -> np.ndarray:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Genome not found at {path}. Train the phase first.")
    return np.load(path)


def visualize_phase(
    phase: int,
    genome_path: Optional[str] = None,
    hidden_size: int = 48,
    fps: int = config.FPS,
    scenario_cycle: Optional[List[int]] = [2],
) -> None:
    genome_path = genome_path or config.PHASE_FILE[phase]
    flat = np.load(genome_path)

    spec = default_phase_spec(phase)
    dt = 1.0 / config.FPS
    repeat = max(1, spec.frame_skip)

    for scen in scenario_cycle:
        game = Game(lives=1)
        game.headless = False
        game.set_scenario_offset(int(scen))
        game.reset()

        print(f"Now showing scenario {scen}.")

        state_size = game.get_state().size
        policy = FullyConnectedPolicy(state_size, hidden_size=hidden_size)
        assert flat.size == policy.total_param_count, \
            f"Genome size {flat.size} != expected {policy.total_param_count} (state={state_size}, hidden={hidden_size})"

        step_fn = phase_step_method(game, phase)
        total_reward, steps = 0.0, 0

        while not game.finished:
            s = game.get_state()
            y = policy.forward(flat, s)
            control = policy.output_to_control(y)
            for _ in range(repeat):
                r, _ = step_fn(control, dt)
                total_reward += (r or 0.0)
                steps += 1
                if game.finished:
                    break

            game._draw()
            game._handle_events()
            game.clock.tick(fps)

        print(f"[VIS scen={scen}] reward={total_reward:.2f} | survived={steps*dt:.2f}s ({steps} steps))")


def play_game() -> None:
    """Run the interactive game with keyboard controls"""
    game = Game(lives=3, headless=False)
    game.run()
