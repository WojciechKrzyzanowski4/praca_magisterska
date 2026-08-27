from __future__ import annotations

import os
from dataclasses import asdict, dataclass

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame

from ..asteroid_constants import config as game_config
from ..asteroid_game import Game
from ..asteroid_policy import FullyConnectedPolicy
from .output import OutputPaths
from .setup import StaticConfig


@dataclass(frozen=True)
class ScenarioResult:
    phase: int
    scenario: int
    seconds: float
    won: bool
    waves: int
    kills: int
    shots: int
    efficiency: float
    mean_alignment: float
    remaining_asteroids: int
    lives: int


@dataclass(frozen=True)
class EvaluationResult:
    scenarios: list[ScenarioResult]
    summary: list[dict[str, int | float]]
    representative: ScenarioResult
    representative_timeline: list[dict[str, int | float]]
    screenshots: dict[str, str]


def build_policy(genome: np.ndarray, hidden_size: int) -> FullyConnectedPolicy:
    probe = Game(lives=1, headless=True)
    policy = FullyConnectedPolicy(probe.get_state().size, hidden_size=hidden_size)
    if genome.size != policy.total_param_count:
        raise ValueError(
            f"Genom ma {genome.size} parametrów, oczekiwano {policy.total_param_count}"
        )
    return policy


def simulate(
    genome: np.ndarray,
    phase: int,
    scenario: int,
    config: StaticConfig,
    *,
    timeline: bool = False,
) -> tuple[ScenarioResult, list[dict[str, int | float]]]:
    game = Game(lives=config.lives, headless=True)
    game.set_scenario_offset(scenario)
    game.reset()
    policy = build_policy(genome, config.hidden_size)
    step_method = getattr(game, f"step_phase{phase}")
    repeat = config.phase(phase).frame_skip
    max_steps = config.max_seconds_by_phase[phase] * config.fps
    steps = 0
    alignment_sum = 0.0
    alignment_steps = 0
    records: list[dict[str, int | float]] = []

    while not game.finished and steps < max_steps:
        control = policy.output_to_control(policy.forward(genome, game.get_state()))
        for _ in range(repeat):
            step_method(control, 1.0 / config.fps)
            steps += 1
            alignment_sum += game._nearest_asteroid_alignment()
            alignment_steps += 1
            if timeline and (steps == 1 or steps % config.fps == 0 or game.finished):
                records.append(
                    {
                        "second": steps / config.fps,
                        "waves": game.waves_cleared,
                        "kills": game.kills,
                        "shots": game.bullets_fired,
                        "remaining_asteroids": len(game.asteroids),
                        "ship_x": float(game.ship.pos.x),
                        "ship_y": float(game.ship.pos.y),
                        "lives": game.lives,
                    }
                )
            if game.finished or steps >= max_steps:
                break

    result = ScenarioResult(
        phase=phase,
        scenario=scenario,
        seconds=steps / config.fps,
        won=bool(game.finished and game.lives > 0),
        waves=game.waves_cleared,
        kills=game.kills,
        shots=game.bullets_fired,
        efficiency=game.kills / max(1, game.bullets_fired),
        mean_alignment=alignment_sum / max(1, alignment_steps),
        remaining_asteroids=len(game.asteroids),
        lives=game.lives,
    )
    return result, records


def summarize(results: list[ScenarioResult]) -> list[dict[str, int | float]]:
    rows: list[dict[str, int | float]] = []
    for phase in (1, 2, 3):
        phase_results = [result for result in results if result.phase == phase]
        rows.append(
            {
                "phase": phase,
                "mean_seconds": float(np.mean([row.seconds for row in phase_results])),
                "std_seconds": float(np.std([row.seconds for row in phase_results])),
                "mean_alignment": float(np.mean([row.mean_alignment for row in phase_results])),
                "mean_kills": float(np.mean([row.kills for row in phase_results])),
                "mean_shots": float(np.mean([row.shots for row in phase_results])),
                "efficiency": sum(row.kills for row in phase_results)
                / max(1, sum(row.shots for row in phase_results)),
                "mean_waves": float(np.mean([row.waves for row in phase_results])),
                "wins": int(sum(row.won for row in phase_results)),
            }
        )
    return rows


def capture_screenshots(
    genomes: dict[int, np.ndarray],
    results: list[ScenarioResult],
    config: StaticConfig,
    paths: OutputPaths,
) -> dict[str, str]:
    outputs: dict[str, str] = {}

    def setup_game(phase: int, scenario: int):
        game = Game(lives=config.lives, headless=False)
        game.set_scenario_offset(scenario)
        game.reset()
        policy = build_policy(genomes[phase], config.hidden_size)
        step_method = getattr(game, f"step_phase{phase}")
        return game, policy, step_method, config.phase(phase).frame_skip

    for phase in (1, 2):
        phase_results = [result for result in results if result.phase == phase]
        scenario = max(phase_results, key=lambda result: result.seconds).scenario
        game, policy, step_method, repeat = setup_game(phase, scenario)
        best_value = -float("inf")
        best_surface = None
        steps = 0
        max_steps = 60 * config.fps
        while not game.finished and steps < max_steps:
            control = policy.output_to_control(
                policy.forward(genomes[phase], game.get_state())
            )
            for _ in range(repeat):
                step_method(control, 1.0 / config.fps)
                steps += 1
                if phase == 1:
                    value = game._nearest_distance() if steps > 2 * config.fps else -1.0
                else:
                    value = (
                        game._nearest_asteroid_alignment()
                        if steps > 2 * config.fps
                        else -1.0
                    )
                if value > best_value:
                    best_value = value
                    game._draw()
                    best_surface = game.screen.copy()
                if game.finished or steps >= max_steps:
                    break
        path = paths.screenshots / f"asteroids_faza{phase}.png"
        if best_surface is None:
            game._draw()
            best_surface = game.screen.copy()
        pygame.image.save(best_surface, path)
        outputs[f"phase{phase}"] = str(path)

    game, policy, step_method, repeat = setup_game(
        3, config.representative_scenario
    )
    action_surface = None
    steps = 0
    max_steps = config.max_seconds_by_phase[3] * config.fps
    while not game.finished and steps < max_steps:
        control = policy.output_to_control(policy.forward(genomes[3], game.get_state()))
        for _ in range(repeat):
            step_method(control, 1.0 / config.fps)
            steps += 1
            if action_surface is None and game.kills >= 250 and len(game.bullets) >= 3:
                game._draw()
                action_surface = game.screen.copy()
            if game.finished or steps >= max_steps:
                break

    if action_surface is None:
        game._draw()
        action_surface = game.screen.copy()
    action_path = paths.screenshots / "asteroids_faza3_walka.png"
    pygame.image.save(action_surface, action_path)
    outputs["phase3_action"] = str(action_path)

    game._draw()
    final_path = paths.screenshots / "asteroids_faza3_wygrana.png"
    pygame.image.save(game.screen, final_path)
    outputs["phase3_victory"] = str(final_path)
    return outputs


def evaluate_genomes(
    genomes: dict[int, np.ndarray],
    config: StaticConfig,
    paths: OutputPaths,
) -> EvaluationResult:
    results: list[ScenarioResult] = []
    for phase in (1, 2, 3):
        for scenario in config.evaluation_scenarios:
            result, _ = simulate(genomes[phase], phase, scenario, config)
            results.append(result)

    representative, timeline = simulate(
        genomes[3],
        3,
        config.representative_scenario,
        config,
        timeline=True,
    )
    screenshots = capture_screenshots(genomes, results, config, paths)
    return EvaluationResult(
        scenarios=results,
        summary=summarize(results),
        representative=representative,
        representative_timeline=timeline,
        screenshots=screenshots,
    )


def scenario_rows(result: EvaluationResult) -> list[dict[str, object]]:
    return [asdict(row) for row in result.scenarios]
