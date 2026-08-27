from __future__ import annotations

import math
import os
import random as pyrand
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from deap import base, creator, tools

from .asteroid_constants import config
from .asteroid_game import Game
from .asteroid_policy import FullyConnectedPolicy


@dataclass
class PhaseSpec:
    name: str
    step_phase: int
    generations: int
    eval_episodes: int = 0
    frame_skip: int = 3


def default_phase_spec(phase: int) -> PhaseSpec:
    if phase == 1:
        return PhaseSpec("P1-DODGE", 1, 150, frame_skip=3)
    if phase == 2:
        return PhaseSpec("P2-AIM",   2, 150, frame_skip=2)
    if phase == 3:
        return PhaseSpec("P3-KILL",  3, 50, frame_skip=2)
    raise ValueError("phase must be 1, 2, or 3")


@dataclass
class DEAPConfig:
    population_size: int = 80
    generations: int = 50
    cx_prob: float = 0.65
    mut_prob: float = 0.03
    mutation_sigma: float = 0.20
    mutation_sigma_min: float = 0.05
    tournament_size: int = 3
    eval_max_steps: int = 60 * 120
    random_seed: int = 123
    elites: int = 2
    reevaluate_topk: int = 5
    training_scenarios: tuple[int, ...] = tuple(config.TRAIN_SCENARIOS)


@dataclass
class EvalResult:
    reward: float
    steps: int
    seconds: float
    aim_alignment: float = 0.0
    mean_kills: float = 0.0
    mean_shots: float = 0.0
    shot_efficiency: float = 0.0
    mean_waves: float = 0.0
    win_rate: float = 0.0


def _step_method(game: Game, phase_idx: int):
    return (game.step_phase1 if phase_idx == 1
            else game.step_phase2 if phase_idx == 2
            else game.step_phase3)


def _make_policy(policy: FullyConnectedPolicy, flat_params: np.ndarray):
    flat = np.asarray(flat_params, dtype=np.float32)
    def policy_fn(state_vec: np.ndarray) -> dict:
        y = policy.forward(flat, state_vec)
        return policy.output_to_control(y)
    return policy_fn


def eval_policy_on_scenarios(
    flat_params: np.ndarray,
    phase: PhaseSpec,
    cfg: DEAPConfig,
    *,
    hidden_size: int,
    lives: int = 1,
    scenarios: List[int] = config.TRAIN_SCENARIOS,
    dt: float = 1.0 / 60.0,
) -> EvalResult:
    """Mean fitness over a fixed set of scenario offsets (deterministic)."""
    probe = Game(lives=lives, headless=True)
    state_size = probe.get_state().size
    policy = FullyConnectedPolicy(state_size, hidden_size=hidden_size)
    assert flat_params.size == policy.total_param_count, \
        f"Genome size {flat_params.size} != expected {policy.total_param_count} (state={state_size}, hidden={hidden_size})"
    pfn = _make_policy(policy, flat_params)

    total_R, total_steps = 0.0, 0
    total_aim_alignment = 0.0
    total_kills, total_shots = 0, 0
    total_waves, total_wins = 0, 0
    repeat = max(1, phase.frame_skip)

    for scen in scenarios:
        game = Game(lives=lives, headless=True)
        game.set_scenario_offset(scen)
        game.reset()
        step_fn = _step_method(game, phase.step_phase)
        steps, R = 0, 0.0
        while not game.finished and steps < cfg.eval_max_steps:
            s = game.get_state()
            ctrl = pfn(s) or {}
            for _ in range(repeat):
                r, _ = step_fn(ctrl, dt)
                R += (r or 0.0)
                steps += 1
                if game.finished or steps >= cfg.eval_max_steps:
                    break

        total_R += R
        total_steps += steps
        if phase.step_phase in (2, 3):
            total_aim_alignment += game.phase2_mean_alignment()
        if phase.step_phase == 3:
            total_kills += game.kills
            total_shots += game.bullets_fired
            total_waves += game.waves_cleared
            total_wins += int(game.finished and game.lives > 0)

    n = max(1, len(scenarios))
    mean_steps = int(round(total_steps / n))
    return EvalResult(
        reward=total_R / n,
        steps=mean_steps,
        seconds=mean_steps * dt,
        aim_alignment=total_aim_alignment / n,
        mean_kills=total_kills / n,
        mean_shots=total_shots / n,
        shot_efficiency=total_kills / max(1, total_shots),
        mean_waves=total_waves / n,
        win_rate=total_wins / n,
    )


def eval_holdout_report(
    flat_params: np.ndarray,
    phase: PhaseSpec,
    cfg: DEAPConfig,
    *,
    hidden_size: int,
) -> List[Tuple[int, float, float]]:
    """Return [(scenario, seconds, reward)] for quick generalization checks."""
    results = []
    for scen in config.HOLDOUT_SCENARIOS:
        res = eval_policy_on_scenarios(
            flat_params, phase, cfg, hidden_size=hidden_size,
            lives=1, scenarios=[scen]
        )
        results.append((scen, res.seconds, res.reward))
    return results



class DEAPCurriculum:
    def __init__(self, cfg: DEAPConfig, hidden_size: int = 48):
        self.cfg = cfg
        self.hidden_size = hidden_size
        self.current_mutation_sigma = cfg.mutation_sigma
        self.history: list[dict[str, int | float]] = []

        # Determinism: NumPy + Python random (DEAP uses python's random)
        self.rng = np.random.default_rng(cfg.random_seed)
        pyrand.seed(cfg.random_seed)

        # Probe sizes
        probe = Game(lives=1, headless=True)
        state_size = probe.get_state().size

        self.policy = FullyConnectedPolicy(state_size, hidden_size=hidden_size)
        self.param_dim = self.policy.total_param_count

        # DEAP types (guard against re-creation)
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

        # toolbox
        self.toolbox = base.Toolbox()
        init_scale = math.sqrt(6.0 / (self.policy.input_size + self.policy.hidden_size))
        self.toolbox.register("attr_float", self.rng.uniform, -init_scale, init_scale)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                              self.toolbox.attr_float, n=self.param_dim)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", self._crossover_blend)
        self.toolbox.register("mutate", self._mutate_gaussian)
        self.toolbox.register("select", tools.selTournament, tournsize=self.cfg.tournament_size)

        # Evaluator used by DEAP
        def eval_wrapper(ind):
            res = eval_policy_on_scenarios(
                ind, default_phase_spec(1),  # phase is passed on evolve_phase; placeholder here
                hidden_size=self.hidden_size, cfg=self.cfg
            )
            return (res.reward,)
        # We'll re-register with the real PhaseSpec inside evolve_phase
        self.toolbox.register("evaluate", eval_wrapper)

    # --- genetic ops ---
    def _crossover_blend(self, ind1, ind2, alpha=0.20):
        if self.rng.random() < self.cfg.cx_prob:
            a = self.rng.uniform(-alpha, 1 + alpha, size=ind1.shape)
            c1 = a * ind1 + (1 - a) * ind2
            c2 = a * ind2 + (1 - a) * ind1
            ind1[:] = c1
            ind2[:] = c2
        return ind1, ind2

    def _mutate_gaussian(self, individual):
        mask = self.rng.random(individual.shape) < self.cfg.mut_prob
        noise = self.rng.normal(0.0, self.current_mutation_sigma, size=individual.shape)
        individual[:] = individual + mask * noise
        return (individual,)

    def _seed_population_from_genome(self, genome: np.ndarray, n: int) -> List[np.ndarray]:
        """Keep the source genome intact and explore with jittered copies."""
        if n <= 0:
            return []

        pop = [creator.Individual(np.array(genome, copy=True))]
        jitter_sigma = max(1e-4, self.cfg.mutation_sigma * 0.5)
        for _ in range(n - 1):
            ind = creator.Individual(np.array(genome, copy=True))
            noise = self.rng.normal(0.0, jitter_sigma, size=ind.shape)
            ind[:] = ind + noise
            pop.append(ind)
        return pop

    # --- main evolution ---
    def evolve_phase(
        self,
        phase: PhaseSpec,
        seed_genome: Optional[np.ndarray] = None,
        checkpoint_path: Optional[str] = None,
    ) -> Tuple[List[np.ndarray], float, np.ndarray]:
        """Run GA on a phase; optionally start from a provided genome."""

        # Rebind evaluator to use this phase's settings
        def eval_wrapper(ind):
            res = eval_policy_on_scenarios(
                ind, phase, self.cfg,
                hidden_size=self.hidden_size, lives=1, scenarios=list(self.cfg.training_scenarios)
            )
            return (res.reward,)
        self.toolbox.register("evaluate", eval_wrapper)

        # initial population
        pop = (self._seed_population_from_genome(seed_genome, self.cfg.population_size)
               if seed_genome is not None else
               self.toolbox.population(n=self.cfg.population_size))

        # initial fitness
        for ind, fit in zip(pop, map(self.toolbox.evaluate, pop)):
            ind.fitness.values = fit

        # GA loop
        for gen in range(phase.generations):
            # Elitism
            elites = tools.selBest(pop, self.cfg.elites)

            # Select & clone the rest
            offspring = self.toolbox.select(pop, len(pop) - self.cfg.elites)
            offspring = list(map(self.toolbox.clone, offspring))

            # Crossover
            for i in range(0, len(offspring), 2):
                if i + 1 < len(offspring):
                    self.toolbox.mate(offspring[i], offspring[i + 1])
                    if hasattr(offspring[i].fitness, "values"):
                        del offspring[i].fitness.values
                    if hasattr(offspring[i + 1].fitness, "values"):
                        del offspring[i + 1].fitness.values

            # Mutation (simple linear anneal for sigma)
            sigma0 = self.cfg.mutation_sigma
            sigma_min = self.cfg.mutation_sigma_min
            frac = gen / max(1, phase.generations - 1)
            self.current_mutation_sigma = sigma_min + (sigma0 - sigma_min) * (1.0 - frac)

            for mutant in offspring:
                self.toolbox.mutate(mutant)
                if hasattr(mutant.fitness, "values"):
                    del mutant.fitness.values

            # Re-evaluate invalids
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            for ind, fit in zip(invalid, map(self.toolbox.evaluate, invalid)):
                ind.fitness.values = fit

            pop[:] = elites + offspring

            # Logging with survival time for the displayed best (mean over scenarios)
            best = tools.selBest(pop, 1)[0]
            mean_fit = float(np.mean([ind.fitness.values[0] for ind in pop]))
            best_res = eval_policy_on_scenarios(
                best,
                phase,
                self.cfg,
                hidden_size=self.hidden_size,
                lives=1,
                scenarios=list(self.cfg.training_scenarios),
            )
            self.history.append(
                {
                    "phase": phase.step_phase,
                    "generation": gen,
                    "best_fitness": float(best.fitness.values[0]),
                    "mean_fitness": mean_fit,
                    "seconds": float(best_res.seconds),
                    "aim_alignment": float(best_res.aim_alignment),
                    "mean_kills": float(best_res.mean_kills),
                    "mean_shots": float(best_res.mean_shots),
                    "shot_efficiency": float(best_res.shot_efficiency),
                    "mean_waves": float(best_res.mean_waves),
                    "win_rate": float(best_res.win_rate),
                    "mutation_sigma": float(self.current_mutation_sigma),
                }
            )
            if checkpoint_path:
                checkpoint_dir = os.path.dirname(checkpoint_path)
                if checkpoint_dir:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                np.save(checkpoint_path, np.asarray(best, dtype=np.float32))
            aim_text = f" | Aim={best_res.aim_alignment:.3f}" if phase.step_phase in (2, 3) else ""
            shooting_text = ""
            if phase.step_phase == 3:
                shooting_text = (
                    f" | Kills={best_res.mean_kills:.2f}"
                    f" | Shots={best_res.mean_shots:.2f}"
                    f" | Eff={best_res.shot_efficiency:.1%}"
                    f" | Waves={best_res.mean_waves:.2f}"
                    f" | Wins={best_res.win_rate:.1%}"
                )
            print(f"[{phase.name}] Gen {gen:03d} | Best={best.fitness.values[0]:.2f} "
                  f"| Mean={mean_fit:.2f} | Survived={best_res.seconds:.2f}s "
                  f"({best_res.steps} steps){aim_text}{shooting_text}", flush=True)

        # Final top-K re-evaluation (fresh resets) to pick a truly-best genome
        topK = tools.selBest(pop, min(self.cfg.reevaluate_topk, len(pop)))
        scores = []
        for ind in topK:
            res = eval_policy_on_scenarios(
                ind,
                phase,
                self.cfg,
                hidden_size=self.hidden_size,
                lives=1,
                scenarios=list(self.cfg.training_scenarios),
            )
            scores.append(res.reward)
        j = int(np.argmax(scores))
        true_best = np.asarray(topK[j], dtype=np.float32)
        true_best_f = float(scores[j])
        return pop, true_best_f, true_best


def load_genome(path: str) -> np.ndarray:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return np.load(path)


def save_genome(path: str, genome: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, genome)
