from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from deap import base, creator, tools

from .snake_constants import config
from .snake_game import Game
from .snake_policy import FullyConnectedPolicy


@dataclass
class DEAPConfig:
    population_size: int = 40
    generations: int = 100
    cx_prob: float = 0.65
    mut_prob: float = 0.30
    mutation_sigma: float = 0.20
    tournament_size: int = 3
    eval_max_steps: int = config.VIS_MAX_STEPS
    eval_episodes: int = 3
    random_seed: int = 123
    elites: int = 2
    reevaluate_topk: int = 5


@dataclass
class EvalResult:
    reward: float
    steps: int
    score: float


def _make_policy(policy: FullyConnectedPolicy, flat_params: np.ndarray):
    flat = np.asarray(flat_params, dtype=np.float32)

    def policy_fn(state_vec: np.ndarray) -> dict:
        y = policy.forward(flat, state_vec)
        return policy.output_to_control(y)

    return policy_fn


def eval_policy(
    flat_params: np.ndarray,
    cfg: DEAPConfig,
    *,
    hidden_size: int,
    seeds: Optional[List[int]] = None,
) -> EvalResult:
    seeds = seeds or [cfg.random_seed + idx for idx in range(cfg.eval_episodes)]

    probe = Game(headless=True, rng=random.Random(seeds[0]))
    state_size = probe.get_state().size
    policy = FullyConnectedPolicy(state_size, hidden_size=hidden_size)
    assert flat_params.size == policy.total_param_count, \
        f"Genome size {flat_params.size} != expected {policy.total_param_count} (state={state_size}, hidden={hidden_size})"
    policy_fn = _make_policy(policy, flat_params)

    total_reward = 0.0
    total_steps = 0
    total_score = 0.0

    for seed in seeds:
        game = Game(headless=True, rng=random.Random(seed))
        episode = game.run_headless_episode(policy_fn, max_steps=cfg.eval_max_steps)
        total_reward += episode.reward
        total_steps += episode.steps
        total_score += episode.score

    count = max(1, len(seeds))
    return EvalResult(
        reward=total_reward / count,
        steps=int(round(total_steps / count)),
        score=total_score / count,
    )


class DEAPTrainer:
    def __init__(self, cfg: DEAPConfig, hidden_size: int = 48):
        self.cfg = cfg
        self.hidden_size = hidden_size
        self.rng = np.random.default_rng(cfg.random_seed)
        random.seed(cfg.random_seed)
        self.current_mutation_sigma = cfg.mutation_sigma
        self.history: list[dict[str, int | float]] = []

        probe = Game(headless=True, rng=random.Random(cfg.random_seed))
        state_size = probe.get_state().size
        self.policy = FullyConnectedPolicy(state_size, hidden_size=hidden_size)
        self.param_dim = self.policy.total_param_count

        if not hasattr(creator, "SnakeFitnessMax"):
            creator.create("SnakeFitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "SnakeIndividual"):
            creator.create("SnakeIndividual", np.ndarray, fitness=creator.SnakeFitnessMax)

        self.toolbox = base.Toolbox()
        init_scale = math.sqrt(6.0 / (self.policy.input_size + self.policy.hidden_size))
        self.toolbox.register("attr_float", self.rng.uniform, -init_scale, init_scale)
        self.toolbox.register(
            "individual",
            tools.initRepeat,
            creator.SnakeIndividual,
            self.toolbox.attr_float,
            n=self.param_dim,
        )
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", self._crossover_blend)
        self.toolbox.register("mutate", self._mutate_gaussian)
        self.toolbox.register("select", tools.selTournament, tournsize=self.cfg.tournament_size)

        def eval_wrapper(individual):
            result = eval_policy(individual, self.cfg, hidden_size=self.hidden_size)
            return (result.reward,)

        self.toolbox.register("evaluate", eval_wrapper)

    def _crossover_blend(self, ind1, ind2, alpha: float = 0.20):
        if self.rng.random() < self.cfg.cx_prob:
            blend = self.rng.uniform(-alpha, 1 + alpha, size=ind1.shape)
            child1 = blend * ind1 + (1 - blend) * ind2
            child2 = blend * ind2 + (1 - blend) * ind1
            ind1[:] = child1
            ind2[:] = child2
        return ind1, ind2

    def _mutate_gaussian(self, individual):
        mask = self.rng.random(individual.shape) < self.cfg.mut_prob
        noise = self.rng.normal(0.0, self.current_mutation_sigma, size=individual.shape)
        individual[:] = individual + mask * noise
        return (individual,)

    def _seed_population_from_genome(self, genome: np.ndarray, n: int) -> List[np.ndarray]:
        population = []
        jitter_sigma = max(1e-4, self.cfg.mutation_sigma * 0.5)
        for _ in range(n):
            individual = creator.SnakeIndividual(np.array(genome, copy=True))
            noise = self.rng.normal(0.0, jitter_sigma, size=individual.shape)
            individual[:] = individual + noise
            population.append(individual)
        return population

    def evolve(self, seed_genome: Optional[np.ndarray] = None) -> Tuple[List[np.ndarray], float, np.ndarray]:
        if seed_genome is not None and seed_genome.size != self.param_dim:
            raise ValueError(
                f"Seed genome has wrong size: {seed_genome.size} != {self.param_dim}"
            )

        population = (
            self._seed_population_from_genome(seed_genome, self.cfg.population_size)
            if seed_genome is not None
            else self.toolbox.population(n=self.cfg.population_size)
        )

        for individual, fit in zip(population, map(self.toolbox.evaluate, population)):
            individual.fitness.values = fit

        sigma0, sigma_min = self.cfg.mutation_sigma, max(0.02, self.cfg.mutation_sigma * 0.25)
        for gen in range(self.cfg.generations):
            elites = tools.selBest(population, self.cfg.elites)
            offspring = self.toolbox.select(population, len(population) - self.cfg.elites)
            offspring = list(map(self.toolbox.clone, offspring))

            for idx in range(0, len(offspring), 2):
                if idx + 1 < len(offspring):
                    self.toolbox.mate(offspring[idx], offspring[idx + 1])
                    if hasattr(offspring[idx].fitness, "values"):
                        del offspring[idx].fitness.values
                    if hasattr(offspring[idx + 1].fitness, "values"):
                        del offspring[idx + 1].fitness.values

            frac = gen / max(1, self.cfg.generations - 1)
            self.current_mutation_sigma = sigma_min + (sigma0 - sigma_min) * (1.0 - frac)

            for mutant in offspring:
                self.toolbox.mutate(mutant)
                if hasattr(mutant.fitness, "values"):
                    del mutant.fitness.values

            invalid = [individual for individual in offspring if not individual.fitness.valid]
            for individual, fit in zip(invalid, map(self.toolbox.evaluate, invalid)):
                individual.fitness.values = fit

            population[:] = elites + offspring

            best = tools.selBest(population, 1)[0]
            mean_fit = float(np.mean([individual.fitness.values[0] for individual in population]))
            best_result = eval_policy(best, self.cfg, hidden_size=self.hidden_size)
            self.history.append(
                {
                    "generation": gen,
                    "best_fitness": float(best.fitness.values[0]),
                    "mean_fitness": mean_fit,
                    "score": float(best_result.score),
                    "steps": int(best_result.steps),
                    "mutation_sigma": float(self.current_mutation_sigma),
                }
            )
            print(
                f"[SNAKE] Gen {gen:03d} | Best={best.fitness.values[0]:.3f} "
                f"| Mean={mean_fit:.3f} | Score={best_result.score:.2f} | Steps={best_result.steps}"
            )

        top_k = tools.selBest(population, min(self.cfg.reevaluate_topk, len(population)))
        scores = [eval_policy(individual, self.cfg, hidden_size=self.hidden_size).reward for individual in top_k]
        best_index = int(np.argmax(scores))
        true_best = np.asarray(top_k[best_index], dtype=np.float32)
        true_best_f = float(scores[best_index])
        return population, true_best_f, true_best


def load_genome(path: str) -> np.ndarray:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return np.load(path)


def save_genome(path: str, genome: np.ndarray) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    np.save(path, genome)
