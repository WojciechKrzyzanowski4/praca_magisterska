from __future__ import annotations

import sys
from typing import Any, Dict, NamedTuple, Optional

import numpy as np
import pygame

from .snake_constants import config
from .snake_entites import Food, Snake


class EpisodeStats(NamedTuple):
    reward: float
    steps: int
    score: int
    length: int


class Game:
    def __init__(self, headless: bool = False, rng=None):
        self.headless = headless
        self.rng = rng
        pygame.init()
        pygame.display.set_caption("Snake")
        flags = pygame.HIDDEN if headless else 0
        self.screen = pygame.display.set_mode((config.WIDTH, config.HEIGHT), flags)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 20)
        self.big_font = pygame.font.SysFont("consolas", 36, bold=True)
        self.reset()

    def reset(self):
        self.snake = Snake()
        self.food = Food(set(self.snake.body), rng=self.rng)
        self.score = 0
        self.steps_survived = 0
        self.steps_without_food = 0
        self.finished = False
        self.won = False

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE,):
                    pygame.quit()
                    sys.exit()
                if event.key in (pygame.K_r, pygame.K_RETURN):
                    if self.finished:
                        self.reset()

    def _keyboard_control(self) -> Dict[str, bool]:
        keys = pygame.key.get_pressed()
        return {
            "move_up": bool(keys[pygame.K_UP] or keys[pygame.K_w]),
            "move_down": bool(keys[pygame.K_DOWN] or keys[pygame.K_s]),
            "move_left": bool(keys[pygame.K_LEFT] or keys[pygame.K_a]),
            "move_right": bool(keys[pygame.K_RIGHT] or keys[pygame.K_d]),
        }

    def _apply_control(self, control: Optional[Dict[str, Any]]) -> None:
        if self.finished or not self.snake.alive or not control:
            return
        direction_x, direction_y = self.snake.direction
        if control.get("turn_left", False):
            self.snake.set_direction((direction_y, -direction_x))
        elif control.get("turn_right", False):
            self.snake.set_direction((-direction_y, direction_x))
        elif control.get("move_up", False):
            self.snake.set_direction(config.UP)
        elif control.get("move_down", False):
            self.snake.set_direction(config.DOWN)
        elif control.get("move_left", False):
            self.snake.set_direction(config.LEFT)
        elif control.get("move_right", False):
            self.snake.set_direction(config.RIGHT)

    def _next_head(self, direction) -> tuple[int, int]:
        head_x, head_y = self.snake.body[0]
        dx, dy = direction
        return head_x + dx, head_y + dy

    def _would_collide(self, cell: tuple[int, int]) -> bool:
        x, y = cell
        if not (0 <= x < config.GRID_WIDTH and 0 <= y < config.GRID_HEIGHT):
            return True
        occupied = self.snake.body if self.snake.grow > 0 else self.snake.body[:-1]
        return cell in occupied

    def step(self, control: Optional[Dict[str, Any]] = None) -> tuple[float, bool]:
        if self.finished:
            return 0.0, True

        self._apply_control(control or {})
        self.snake.update()
        self.steps_survived += 1
        self.steps_without_food += 1

        reward = config.STEP_PENALTY

        if self.snake.alive and self.food.pos is not None and self.snake.body[0] == self.food.pos:
            speed_fraction = max(
                0.0,
                1.0 - self.steps_without_food / config.MAX_STEPS_WITHOUT_FOOD,
            )
            reward += config.FOOD_REWARD + config.SPEED_REWARD_MAX * speed_fraction
            self.snake.feed()
            self.score += 1
            self.steps_without_food = 0
            self.food.respawn(set(self.snake.body))
            if self.food.pos is None:
                self.finished = True
                self.won = True

        if not self.snake.alive:
            self.finished = True
        elif self.steps_without_food >= config.MAX_STEPS_WITHOUT_FOOD:
            self.finished = True

        return float(reward), self.finished

    def _update(self):
        if self.finished:
            return
        self.step(self._keyboard_control())

    def _draw(self):
        if self.headless:
            return
        self.screen.fill(config.COLOR_BG)
        if not self.finished:
            self.food.draw(self.screen)
            self.snake.draw(self.screen)
            hud = self.font.render(f"Score: {self.score}",
                                   True, config.COLOR_FG)
            self.screen.blit(hud, (12, 10))
            instr = self.font.render(
                "Move: Arrow Keys / WASD   Restart: R/Enter   Quit: Esc",
                True,
                config.COLOR_MUTED
            )
            self.screen.blit(instr, (12, config.HEIGHT - 28))
        else:
            title_text = "You Win" if self.won else "Game Over"
            title = self.big_font.render(title_text, True, config.COLOR_FG)
            score = self.font.render(f"Score: {self.score}", True, config.COLOR_FG)
            tip = self.font.render("Press R or Enter to play again • Esc to quit", True, config.COLOR_MUTED)

            self.screen.blit(title, ((config.WIDTH - title.get_width()) // 2,
                                     config.HEIGHT // 2 - 40))
            self.screen.blit(score, ((config.WIDTH - score.get_width()) // 2,
                                     config.HEIGHT // 2))
            self.screen.blit(tip, ((config.WIDTH - tip.get_width()) // 2,
                                   config.HEIGHT // 2 + 30))
        pygame.display.flip()

    def get_state(self) -> np.ndarray:
        # Input layout:
        #   [0:24]  eight head-local rays ordered clockwise from straight ahead;
        #           every ray contains inverse wall distance, inverse body
        #           distance, and a flag indicating visible food
        #   [24:27] immediate collision flags: left, straight, right
        #   [27:30] food in the head-local frame: forward component, right
        #           component, and normalized Euclidean distance
        #   [30:32] tail position in the head-local frame
        #   [32]    normalized snake length
        head_x, head_y = self.snake.body[0]
        food_x, food_y = self.food.pos if self.food.pos is not None else (head_x, head_y)
        max_w = max(1, config.GRID_WIDTH - 1)
        max_h = max(1, config.GRID_HEIGHT - 1)
        diagonal = float(np.hypot(max_w, max_h))
        scale = float(max(max_w, max_h))

        forward = self.snake.direction
        right = (-forward[1], forward[0])
        backward = (-forward[0], -forward[1])
        left = (-right[0], -right[1])
        ray_directions = [
            forward,
            (forward[0] + right[0], forward[1] + right[1]),
            right,
            (backward[0] + right[0], backward[1] + right[1]),
            backward,
            (backward[0] + left[0], backward[1] + left[1]),
            left,
            (forward[0] + left[0], forward[1] + left[1]),
        ]

        body = set(self.snake.body[1:])
        state = []
        for ray_x, ray_y in ray_directions:
            distance = 0
            body_distance = None
            food_visible = 0.0

            while True:
                distance += 1
                cell = (
                    head_x + ray_x * distance,
                    head_y + ray_y * distance,
                )
                cell_x, cell_y = cell
                if not (
                    0 <= cell_x < config.GRID_WIDTH
                    and 0 <= cell_y < config.GRID_HEIGHT
                ):
                    break
                if body_distance is None and cell in body:
                    body_distance = distance
                if self.food.pos is not None and cell == self.food.pos:
                    food_visible = 1.0

            wall_proximity = 1.0 / distance
            body_proximity = (
                0.0 if body_distance is None else 1.0 / body_distance
            )
            state.extend([wall_proximity, body_proximity, food_visible])

        state.extend([
            float(self._would_collide(self._next_head(left))),
            float(self._would_collide(self._next_head(forward))),
            float(self._would_collide(self._next_head(right))),
        ])

        food_delta_x = food_x - head_x
        food_delta_y = food_y - head_y
        food_forward = (
            food_delta_x * forward[0] + food_delta_y * forward[1]
        ) / scale
        food_right = (
            food_delta_x * right[0] + food_delta_y * right[1]
        ) / scale
        food_distance = float(np.hypot(food_x - head_x, food_y - head_y) / diagonal)
        state.extend([
            float(food_forward),
            float(food_right),
            food_distance,
        ])

        tail_x, tail_y = self.snake.body[-1]
        tail_delta_x = tail_x - head_x
        tail_delta_y = tail_y - head_y
        tail_forward = (
            tail_delta_x * forward[0] + tail_delta_y * forward[1]
        ) / scale
        tail_right = (
            tail_delta_x * right[0] + tail_delta_y * right[1]
        ) / scale
        state.extend([float(tail_forward), float(tail_right)])
        state.append(
            float(len(self.snake.body) / (config.GRID_WIDTH * config.GRID_HEIGHT))
        )
        return np.asarray(state, dtype=np.float32)

    def run_headless_episode(self, policy_fn, max_steps: int) -> EpisodeStats:
        self.headless = True
        self.reset()

        total_reward = 0.0
        steps = 0

        while not self.finished and steps < max_steps:
            state_vec = self.get_state()
            control = policy_fn(state_vec) or {}
            reward, _ = self.step(control)
            total_reward += float(reward or 0.0)
            steps += 1

        return EpisodeStats(
            reward=total_reward,
            steps=steps,
            score=self.score,
            length=len(self.snake.body),
        )

    def run(self):
        while True:
            self.clock.tick(config.FPS)
            self._handle_events()
            self._update()
            self._draw()

if __name__ == "__main__":
    game = Game()
    game.run()
