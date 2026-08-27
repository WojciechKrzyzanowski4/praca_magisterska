import math
import sys
from typing import Any, Dict, NamedTuple, Optional, Tuple

import numpy as np
import pygame

from .asteroid_constants import config
from .asteroid_entites import Asteroid, Ship
from .asteroid_utils import (
    aimed_heading_deg,
    clamp,
    cycle_get,
    spawn_radius,
    spawn_speed,
    vec_from_heading,
)


class NearestInfo(NamedTuple):
    pos: Optional[config.Vec2]
    vel: Tuple[float, float]
    dist: float
    r_hat: Tuple[float, float]


class EpisodeStats(NamedTuple):
    reward: float
    steps: int
    kills: int
    bullets: int
    waves: int
    deaths: int


class Game:
    def __init__(self, lives: int = 3, waves: int = 5, headless: bool = False):
        self.headless = headless
        self.finished = False
        self.waves = waves
        self.wave = 0
        self.lives_count = lives
        self.lives = lives

        pygame.init()
        pygame.display.set_caption("Asteroids")
        self.screen = pygame.display.set_mode((config.WIDTH, config.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 20)
        self.big_font = pygame.font.SysFont("consolas", 36, bold=True)

        self.ship = Ship(config.Vec2(config.WIDTH / 2, config.HEIGHT / 2))
        self.bullets: list = []
        self.asteroids: list[Asteroid] = []

        self.score = 0
        self.kills = 0
        self.waves_cleared = 0
        self.bullets_fired = 0
        self.spawn_idx = 0
        self.angle_idx = 0

        self.phase1_turn_direction = 0
        self.phase1_turn_streak = 0.0
        self.phase2_alignment_sum = 0.0
        self.phase2_alignment_steps = 0
        self.phase3_seconds_since_kill = 0.0
        self.prev_dist_potential = None

        self._spawn_initial_asteroids()

    def reset(self) -> None:
        self.finished = False
        self.lives = self.lives_count
        self.ship = Ship(config.Vec2(config.WIDTH / 2, config.HEIGHT / 2))
        self.bullets.clear()
        self.asteroids.clear()

        self.score = 0
        self.kills = 0
        self.waves_cleared = 0
        self.bullets_fired = 0

        self.phase1_turn_direction = 0
        self.phase1_turn_streak = 0.0
        self.phase2_alignment_sum = 0.0
        self.phase2_alignment_steps = 0
        self.phase3_seconds_since_kill = 0.0
        self.prev_dist_potential = None

        self._spawn_initial_asteroids()

    def set_scenario_offset(self, idx: int):
        self.spawn_idx = idx % len(config.AST_SPAWN_POINTS)
        self.angle_idx = idx % len(config.AST_SPAWN_HEADINGS_DEG)

    def _make_vf(self, vel: config.Vec2):
        base_dir = vel.normalize() if vel.length() > 1e-6 else config.Vec2(1.0, 0.0)
        base_speed = vel.length()
        def vf(extra: float = 1.0, _bd=base_dir, _bs=base_speed):
            return _bd * (_bs * extra)
        return vf

    def _spawn_initial_asteroids(self) -> None:
        self.asteroids.clear()
        spawned = 0
        if config.AST_FORCE_CLOSURE:
            px, py = cycle_get(config.AST_SPAWN_POINTS, self.spawn_idx)
            self.spawn_idx += 1
            pos = config.Vec2(px * config.WIDTH, py * config.HEIGHT)
            head_deg = aimed_heading_deg(pos, self.ship.pos, config.AST_FORCE_CLOSURE_OFFSET_DEG)
            speed = spawn_speed()
            vel = vec_from_heading(head_deg, speed)
            r = spawn_radius()
            self.asteroids.append(Asteroid(pos, vel, r, vel_factory=self._make_vf(vel)))
            spawned += 1
        remaining = max(0, config.AST_INIT_COUNT - spawned)
        for _ in range(remaining):
            px, py = cycle_get(config.AST_SPAWN_POINTS, self.spawn_idx)
            hd = cycle_get(config.AST_SPAWN_HEADINGS_DEG, self.angle_idx)
            self.spawn_idx += 1
            self.angle_idx += 1
            pos = config.Vec2(px * config.WIDTH, py * config.HEIGHT)
            if (pos - self.ship.pos).length() < 140.0:
                pos = pos + vec_from_heading(hd, 200.0)
            speed = spawn_speed()
            vel = vec_from_heading(hd, speed)
            r = spawn_radius()
            self.asteroids.append(Asteroid(pos, vel, r, vel_factory=self._make_vf(vel)))

    def _handle_events(self) -> None:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            if e.type == pygame.KEYDOWN and (e.key in (pygame.K_ESCAPE, pygame.K_q)):
                pygame.quit()
                sys.exit(0)
            if e.type == pygame.KEYDOWN and (e.key in (pygame.K_r, pygame.K_RETURN)):
                self.reset()

    def _update_bullets(self, dt: float) -> None:
        for b in self.bullets:
            b.update(dt)
        self.bullets = [b for b in self.bullets if b.alive()]

    def _update_asteroids(self, dt: float) -> None:
        for a in self.asteroids:
            a.update(dt)

    def _draw_hud(self) -> None:
        hud = self.font.render(
            f"Score: {self.score}   Lives: {self.lives}   Wave: {self.wave}   Kills: {self.kills}",
            True, config.COLOR_FG
        )
        self.screen.blit(hud, (12, 10))
        instr = self.font.render("←/→ rotate   ↑ thrust   SPACE shoot   ESC quit", True, config.COLOR_MUTED)
        self.screen.blit(instr, (12, config.HEIGHT - 28))

    def _draw_game_over(self) -> None:
        if self.lives > 0:
            title = self.big_font.render("You win!", True, config.COLOR_FG)
        else:
            title = self.big_font.render("Game Over!", True, config.COLOR_FG)
        score = self.font.render(f"Score: {self.score}", True, config.COLOR_FG)
        tip = self.font.render("Press R or Enter to play again • Esc to quit", True, config.COLOR_MUTED)
        self.screen.blit(title, ((config.WIDTH - title.get_width()) // 2, config.HEIGHT // 2 - 40))
        self.screen.blit(score, ((config.WIDTH - score.get_width()) // 2, config.HEIGHT // 2))
        self.screen.blit(tip, ((config.WIDTH - tip.get_width()) // 2, config.HEIGHT // 2 + 30))


    def _draw(self) -> None:
        if self.headless:
            return
        self.screen.fill(config.COLOR_BG)
        if not self.finished:
            self.ship.draw(self.screen)
            for a in self.asteroids:
                a.draw(self.screen)
            for b in self.bullets:
                b.draw(self.screen)
            self._draw_hud()
        else:
            self._draw_game_over()
        pygame.display.flip()

    def _handle_collisions(self) -> None:
        new_asteroids: list[Asteroid] = []
        for a in self.asteroids:
            hit = False
            for b in self.bullets:
                if (b.pos - a.pos).length() <= a.radius:
                    hit = True
                    b.ttl = 0
                    break
            if hit:
                self.kills += 1
                self.score += max(1, int(60 / max(a.radius, 1)))
                new_asteroids += a.split()
            else:
                new_asteroids.append(a)
        self.asteroids = new_asteroids
        if self.ship.invincibility <= 0:
            for a in self.asteroids:
                if (self.ship.pos - a.pos).length() <= a.radius + config.SHIP_RADIUS * 0.75:
                    self.lives -= 1
                    if self.lives <= 0:
                        self.finished = True
                    else:
                        self.ship = Ship(config.Vec2(config.WIDTH / 2, config.HEIGHT / 2))
                    break
        if not self.asteroids and not self.finished:
            self.wave += 1
            self.waves_cleared += 1
            if self.waves_cleared >= self.waves:
                self.finished = True
            else:
                self._spawn_initial_asteroids()

    def _update(self, dt: float) -> None:
        if self.finished:
            return
        keys = pygame.key.get_pressed()
        self.ship.update(dt, keys)
        if keys[pygame.K_SPACE] and self.ship.can_shoot():
            self.bullets.append(self.ship.shoot())
            self.bullets_fired += 1
        self._update_bullets(dt)
        self._update_asteroids(dt)
        self._handle_collisions()

    def run(self) -> None:
        while True:
            dt = self.clock.tick(config.FPS) / 1000.0
            self._handle_events()
            self._update(dt)
            self._draw()

    def get_state(self, max_asteroids: int = config.OBS_MAX_ASTEROIDS) -> np.ndarray:
        # Input layout:
        #   [0:4]   ship features in ship-local coordinates:
        #           forward velocity, lateral velocity, normalized shoot cooldown,
        #           normalized invincibility timer
        #   Then, for each nearest asteroid (up to max_asteroids), append:
        #           normalized distance,
        #           bearing cosine relative to ship forward,
        #           bearing sine relative to ship right,
        #           relative forward velocity,
        #           relative lateral velocity,
        #           normalized asteroid radius
        #   Missing asteroid slots are zero-padded so the network always sees
        #   a fixed-size vector ordered by nearest threats first.
        # ship frame
        ang = math.radians(self.ship.angle)
        fwd = config.Vec2(math.cos(ang), math.sin(ang))
        right = config.Vec2(fwd.y, -fwd.x)

        # ship kinematics in its own frame
        sv_fwd = clamp(self.ship.vel.dot(fwd) / max(1e-6, config.SHIP_SPEED_NORM), -1.0, 1.0)
        sv_right = clamp(self.ship.vel.dot(right) / max(1e-6, config.SHIP_SPEED_NORM), -1.0, 1.0)

        cooldown_n = clamp(1.0 - (self.ship.cooldown / max(1e-6, config.SHOOT_COOLDOWN)), 0.0, 1.0)
        invinc_n = clamp(1.0 - (self.ship.invincibility / 1.2), 0.0, 1.0)

        state = [sv_fwd, sv_right, cooldown_n, invinc_n]

        ship_pos = self.ship.pos
        # nearest-first for stability
        asts = sorted(self.asteroids, key=lambda a: (a.pos - ship_pos).length())[:max_asteroids]

        # normalize distances by arena size (diagonal)
        diag = (config.WIDTH ** 2 + config.HEIGHT ** 2) ** 0.5
        d_norm = max(1e-6, 0.75 * diag)

        for a in asts:
            rel = a.pos - ship_pos
            d = rel.length()
            if d > 1e-6:
                r_hat = rel / d
                bearing_cos = float(r_hat.dot(fwd))  # +1 = straight ahead
                bearing_sin = float(r_hat.dot(right))  # +1 = to the right
            else:
                d, bearing_cos, bearing_sin = 0.0, 1.0, 0.0

            # relative velocity (asteroid - ship) in ship frame
            relv = a.vel - self.ship.vel
            v_fwd = clamp(relv.dot(fwd) / max(1e-6, config.AST_SPEED_NORM), -1.0, 1.0)
            v_right = clamp(relv.dot(right) / max(1e-6, config.AST_SPEED_NORM), -1.0, 1.0)

            dist = clamp(d / d_norm, 0.0, 1.0)
            size = clamp(a.radius / max(1e-6, config.AST_MAX_RADIUS), 0.0, 1.0)

            # Per-asteroid block order is intentionally stable because the policy
            # is fully connected and relies on this exact feature packing.
            state.extend([dist, bearing_cos, bearing_sin, v_fwd, v_right, size])

        # pad
        needed = max_asteroids - len(asts)
        if needed > 0:
            state.extend([0.0] * (needed * 6))

        return np.asarray(state, dtype=np.float32)

    def common_step(self, control: Dict[str, Any], fixed_dt: float = 1.0 / 60) -> Tuple[int, int]:
        shots_before = self.bullets_fired
        kills_before = self.kills
        self.ship.update_with_controls(
            fixed_dt,
            control.get('rotate_left', False),
            control.get('rotate_right', False),
            control.get('thrust', False)
        )
        if control.get('shoot', False) and self.ship.can_shoot():
            self.bullets.append(self.ship.shoot())
            self.bullets_fired += 1
        self._update_bullets(fixed_dt)
        self._update_asteroids(fixed_dt)
        self._handle_collisions()
        return shots_before, kills_before

    SURVIVE_PER_SEC: float = 50.0
    DEATH_PENALTY: float = 1200.0
    DIST_CAP: float = 600.0
    DIST_GAIN: float = 0.02
    P1_TURN_STREAK_GRACE: float = 1.0
    P1_SPIN_COST_PER_SEC: float = 2.0

    def _nearest_distance(self) -> float:
        if not self.asteroids:
            return float("inf")
        sp = self.ship.pos
        return min((a.pos - sp).length() for a in self.asteroids)

    def _avoidance_reward_after_step(self, control: dict, fixed_dt: float) -> float:
        """Shared phase-one behavior: survival, distance and anti-spin reward."""
        reward = self.SURVIVE_PER_SEC * fixed_dt

        d = self._nearest_distance()
        d_eff = min(d, self.DIST_CAP)
        if self.prev_dist_potential is None:
            self.prev_dist_potential = d_eff
        reward += self.DIST_GAIN * (d_eff - self.prev_dist_potential)
        self.prev_dist_potential = d_eff

        rotate_left = bool(control.get("rotate_left", False))
        rotate_right = bool(control.get("rotate_right", False))
        turn_direction = -1 if rotate_left and not rotate_right else 1 if rotate_right and not rotate_left else 0

        if turn_direction != 0 and turn_direction == self.phase1_turn_direction:
            self.phase1_turn_streak += fixed_dt
        elif turn_direction != 0:
            self.phase1_turn_streak = fixed_dt
        else:
            self.phase1_turn_streak = 0.0
        self.phase1_turn_direction = turn_direction

        # Only sustained rotation in one direction is penalized. Short turns and
        # direction changes remain free, so ordinary avoidance is unaffected.
        excess_streak = max(0.0, self.phase1_turn_streak - self.P1_TURN_STREAK_GRACE)
        spin_strength = clamp(excess_streak / 2.0, 0.0, 1.0)
        reward -= self.P1_SPIN_COST_PER_SEC * spin_strength * fixed_dt

        if self.finished and self.lives == 0:
            reward -= self.DEATH_PENALTY

        return reward

    def step_phase1(self, control: dict, fixed_dt: float = 1.0 / 60.0):
        """Phase 1: survive + gently increase nearest distance."""
        shots_before, _ = self.common_step(control, fixed_dt)
        reward = self._avoidance_reward_after_step(control, fixed_dt)

        shots_delta = self.bullets_fired - shots_before
        if shots_delta > 0:
            reward -= 200.0 * shots_delta

        return reward, self.finished

    P2_AIM_REWARD_PER_SEC: float = 10.0

    def _nearest_asteroid_alignment(self) -> float:
        """Cosine alignment of the ship nose with the nearest asteroid."""
        if not self.asteroids:
            return 0.0

        nearest = min(self.asteroids, key=lambda asteroid: (asteroid.pos - self.ship.pos).length_squared())
        relative_position = nearest.pos - self.ship.pos
        if relative_position.length_squared() <= 1e-9:
            return 1.0

        ship_forward = config.Vec2(1.0, 0.0).rotate(self.ship.angle)
        return float(ship_forward.dot(relative_position.normalize()))

    def phase2_mean_alignment(self) -> float:
        return self.phase2_alignment_sum / max(1, self.phase2_alignment_steps)

    def step_phase2(self, control: Dict[str, Any], fixed_dt: float = 1.0 / 60.0):
        """Phase 2: preserve phase-one avoidance and learn to aim without shooting."""
        phase2_control = dict(control)
        phase2_control["shoot"] = False
        self.common_step(phase2_control, fixed_dt)
        reward = self._avoidance_reward_after_step(phase2_control, fixed_dt)

        alignment = self._nearest_asteroid_alignment()
        reward += self.P2_AIM_REWARD_PER_SEC * alignment * fixed_dt
        self.phase2_alignment_sum += alignment
        self.phase2_alignment_steps += 1

        return reward, self.finished

    P3_KILL_REWARD: float = 200.0
    P3_SHOT_COST: float = 100.0
    P3_WAVE_CLEAR_REWARD: float = 3000.0
    P3_WIN_REWARD: float = 5000.0
    P3_PROGRESS_GRACE_SECONDS: float = 5.0
    P3_STALL_COST_PER_SEC: float = 60.0

    def step_phase3(self, control: Dict[str, Any], fixed_dt: float = 1.0 / 60.0):
        """Phase 3: preserve avoidance and aiming, then learn efficient shooting."""
        waves_before = self.waves_cleared
        shots_before, kills_before = self.common_step(control, fixed_dt)
        reward = self._avoidance_reward_after_step(control, fixed_dt)

        alignment = self._nearest_asteroid_alignment()
        reward += self.P2_AIM_REWARD_PER_SEC * alignment * fixed_dt
        self.phase2_alignment_sum += alignment
        self.phase2_alignment_steps += 1

        kills_delta = self.kills - kills_before
        shots_delta = self.bullets_fired - shots_before
        waves_delta = self.waves_cleared - waves_before
        reward += self.P3_KILL_REWARD * float(kills_delta)
        reward -= self.P3_SHOT_COST * float(shots_delta)
        reward += self.P3_WAVE_CLEAR_REWARD * float(waves_delta)

        if kills_delta > 0:
            self.phase3_seconds_since_kill = 0.0
        else:
            self.phase3_seconds_since_kill += fixed_dt

        if not self.finished and self.phase3_seconds_since_kill > self.P3_PROGRESS_GRACE_SECONDS:
            reward -= self.P3_STALL_COST_PER_SEC * fixed_dt

        if self.finished and self.lives > 0:
            reward += self.P3_WIN_REWARD

        return reward, self.finished

    def run_headless_episode_phase(self, policy_fn, phase: int, max_steps: int = 60 * 30) -> EpisodeStats:
        self.headless = True
        self.finished = False
        self.lives = self.lives_count

        total_reward = 0.0
        steps = 0
        kills0 = self.kills
        bullets0 = self.bullets_fired

        while not self.finished and steps < max_steps:
            state_vec = self.get_state()
            control = policy_fn(state_vec) or {}
            if phase == 1:
                r, _ = self.step_phase1(control, 1.0 / 60.0)
            elif phase == 2:
                r, _ = self.step_phase2(control, 1.0 / 60.0)
            else:
                r, _ = self.step_phase3(control, 1.0 / 60.0)
            total_reward += (r or 0.0)
            steps += 1

        return EpisodeStats(
            reward=total_reward,
            steps=steps,
            kills=self.kills - kills0,
            bullets=self.bullets_fired - bullets0,
            waves=self.waves_cleared,
            deaths=int(self.lives_count - self.lives),
        )
