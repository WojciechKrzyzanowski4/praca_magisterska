import math
import os
import random
from typing import Optional

import numpy as np

from .asteroid_constants import config


def wrap(pos: config.Vec2) -> config.Vec2:
    return config.Vec2(pos.x % config.WIDTH, pos.y % config.HEIGHT)


def rand_vel(speed_min=config.AST_SPEED_RANGE[0], speed_max=config.AST_SPEED_RANGE[1]) -> config.Vec2:
    ang = random.uniform(0, math.tau)
    spd = random.uniform(speed_min, speed_max)
    return config.Vec2(math.cos(ang), math.sin(ang)) * spd


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def norm(v, scale):
    if scale <= 0:
        return 0.0
    return clamp(v / scale, -1.0, 1.0)


def angle_diff_deg(a_deg: float, b_deg: float) -> float:
    x = (a_deg - b_deg + 180.0) % 360.0 - 180.0
    return x


def cycle_get(seq, idx):
    return seq[idx % len(seq)]


def vec_from_heading(deg, speed):
    rad = math.radians(deg)
    return config.Vec2(math.cos(rad) * speed, math.sin(rad) * speed)


def aimed_heading_deg(from_pos, to_pos, extra_deg=0.0):
    d = to_pos - from_pos
    ang = math.degrees(math.atan2(d.y, d.x))
    return ang + extra_deg


def spawn_speed():
    if config.AST_BASE_SPEED is not None:
        return float(config.AST_BASE_SPEED)
    lo, hi = config.AST_SPEED_RANGE
    return 0.5 * (lo + hi)


def spawn_radius():
    lo, hi = config.AST_INIT_RADIUS_RANGE
    return float((lo * hi) ** 0.5)


def ensure_output_dir():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)


def load_genome_if_exists(path: str) -> Optional[np.ndarray]:
    return np.load(path) if os.path.isfile(path) else None


def phase_step_method(game, phase: int):
    method_name = f"step_phase{phase}"
    if hasattr(game, method_name):
        return getattr(game, method_name)
    return game.step_phase3
