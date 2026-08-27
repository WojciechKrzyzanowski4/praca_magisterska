import dataclasses
import os

import pygame


@dataclasses.dataclass
class AsteroidConfig:
    WIDTH, HEIGHT = 960, 720
    FPS = 60

    SHIP_RADIUS = 12
    SHIP_ROT_SPEED = 320
    SHIP_THRUST = 240
    SHIP_FRICTION = 0.980
    BULLET_SPEED = 720
    BULLET_TTL = 1.2
    SHOOT_COOLDOWN = 0.10
    WAVE_SPEED_GROWTH = 0.02


    AST_MIN_RADIUS = 14
    AST_MAX_RADIUS = 48
    SPLIT_LARGE_MIN = 30.0
    SPLIT_MED_MIN   = 22.0
    AST_SPEED_RANGE = (30, 90)
    AST_SPLIT_SCALE = 0.7

    COLOR_BG = (0, 0, 0)
    COLOR_FG = (230, 235, 240)
    COLOR_MUTED = (150, 160, 170)

    OBS_MAX_ASTEROIDS = 12
    SHIP_SPEED_NORM = 600.0
    AST_SPEED_NORM = 360.0
    Vec2 = pygame.math.Vector2

    V_MIN_FOR_IDLE = 10.0

    AST_SPAWN_POINTS = [
        (0.10, 0.10),
        (0.90, 0.10),
        (0.10, 0.90),
        (0.90, 0.90),
        (0.50, 0.00),
        (0.00, 0.50),
        (1.00, 0.50),
        (0.50, 1.00),
        (0.25, 0.00),
        (0.75, 0.00),
        (0.25, 1.00),
        (0.75, 1.00),
    ]

    USE_DETERMINISTIC_SPAWNS = True

    AST_SPAWN_HEADINGS_DEG = [
        45,
        135,
        -45,
        -135,
        0,
        90,
        180,
        -90,
        30,
        150,
        -30,
        -150,
    ]

    AST_INIT_COUNT = len(AST_SPAWN_HEADINGS_DEG)

    AST_FORCE_CLOSURE = True

    AST_FORCE_CLOSURE_OFFSET_DEG = 8.0

    AST_BASE_SPEED = None

    AST_INIT_RADIUS_RANGE = (AST_MAX_RADIUS * 0.5, AST_MAX_RADIUS)

    SPLIT_CHILD_ANGLE_OFFSETS_DEG = [-35.0, +35.0, 0.0]
    SPLIT_SPEED_MULT = 1.25

    MODULE_DIR = os.path.abspath(os.path.dirname(__file__))
    OUTPUT_DIR = os.path.join(MODULE_DIR, "output")
    PHASE_FILE = {
        1: os.path.join(OUTPUT_DIR, "asteroid_killer_phase1.npy"),
        2: os.path.join(OUTPUT_DIR, "asteroid_killer_phase2.npy"),
        3: os.path.join(OUTPUT_DIR, "asteroid_killer_phase3.npy"),
    }

    TRAIN_SCENARIOS = [0, 2, 4, 6, 8, 10, 12, 14]
    HOLDOUT_SCENARIOS = [1, 5, 9]


config = AsteroidConfig()
