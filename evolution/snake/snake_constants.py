import dataclasses
import os


@dataclasses.dataclass
class SnakeConstants:
    FPS = 20
    CELL_SIZE = 20
    GRID_WIDTH = 30
    GRID_HEIGHT = 24
    WIDTH = GRID_WIDTH * CELL_SIZE
    HEIGHT = GRID_HEIGHT * CELL_SIZE
    SNAKE_COLOR = (60, 220, 130)
    SNAKE_HEAD_COLOR = (90, 250, 160)
    FOOD_COLOR = (240, 80, 95)
    TEXT_COLOR = (235, 235, 235)

    COLOR_BG = (0, 0, 0)
    COLOR_FG = (230, 235, 240)
    COLOR_MUTED = (150, 160, 170)

    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")
    MODEL_FILE = os.path.join(OUTPUT_DIR, "snake_agent.npy")

    FOOD_REWARD = 1.0
    SPEED_REWARD_MAX = 2.0
    STEP_PENALTY = -0.002
    MAX_STEPS_WITHOUT_FOOD = 300
    VIS_MAX_STEPS = GRID_WIDTH * GRID_HEIGHT * 20


config = SnakeConstants()
