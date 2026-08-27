import os
import random

import pygame

from .snake_constants import config


def draw_cell(surface, pos, color, inset=2):
    if pos is None:
        return
    x, y = pos
    rect = pygame.Rect(x * config.CELL_SIZE + inset, y * config.CELL_SIZE + inset,
                       config.CELL_SIZE - inset * 2, config.CELL_SIZE - inset * 2)
    pygame.draw.rect(surface, color, rect)


def random_empty_cell(occupied, rng=None):
    rng = rng or random
    free_cells = [
        (x, y)
        for y in range(config.GRID_HEIGHT)
        for x in range(config.GRID_WIDTH)
        if (x, y) not in occupied
    ]
    if not free_cells:
        return None
    selected_cell = rng.choice(free_cells)
    return int(selected_cell[0]), int(selected_cell[1])


def ensure_output_dir():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
