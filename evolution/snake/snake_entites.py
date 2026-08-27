from .snake_constants import config
from .snake_utils import draw_cell, random_empty_cell


class Snake:
    def __init__(self):
        cx, cy = config.GRID_WIDTH // 2, config.GRID_HEIGHT // 2
        self.body = [(cx, cy), (cx - 1, cy), (cx - 2, cy)]
        self.direction = config.RIGHT
        self.grow = 0
        self.alive = True

    def set_direction(self, new_dir):
        if (new_dir[0] == -self.direction[0] and new_dir[0] != 0) or \
           (new_dir[1] == -self.direction[1] and new_dir[1] != 0):
            return
        self.direction = new_dir

    def update(self):
        if not self.alive:
            return
        head_x, head_y = self.body[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)

        if not (0 <= new_head[0] < config.GRID_WIDTH and 0 <= new_head[1] < config.GRID_HEIGHT):
            self.alive = False
            return

        occupied = self.body if self.grow > 0 else self.body[:-1]
        if new_head in occupied:
            self.alive = False
            return

        self.body.insert(0, new_head)
        if self.grow > 0:
            self.grow -= 1
        else:
            self.body.pop()

    def feed(self):
        self.grow += 1

    def draw(self, surface):
        if self.body:
            draw_cell(surface, self.body[0], config.SNAKE_HEAD_COLOR)
            for segment in self.body[1:]:
                draw_cell(surface, segment, config.SNAKE_COLOR)

class Food:
    def __init__(self, occupied, rng=None):
        self.rng = rng
        self.pos = random_empty_cell(occupied, rng=self.rng)

    def respawn(self, occupied):
        self.pos = random_empty_cell(occupied, rng=self.rng)

    def draw(self, surface):
        draw_cell(surface, self.pos, config.FOOD_COLOR, inset=4)
