import math
import random

import pygame

from .asteroid_constants import config
from .asteroid_utils import wrap


class Ship:
    def __init__(self, pos: config.Vec2):
        self.pos = config.Vec2(pos)
        self.vel = config.Vec2(0, 0)
        self.angle = -90
        self.cooldown = 0.0
        self.invincibility = 1.2

    def update(self, dt: float, keys):
        if keys[pygame.K_LEFT]:
            self.angle -= config.SHIP_ROT_SPEED * dt
        if keys[pygame.K_RIGHT]:
            self.angle += config.SHIP_ROT_SPEED * dt
        if keys[pygame.K_UP]:
            thrust_vec = config.Vec2(1, 0).rotate(self.angle)
            self.vel += thrust_vec * config.SHIP_THRUST * dt
        self.vel *= config.SHIP_FRICTION
        self.pos += self.vel * dt
        self.pos = wrap(self.pos)
        self.cooldown = max(0.0, self.cooldown - dt)
        self.invincibility = max(0.0, self.invincibility - dt)

    def can_shoot(self) -> bool:
        return self.cooldown <= 0.0

    def shoot(self):
        self.cooldown = config.SHOOT_COOLDOWN
        dir_vec = config.Vec2(1, 0).rotate(self.angle)
        bullet_vel = self.vel + dir_vec * config.BULLET_SPEED
        tip = self.pos + dir_vec * (config.SHIP_RADIUS + 6)
        return Bullet(tip, bullet_vel)

    def shape(self):
        dir_vec = config.Vec2(1, 0).rotate(self.angle)
        left = config.Vec2(1, 0).rotate(self.angle + 140)
        right = config.Vec2(1, 0).rotate(self.angle - 140)
        return [self.pos + dir_vec * config.SHIP_RADIUS,
                self.pos + left * (config.SHIP_RADIUS * 0.9),
                self.pos + right * (config.SHIP_RADIUS * 0.9)]

    def draw(self, surf):
        pts = self.shape()
        color = config.COLOR_FG if self.invincibility <= 0 else config.COLOR_MUTED
        pygame.draw.polygon(surf, color, pts, 2)

    def update_with_controls(self, dt: float, rotate_left: bool, rotate_right: bool, thrust: bool):
        if rotate_left:
            self.angle -= config.SHIP_ROT_SPEED * dt
        if rotate_right:
            self.angle += config.SHIP_ROT_SPEED * dt
        if thrust:
            thrust_vec = config.Vec2(1, 0).rotate(self.angle)
            self.vel += thrust_vec * config.SHIP_THRUST * dt
        self.vel *= config.SHIP_FRICTION
        self.pos += self.vel * dt
        self.pos = wrap(self.pos)
        self.cooldown = max(0.0, self.cooldown - dt)
        self.invincibility = max(0.0, self.invincibility - dt)


class Bullet:
    def __init__(self, pos: config.Vec2, vel: config.Vec2):
        self.pos = config.Vec2(pos)
        self.vel = config.Vec2(vel)
        self.ttl = config.BULLET_TTL

    def update(self, dt: float):
        self.pos += self.vel * dt
        self.pos = wrap(self.pos)
        self.ttl -= dt

    def alive(self):
        return self.ttl > 0

    def draw(self, surf):
        pygame.draw.circle(surf, config.COLOR_FG, self.pos, 2)


class Asteroid:
    def __init__(self, pos: config.Vec2, vel: config.Vec2, radius: float, vel_factory):
        self.pos = config.Vec2(pos)
        self.vel = config.Vec2(vel)
        self.radius = radius
        self.vel_factory = vel_factory
        k = max(8, int(self.radius / 2))
        self.pts = []
        for i in range(k):
            ang = (i / k) * math.tau
            jitter = random.uniform(0.85, 1.15)
            r = self.radius * jitter
            self.pts.append(config.Vec2(math.cos(ang), math.sin(ang)) * r)
        self.spin = random.uniform(-60, 60)
        self.angle = random.uniform(0, 360)

    def update(self, dt: float):
        self.pos += self.vel * dt
        self.pos = wrap(self.pos)
        self.angle += self.spin * dt

    def draw(self, surf):
        rot = [p.rotate(self.angle) + self.pos for p in self.pts]
        pygame.draw.polygon(surf, config.COLOR_FG, rot, 2)

    def split(self):
        if self.radius < config.SPLIT_MED_MIN:
            return []
        child_count = 2 if self.radius >= config.SPLIT_LARGE_MIN else 3
        r = max(config.AST_MIN_RADIUS, int(self.radius * config.AST_SPLIT_SCALE))
        pv = getattr(self, "vel", config.Vec2(0.0, 0.0))
        ps = pv.length()
        if ps < 1e-6:
            parent_heading_deg = 0.0
            ps = float(config.AST_SPEED_RANGE[0])
        else:
            parent_heading_deg = math.degrees(math.atan2(pv.y, pv.x))
        child_speed = ps * config.SPLIT_SPEED_MULT
        offsets = config.SPLIT_CHILD_ANGLE_OFFSETS_DEG[:child_count]
        children = []
        for off in offsets:
            head = parent_heading_deg + off
            v = config.Vec2(
                math.cos(math.radians(head)) * child_speed,
                math.sin(math.radians(head)) * child_speed
            )
            base_dir = v.normalize() if v.length() > 1e-6 else config.Vec2(1.0, 0.0)
            base_speed = child_speed
            def vf(extra=1.0, _bd=base_dir, _bs=base_speed):
                return _bd * (_bs * extra)
            children.append(Asteroid(self.pos, v, r, vf))
        return children
