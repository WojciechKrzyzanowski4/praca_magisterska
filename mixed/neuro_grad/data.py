from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class RegressionData:
    x: np.ndarray
    y: np.ndarray
    clean_y: np.ndarray
    target_name: str
    input_dim: int
    grid_shape: tuple[int, int] | None = None


CURVE_FUNCTIONS: dict[str, str] = {
    "sin_pi": "sin(pi*x)",
    "gaussian_bump": "2*exp(-8*x^2)-1",
    "wave_trend": "0.65*sin(2*pi*x)+0.35*x",
    "tanh_cubic": "tanh(3*x)+0.25*x^3",
}

SURFACE_FUNCTIONS: dict[str, str] = {
    "ripple": "sin(pi*x1)*cos(pi*x2)",
    "bowl": "0.8*(x1^2+x2^2)-0.7",
    "saddle": "x1^2-x2^2",
    "gaussian_hill": "2*exp(-4*(x1^2+x2^2))-1",
    "mixed_surface": "0.5*sin(2*pi*x1)+0.35*cos(2*pi*x2)+0.25*x1*x2",
}

TARGET_FUNCTIONS: dict[str, str] = {
    **CURVE_FUNCTIONS,
    **SURFACE_FUNCTIONS,
}


def _curve_target_values(x: np.ndarray, target_function: str) -> np.ndarray:
    if target_function == "sin_pi":
        y = np.sin(np.pi * x)
    elif target_function == "gaussian_bump":
        y = 2.0 * np.exp(-8.0 * np.square(x)) - 1.0
    elif target_function == "wave_trend":
        y = 0.65 * np.sin(2.0 * np.pi * x) + 0.35 * x
    elif target_function == "tanh_cubic":
        y = np.tanh(3.0 * x) + 0.25 * np.power(x, 3)
    else:
        options = ", ".join(CURVE_FUNCTIONS)
        raise ValueError(f"Unknown curve target_function '{target_function}'. Available: {options}")

    return y.astype(np.float32)


def _surface_target_values(points: np.ndarray, target_function: str) -> np.ndarray:
    x1 = points[:, 0:1]
    x2 = points[:, 1:2]
    if target_function == "ripple":
        y = np.sin(np.pi * x1) * np.cos(np.pi * x2)
    elif target_function == "bowl":
        y = 0.8 * (np.square(x1) + np.square(x2)) - 0.7
    elif target_function == "saddle":
        y = np.square(x1) - np.square(x2)
    elif target_function == "gaussian_hill":
        y = 2.0 * np.exp(-4.0 * (np.square(x1) + np.square(x2))) - 1.0
    elif target_function == "mixed_surface":
        y = 0.5 * np.sin(2.0 * np.pi * x1) + 0.35 * np.cos(2.0 * np.pi * x2) + 0.25 * x1 * x2
    else:
        options = ", ".join(SURFACE_FUNCTIONS)
        raise ValueError(f"Unknown surface target_function '{target_function}'. Available: {options}")

    return y.astype(np.float32)


def generate_data(
    n_points: int = 200,
    x_min: float = -1.0,
    x_max: float = 1.0,
    noise_std: float = 0.0,
    seed: int = 42,
    target_function: str = "sin_pi",
    experiment_type: Literal["curve", "surface"] = "curve",
    grid_size: int = 35,
) -> RegressionData:
    """Generate synthetic regression data with optional Gaussian noise."""
    if n_points < 2:
        raise ValueError("n_points must be at least 2")
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    if x_min >= x_max:
        raise ValueError("x_min must be smaller than x_max")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")

    rng = np.random.default_rng(seed)
    if experiment_type == "curve":
        x = np.linspace(x_min, x_max, n_points, dtype=np.float32).reshape(-1, 1)
        clean_y = _curve_target_values(x, target_function)
        input_dim = 1
        grid_shape = None
    elif experiment_type == "surface":
        axis = np.linspace(x_min, x_max, grid_size, dtype=np.float32)
        x1_grid, x2_grid = np.meshgrid(axis, axis)
        x = np.column_stack((x1_grid.reshape(-1), x2_grid.reshape(-1))).astype(np.float32)
        clean_y = _surface_target_values(x, target_function)
        input_dim = 2
        grid_shape = (grid_size, grid_size)
    else:
        raise ValueError("experiment_type must be 'curve' or 'surface'")

    if noise_std == 0:
        y = clean_y.copy()
    else:
        noise = rng.normal(0.0, noise_std, size=clean_y.shape).astype(np.float32)
        y = clean_y + noise
    return RegressionData(
        x=x,
        y=y.astype(np.float32),
        clean_y=clean_y,
        target_name=TARGET_FUNCTIONS[target_function],
        input_dim=input_dim,
        grid_shape=grid_shape,
    )
