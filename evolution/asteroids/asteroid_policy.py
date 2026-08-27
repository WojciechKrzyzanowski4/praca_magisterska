from typing import Tuple

import numpy as np


class FullyConnectedPolicy:
    """A compact MLP: input -> tanh(hidden) -> sigmoid(4 outputs).
    Outputs map to booleans: [rotate_left, rotate_right, thrust, shoot].
    """
    def __init__(self, input_size: int, hidden_size: int = 48):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = 4
        self.size_w1 = self.hidden_size * self.input_size
        self.size_b1 = self.hidden_size
        self.size_w2 = self.output_size * self.hidden_size
        self.size_b2 = self.output_size
        self.total_param_count = self.size_w1 + self.size_b1 + self.size_w2 + self.size_b2

    def unpack(self, flat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        i, h, o = self.input_size, self.hidden_size, self.output_size
        idx = 0
        w1 = flat[idx: idx + h*i].reshape(h, i); idx += h*i
        b1 = flat[idx: idx + h]; idx += h
        w2 = flat[idx: idx + o*h].reshape(o, h); idx += o*h
        b2 = flat[idx: idx + o]; idx += o
        return w1, b1, w2, b2

    def forward(self, flat: np.ndarray, x: np.ndarray) -> np.ndarray:
        w1, b1, w2, b2 = self.unpack(flat)
        h = np.tanh(w1 @ x + b1)
        z = w2 @ h + b2
        return 1.0 / (1.0 + np.exp(-z))  # sigmoid

    def output_to_control(self, y: np.ndarray) -> dict:
        return {
            "rotate_left":  bool(y[0] > 0.55 and y[0] > y[1] + 0.05),
            "rotate_right": bool(y[1] > 0.55 and y[1] > y[0] + 0.05),
            "thrust":       bool(y[2] > 0.50),
            "shoot":        bool(y[3] > 0.60),
        }
