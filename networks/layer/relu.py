import numpy as np

from networks.layer.base import BaseLayer


class ReLULayer(BaseLayer):
    def __init__(self):
        self.positive_mask = None

    def forward(self, x):
        x = np.asarray(x)
        self.positive_mask = x > 0
        return np.maximum(x, 0)

    def backward(self, grad_output):
        if self.positive_mask is None:
            raise RuntimeError("forward must be called before backward")

        grad_output = np.asarray(grad_output)
        if grad_output.shape != self.positive_mask.shape:
            raise ValueError("grad_output shape does not match the layer output")

        return grad_output * self.positive_mask

    def parameters(self):
        return []
