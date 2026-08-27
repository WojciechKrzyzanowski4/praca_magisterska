import numpy as np

from networks.layer.base import BaseLayer


class TanhLayer(BaseLayer):
    def __init__(self):
        self.cached_output = None

    def forward(self, x):
        self.cached_output = np.tanh(np.asarray(x))
        return self.cached_output

    def backward(self, grad_output):
        if self.cached_output is None:
            raise RuntimeError("forward must be called before backward")

        grad_output = np.asarray(grad_output)
        if grad_output.shape != self.cached_output.shape:
            raise ValueError("grad_output shape does not match the layer output")

        local_gradient = 1.0 - self.cached_output ** 2
        return grad_output * local_gradient

    def parameters(self):
        return []
