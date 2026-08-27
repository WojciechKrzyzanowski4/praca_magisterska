import numpy as np

from networks.layer.base import BaseLayer
from networks.parameter import Parameter


class DenseLayer(BaseLayer):
    def __init__(self, input_size, output_size):
        if input_size < 1 or output_size < 1:
            raise ValueError("input_size and output_size must be positive")
        scale = np.sqrt(2.0 / (input_size + output_size))
        self.weight = Parameter(
            np.random.randn(input_size, output_size) * scale
        )
        self.bias = Parameter(np.zeros(output_size))
        self.cached_input = None
        self.input_was_1d = False

    def forward(self, x):
        x = np.asarray(x)
        if x.ndim not in (1, 2):
            raise ValueError("DenseLayer expects a vector or a batch of vectors")

        self.input_was_1d = x.ndim == 1
        x_batch = x.reshape(1, -1) if self.input_was_1d else x
        if x_batch.shape[1] != self.weight.data.shape[0]:
            raise ValueError(
                f"Expected {self.weight.data.shape[0]} input features, "
                f"received {x_batch.shape[1]}"
            )

        self.cached_input = x_batch
        output = x_batch @ self.weight.data + self.bias.data
        return output[0] if self.input_was_1d else output

    def backward(self, grad_output):
        if self.cached_input is None:
            raise RuntimeError("forward must be called before backward")

        grad_output = np.asarray(grad_output)
        grad_batch = grad_output.reshape(1, -1) if self.input_was_1d else grad_output
        if grad_batch.shape != (
            self.cached_input.shape[0],
            self.weight.data.shape[1],
        ):
            raise ValueError("grad_output shape does not match the layer output")

        self.weight.grad[...] = self.cached_input.T @ grad_batch
        self.bias.grad[...] = np.sum(grad_batch, axis=0)

        grad_input = grad_batch @ self.weight.data.T
        return grad_input[0] if self.input_was_1d else grad_input

    def parameters(self):
        return [self.weight, self.bias]
