import numpy as np

from networks.layer.base import BaseLayer


class FlattenLayer(BaseLayer):
    def __init__(self):
        self.input_shape = None

    def forward(self, x):
        x = np.asarray(x)
        if x.ndim != 4:
            raise ValueError(
                "FlattenLayer expects an NCHW tensor with four dimensions"
            )

        self.input_shape = x.shape
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1)

    def backward(self, grad_output):
        if self.input_shape is None:
            raise RuntimeError("forward must be called before backward")

        grad_output = np.asarray(grad_output)
        expected_shape = (
            self.input_shape[0],
            int(np.prod(self.input_shape[1:])),
        )
        if grad_output.shape != expected_shape:
            raise ValueError("grad_output shape does not match the layer output")

        return grad_output.reshape(self.input_shape)

    def parameters(self):
        return []
