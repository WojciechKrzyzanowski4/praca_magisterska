import numpy as np

from networks.layer.base import BaseLayer
from networks.parameter import Parameter


class Conv2DLayer(BaseLayer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
    ):
        if not isinstance(in_channels, int) or in_channels < 1:
            raise ValueError("in_channels must be a positive integer")
        if not isinstance(out_channels, int) or out_channels < 1:
            raise ValueError("out_channels must be a positive integer")
        if not isinstance(kernel_size, int) or kernel_size < 1:
            raise ValueError("kernel_size must be a positive integer")
        if not isinstance(stride, int) or stride < 1:
            raise ValueError("stride must be a positive integer")
        if not isinstance(padding, int) or padding < 0:
            raise ValueError("padding must be a non-negative integer")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        fan_in = in_channels * kernel_size * kernel_size
        scale = np.sqrt(2.0 / fan_in)
        self.weight = Parameter(
            np.random.randn(
                out_channels,
                in_channels,
                kernel_size,
                kernel_size,
            ) * scale
        )
        self.bias = Parameter(np.zeros(out_channels))

        self.input_shape = None
        self.cached_columns = None
        self.output_shape = None

    def _output_size(self, input_size):
        padded_size = input_size + 2 * self.padding
        if padded_size < self.kernel_size:
            raise ValueError("kernel_size cannot be larger than the padded input")
        return (padded_size - self.kernel_size) // self.stride + 1

    def _input_to_columns(self, x):
        if self.padding > 0:
            x = np.pad(
                x,
                (
                    (0, 0),
                    (0, 0),
                    (self.padding, self.padding),
                    (self.padding, self.padding),
                ),
            )

        windows = np.lib.stride_tricks.sliding_window_view(
            x,
            (self.kernel_size, self.kernel_size),
            axis=(2, 3),
        )
        windows = windows[:, :, ::self.stride, ::self.stride, :, :]
        columns = windows.transpose(0, 2, 3, 1, 4, 5).reshape(
            -1,
            self.in_channels * self.kernel_size * self.kernel_size,
        )
        return columns

    def forward(self, x):
        x = np.asarray(x)
        if x.ndim != 4:
            raise ValueError(
                "Conv2DLayer expects an NCHW tensor with four dimensions"
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, "
                f"received {x.shape[1]}"
            )

        batch_size, _, input_height, input_width = x.shape
        output_height = self._output_size(input_height)
        output_width = self._output_size(input_width)
        columns = self._input_to_columns(x)
        flattened_weights = self.weight.data.reshape(
            self.out_channels,
            -1,
        )

        output_columns = columns @ flattened_weights.T + self.bias.data
        output = output_columns.reshape(
            batch_size,
            output_height,
            output_width,
            self.out_channels,
        ).transpose(0, 3, 1, 2)

        self.input_shape = x.shape
        self.cached_columns = columns
        self.output_shape = output.shape
        return output

    def backward(self, grad_output):
        if (
            self.input_shape is None
            or self.cached_columns is None
            or self.output_shape is None
        ):
            raise RuntimeError("forward must be called before backward")

        grad_output = np.asarray(grad_output)
        if grad_output.shape != self.output_shape:
            raise ValueError("grad_output shape does not match the layer output")

        batch_size, _, input_height, input_width = self.input_shape
        output_height, output_width = self.output_shape[2:]
        grad_output_columns = grad_output.transpose(0, 2, 3, 1).reshape(
            -1,
            self.out_channels,
        )
        flattened_weights = self.weight.data.reshape(
            self.out_channels,
            -1,
        )

        self.weight.grad[...] = (
            grad_output_columns.T @ self.cached_columns
        ).reshape(self.weight.data.shape)
        self.bias.grad[...] = np.sum(grad_output_columns, axis=0)

        grad_columns = grad_output_columns @ flattened_weights
        grad_windows = grad_columns.reshape(
            batch_size,
            output_height,
            output_width,
            self.in_channels,
            self.kernel_size,
            self.kernel_size,
        )
        padded_height = input_height + 2 * self.padding
        padded_width = input_width + 2 * self.padding
        grad_padded_input = np.zeros(
            (
                batch_size,
                self.in_channels,
                padded_height,
                padded_width,
            ),
            dtype=grad_output.dtype,
        )

        for kernel_row in range(self.kernel_size):
            row_end = kernel_row + self.stride * output_height
            for kernel_column in range(self.kernel_size):
                column_end = kernel_column + self.stride * output_width
                grad_padded_input[
                    :,
                    :,
                    kernel_row:row_end:self.stride,
                    kernel_column:column_end:self.stride,
                ] += grad_windows[
                    :,
                    :,
                    :,
                    :,
                    kernel_row,
                    kernel_column,
                ].transpose(0, 3, 1, 2)

        if self.padding == 0:
            return grad_padded_input
        return grad_padded_input[
            :,
            :,
            self.padding:-self.padding,
            self.padding:-self.padding,
        ]

    def parameters(self):
        return [self.weight, self.bias]
