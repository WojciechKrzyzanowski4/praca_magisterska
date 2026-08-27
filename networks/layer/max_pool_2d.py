import numpy as np

from networks.layer.base import BaseLayer


class MaxPool2DLayer(BaseLayer):
    def __init__(self, kernel_size=2, stride=2):
        if not isinstance(kernel_size, int) or kernel_size < 1:
            raise ValueError("kernel_size must be a positive integer")
        if not isinstance(stride, int) or stride < 1:
            raise ValueError("stride must be a positive integer")

        self.kernel_size = kernel_size
        self.stride = stride
        self.input_shape = None
        self.maximum_indices = None

    def forward(self, x):
        x = np.asarray(x)
        if x.ndim != 4:
            raise ValueError(
                "MaxPool2DLayer expects an NCHW tensor with four dimensions"
            )

        batch_size, channels, input_height, input_width = x.shape
        if self.kernel_size > input_height or self.kernel_size > input_width:
            raise ValueError("kernel_size cannot be larger than the input")

        output_height = (
            input_height - self.kernel_size
        ) // self.stride + 1
        output_width = (
            input_width - self.kernel_size
        ) // self.stride + 1

        output = np.empty(
            (batch_size, channels, output_height, output_width),
            dtype=x.dtype,
        )
        maximum_indices = np.empty(output.shape, dtype=np.intp)

        for output_row in range(output_height):
            input_row = output_row * self.stride
            for output_column in range(output_width):
                input_column = output_column * self.stride
                window = x[
                    :,
                    :,
                    input_row:input_row + self.kernel_size,
                    input_column:input_column + self.kernel_size,
                ]
                flattened_window = window.reshape(
                    batch_size,
                    channels,
                    -1,
                )
                indices = np.argmax(flattened_window, axis=2)

                output[:, :, output_row, output_column] = (
                    np.take_along_axis(
                        flattened_window,
                        indices[:, :, None],
                        axis=2,
                    )[:, :, 0]
                )
                maximum_indices[:, :, output_row, output_column] = indices

        self.input_shape = x.shape
        self.maximum_indices = maximum_indices
        return output

    def backward(self, grad_output):
        if self.input_shape is None or self.maximum_indices is None:
            raise RuntimeError("forward must be called before backward")

        grad_output = np.asarray(grad_output)
        if grad_output.shape != self.maximum_indices.shape:
            raise ValueError("grad_output shape does not match the layer output")

        batch_size, channels = self.input_shape[:2]
        output_height, output_width = grad_output.shape[2:]
        grad_input = np.zeros(self.input_shape, dtype=grad_output.dtype)
        batch_indices = np.arange(batch_size)[:, None]
        channel_indices = np.arange(channels)[None, :]

        for output_row in range(output_height):
            input_row = output_row * self.stride
            for output_column in range(output_width):
                input_column = output_column * self.stride
                indices = self.maximum_indices[
                    :,
                    :,
                    output_row,
                    output_column,
                ]
                row_offsets = indices // self.kernel_size
                column_offsets = indices % self.kernel_size

                np.add.at(
                    grad_input,
                    (
                        batch_indices,
                        channel_indices,
                        input_row + row_offsets,
                        input_column + column_offsets,
                    ),
                    grad_output[:, :, output_row, output_column],
                )

        return grad_input

    def parameters(self):
        return []
