from __future__ import annotations

from pathlib import Path

import numpy as np

from networks.layer.dense import DenseLayer
from networks.layer.tanh import TanhLayer
from networks.sequential import Sequential


class FullyConnectedRegressor(Sequential):
    """Tanh MLP built from the local NumPy neural-network components."""

    def __init__(
        self,
        depth: int,
        width: int,
        input_size: int = 1,
        output_size: int = 1,
    ) -> None:
        if depth < 1:
            raise ValueError("depth must be at least 1")
        if width < 1:
            raise ValueError("width must be at least 1")

        self.depth = depth
        self.width = width
        self.input_size = input_size
        self.output_size = output_size

        layers = [DenseLayer(input_size, width), TanhLayer()]
        for _ in range(depth - 1):
            layers.extend([DenseLayer(width, width), TanhLayer()])
        layers.append(DenseLayer(width, output_size))

        super().__init__(*layers)

    def flat_parameters(self) -> np.ndarray:
        parameters = [
            parameter.data.reshape(-1)
            for parameter in self.parameters()
        ]
        if not parameters:
            return np.array([], dtype=np.float64)
        return np.concatenate(parameters)


class FullyConnectedSurfaceRegressor(FullyConnectedRegressor):
    """MLP for surface regression: (x1, x2) -> z."""

    def __init__(self, depth: int, width: int) -> None:
        super().__init__(depth=depth, width=width, input_size=2, output_size=1)


def save_model(model: FullyConnectedRegressor, path: str | Path) -> str:
    """Save model architecture and parameters in one portable NumPy checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "depth": np.asarray(model.depth, dtype=np.int64),
        "width": np.asarray(model.width, dtype=np.int64),
        "input_size": np.asarray(model.input_size, dtype=np.int64),
        "output_size": np.asarray(model.output_size, dtype=np.int64),
    }
    payload.update(
        {
            f"parameter_{index}": parameter.data
            for index, parameter in enumerate(model.parameters())
        }
    )
    with path.open("wb") as checkpoint:
        np.savez_compressed(checkpoint, **payload)
    return str(path)


def load_model(path: str | Path) -> FullyConnectedRegressor:
    """Restore a model without requiring its architecture to be supplied again."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    with np.load(path) as checkpoint:
        model = FullyConnectedRegressor(
            depth=int(checkpoint["depth"]),
            width=int(checkpoint["width"]),
            input_size=int(checkpoint["input_size"]),
            output_size=int(checkpoint["output_size"]),
        )
        parameters = model.parameters()
        expected = {
            "depth",
            "width",
            "input_size",
            "output_size",
            *(f"parameter_{index}" for index in range(len(parameters))),
        }
        if set(checkpoint.files) != expected:
            raise ValueError("checkpoint does not match the saved model architecture")
        for index, parameter in enumerate(parameters):
            saved = checkpoint[f"parameter_{index}"]
            if saved.shape != parameter.data.shape:
                raise ValueError("checkpoint parameter shape does not match the model")
            parameter.data[...] = saved
    return model
