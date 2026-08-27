from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from networks.layer import (
    Conv2DLayer,
    DenseLayer,
    FlattenLayer,
    MaxPool2DLayer,
    ReLULayer,
)
from networks.sequential import Sequential


@dataclass
class Config:
    batch_size: int = 64
    epochs: int = 5
    lr: float = 1e-3
    seed: int = 42


class SimpleCNN(Sequential):
    def __init__(self):
        super().__init__(
            Conv2DLayer(1, 16, kernel_size=3, padding=1),
            ReLULayer(),
            MaxPool2DLayer(kernel_size=2, stride=2),
            Conv2DLayer(16, 32, kernel_size=3, padding=1),
            ReLULayer(),
            MaxPool2DLayer(kernel_size=2, stride=2),
            FlattenLayer(),
            DenseLayer(32 * 7 * 7, 128),
            ReLULayer(),
            DenseLayer(128, 10),
        )

    def classifier_parameters(self):
        return self.layers[7].parameters() + self.layers[9].parameters()


def resolve_device(device: str | None = None) -> str:
    if device not in (None, "auto", "cpu"):
        raise ValueError("The NumPy implementation supports CPU execution only")
    return "cpu"


def probabilities_from_logits(logits):
    logits = np.asarray(logits)
    if logits.ndim != 2:
        raise ValueError("logits must have shape (batch, classes)")

    shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted_logits)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def predict_images(
    model: SimpleCNN,
    images,
    *,
    device: str | None = None,
):
    resolve_device(device)
    images = np.asarray(images)
    if images.ndim == 2:
        images = images[None, :, :]
    if images.ndim != 3:
        raise ValueError("images must have shape (batch, height, width)")

    logits = model.forward(images[:, None, :, :])
    probabilities = probabilities_from_logits(logits)
    predictions = np.argmax(probabilities, axis=1)
    confidences = probabilities[np.arange(len(predictions)), predictions]
    return predictions, confidences


def save_model(model: SimpleCNN, path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    parameters = {
        f"parameter_{index}": parameter.data
        for index, parameter in enumerate(model.parameters())
    }
    with path.open("wb") as checkpoint:
        np.savez(checkpoint, **parameters)
    return str(path)


def load_model(path) -> SimpleCNN:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    model = SimpleCNN()
    with np.load(path) as checkpoint:
        parameters = model.parameters()
        expected_files = {f"parameter_{index}" for index in range(len(parameters))}
        if set(checkpoint.files) != expected_files:
            raise ValueError("checkpoint parameter count does not match SimpleCNN")
        for index, parameter in enumerate(parameters):
            saved = checkpoint[f"parameter_{index}"]
            if saved.shape != parameter.data.shape:
                raise ValueError("checkpoint parameter shape does not match SimpleCNN")
            parameter.data[...] = saved
    return model
