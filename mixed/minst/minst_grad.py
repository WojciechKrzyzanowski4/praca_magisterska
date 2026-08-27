from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from networks.loss import CrossEntropyLoss
from networks.optimizer import Adam

from .minst_dataset import MNISTBundle
from .minst_model import SimpleCNN, load_model, save_model


@dataclass
class TrainingHistory:
    train_losses: list[float] = field(default_factory=list)
    test_accuracies: list[float] = field(default_factory=list)


def iterate_minibatches(images, labels, batch_size, rng, *, shuffle):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if len(images) != len(labels):
        raise ValueError("images and labels must contain the same number of samples")

    indices = np.arange(len(images))
    if shuffle:
        rng.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]
        yield images[batch_indices, None, :, :], labels[batch_indices]


def train_one_epoch(
    model,
    images,
    labels,
    optimizer,
    loss_function,
    batch_size,
    rng,
):
    total_loss = 0.0
    total_examples = 0

    for batch_images, batch_labels in iterate_minibatches(
        images,
        labels,
        batch_size,
        rng,
        shuffle=True,
    ):
        logits = model.forward(batch_images)
        loss = loss_function.forward(logits, batch_labels)
        model.backward(loss_function.backward())
        optimizer.step()
        optimizer.zero_grad()

        total_loss += float(loss) * len(batch_images)
        total_examples += len(batch_images)

    return total_loss / max(1, total_examples)


def evaluate(model, images, labels, batch_size=128):
    correct = 0
    total = 0
    rng = np.random.default_rng(0)

    for batch_images, batch_labels in iterate_minibatches(
        images,
        labels,
        batch_size,
        rng,
        shuffle=False,
    ):
        logits = model.forward(batch_images)
        predictions = np.argmax(logits, axis=1)
        correct += int(np.sum(predictions == batch_labels))
        total += len(batch_labels)

    return correct / max(1, total)


def evaluate_loss(model, images, labels, batch_size=128) -> float:
    loss_function = CrossEntropyLoss()
    total_loss = 0.0
    total_examples = 0
    rng = np.random.default_rng(0)

    for batch_images, batch_labels in iterate_minibatches(
        images,
        labels,
        batch_size,
        rng,
        shuffle=False,
    ):
        loss = loss_function.forward(model.forward(batch_images), batch_labels)
        total_loss += float(loss) * len(batch_labels)
        total_examples += len(batch_labels)

    return total_loss / max(1, total_examples)


def train_model(
    bundle: MNISTBundle,
    *,
    epochs=3,
    batch_size=64,
    learning_rate=1e-3,
    seed=42,
    callback=None,
):
    if epochs < 1:
        raise ValueError("epochs must be positive")

    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    model = SimpleCNN()
    loss_function = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), learning_rate=learning_rate)
    history = TrainingHistory()

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model,
            bundle.x_train,
            bundle.y_train,
            optimizer,
            loss_function,
            batch_size,
            rng,
        )
        test_accuracy = evaluate(
            model,
            bundle.x_test,
            bundle.y_test,
            batch_size=batch_size,
        )
        history.train_losses.append(train_loss)
        history.test_accuracies.append(test_accuracy)
        if callback is not None:
            callback(epoch, train_loss, test_accuracy, model)

    return model, history


def load_grad_model(checkpoint_path: str, device: str | None = None):
    if device not in (None, "auto", "cpu"):
        raise ValueError("The NumPy implementation supports CPU execution only")
    return load_model(checkpoint_path), "cpu"


__all__ = [
    "TrainingHistory",
    "evaluate",
    "evaluate_loss",
    "iterate_minibatches",
    "load_grad_model",
    "save_model",
    "train_model",
    "train_one_epoch",
]
