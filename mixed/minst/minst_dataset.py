from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from typing import Optional

import numpy as np


MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
DATA_PATH = os.path.join(DATA_DIR, "mnist.npz")
DEFAULT_VIZ_PATH = os.path.join(OUTPUT_DIR, "mnist_samples.png")
DEFAULT_OUTPUT_DIR = OUTPUT_DIR
DEFAULT_GRAD_OUTPUT_FILE = "mnist_lab_gd.npz"
DEFAULT_EVO_OUTPUT_FILE = "mnist_lab_evo.npz"


@dataclass
class MNISTBundle:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def ensure_directories() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def download_mnist(force: bool = False, url: str = MNIST_URL, path: str = DATA_PATH) -> str:
    ensure_directories()
    if os.path.isfile(path) and not force:
        return path
    urllib.request.urlretrieve(url, path)
    return path


def load_mnist(
    *,
    normalize: bool = True,
    flatten: bool = False,
    train_limit: Optional[int] = None,
    test_limit: Optional[int] = None,
    path: str = DATA_PATH,
) -> MNISTBundle:
    if not os.path.isfile(path):
        download_mnist(path=path)

    with np.load(path) as data:
        x_train = data["x_train"].astype(np.float32)
        y_train = data["y_train"].astype(np.int64)
        x_test = data["x_test"].astype(np.float32)
        y_test = data["y_test"].astype(np.int64)

    if train_limit is not None:
        x_train = x_train[:train_limit]
        y_train = y_train[:train_limit]
    if test_limit is not None:
        x_test = x_test[:test_limit]
        y_test = y_test[:test_limit]

    if normalize:
        x_train /= 255.0
        x_test /= 255.0

    if flatten:
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)

    return MNISTBundle(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
    )


def dataset_summary(bundle: MNISTBundle) -> dict:
    def split_info(images: np.ndarray, labels: np.ndarray) -> dict:
        counts = np.bincount(labels, minlength=10)
        return {
            "shape": tuple(images.shape),
            "dtype": str(images.dtype),
            "min": float(images.min()),
            "max": float(images.max()),
            "label_counts": counts.tolist(),
        }

    return {
        "train": split_info(bundle.x_train, bundle.y_train),
        "test": split_info(bundle.x_test, bundle.y_test),
    }


def sample_visualization_indices(total: int, *, count: int = 16, seed: int = 42) -> np.ndarray:
    if total <= 0:
        raise ValueError("Cannot visualize an empty split")

    sample_count = min(max(1, int(count)), total)
    rng = np.random.default_rng(seed)
    return rng.choice(total, size=sample_count, replace=False)


def save_sample_visualization(
    bundle: MNISTBundle,
    *,
    split: str = "train",
    count: int = 16,
    seed: int = 42,
    output_path: str | None = DEFAULT_VIZ_PATH,
    show: bool = False,
    indices: np.ndarray | None = None,
    predicted_labels: np.ndarray | None = None,
    predicted_confidences: np.ndarray | None = None,
) -> str:
    import matplotlib.pyplot as plt

    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")

    images = bundle.x_train if split == "train" else bundle.x_test
    labels = bundle.y_train if split == "train" else bundle.y_test
    if images.ndim != 3:
        raise ValueError("Visualization expects unflattened images with shape (N, 28, 28)")

    ensure_directories()
    output_path = output_path or DEFAULT_VIZ_PATH
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    indices = (
        sample_visualization_indices(len(images), count=count, seed=seed)
        if indices is None
        else np.asarray(indices, dtype=np.int64)
    )

    if predicted_labels is not None:
        predicted_labels = np.asarray(predicted_labels, dtype=np.int64)
        if len(predicted_labels) != len(indices):
            raise ValueError("predicted_labels must match the number of visualized indices")
    if predicted_confidences is not None:
        predicted_confidences = np.asarray(predicted_confidences, dtype=np.float32)
        if len(predicted_confidences) != len(indices):
            raise ValueError("predicted_confidences must match the number of visualized indices")

    cols = int(np.ceil(np.sqrt(len(indices))))
    rows = int(np.ceil(len(indices) / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.0))
    axes = np.atleast_1d(axes).ravel()

    for position, (ax, idx) in enumerate(zip(axes, indices)):
        true_label = int(labels[idx])
        ax.imshow(images[idx], cmap="gray")
        if predicted_labels is None:
            ax.set_title(f"label={true_label}", fontsize=10)
        else:
            pred_label = int(predicted_labels[position])
            confidence = ""
            if predicted_confidences is not None:
                confidence = f" ({predicted_confidences[position] * 100:.1f}%)"
            ax.set_title(
                f"true={true_label}\npred={pred_label}{confidence}",
                fontsize=10,
                color="darkgreen" if pred_label == true_label else "crimson",
            )
        ax.axis("off")

    for ax in axes[len(indices):]:
        ax.axis("off")

    fig.suptitle(f"MNIST sample grid ({split} split)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return output_path
