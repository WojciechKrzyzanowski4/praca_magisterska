from __future__ import annotations

import numpy as np

from .minst_dataset import (
    DEFAULT_EVO_OUTPUT_FILE,
    DEFAULT_GRAD_OUTPUT_FILE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_VIZ_PATH,
    dataset_summary,
    download_mnist,
    load_mnist,
    sample_visualization_indices,
    save_sample_visualization,
)
from .minst_evo import load_evo_model
from .minst_grad import load_grad_model
from .minst_model import predict_images



def prepare_dataset(force: bool = False) -> str:
    path = download_mnist(force=force)
    print(f"[DATA] MNIST available at: {path}")
    return path


def print_dataset_summary(
    *,
    normalize: bool = True,
    flatten: bool = False,
    train_limit: int | None = None,
    test_limit: int | None = None,
) -> None:
    bundle = load_mnist(
        normalize=normalize,
        flatten=flatten,
        train_limit=train_limit,
        test_limit=test_limit,
    )
    summary = dataset_summary(bundle)
    for split_name in ("train", "test"):
        split = summary[split_name]
        print(
            f"[{split_name.upper()}] shape={split['shape']} dtype={split['dtype']} "
            f"range=({split['min']:.3f}, {split['max']:.3f})"
        )
        print(f"[{split_name.upper()}] label_counts={split['label_counts']}")


def visualize_dataset(
    *,
    split: str = "train",
    count: int = 16,
    seed: int = 42,
    output_path: str | None = DEFAULT_VIZ_PATH,
    checkpoint_path: str | None = None,
    show: bool = False,
) -> str:
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")

    bundle = load_mnist(normalize=True, flatten=False)
    images = bundle.x_train if split == "train" else bundle.x_test
    labels = bundle.y_train if split == "train" else bundle.y_test
    indices = sample_visualization_indices(len(images), count=count, seed=seed)
    predicted_labels: np.ndarray | None = None
    predicted_confidences: np.ndarray | None = None

    if checkpoint_path is not None:
        model, _ = load_grad_model(checkpoint_path)
        predicted_labels, predicted_confidences = predict_images(
            model,
            images[indices],
        )
        correct = int((predicted_labels == labels[indices]).sum())
        print(
            f"[VIZ] Checkpoint guesses on sampled digits: "
            f"{correct}/{len(indices)} correct"
        )

    path = save_sample_visualization(
        bundle,
        split=split,
        count=count,
        seed=seed,
        output_path=output_path or DEFAULT_VIZ_PATH,
        show=show,
        indices=indices,
        predicted_labels=predicted_labels,
        predicted_confidences=predicted_confidences,
    )
    print(f"[VIZ] Wrote sample visualization to: {path}")
    return path
