from __future__ import annotations

import numpy as np

from ..minst_dataset import MNISTBundle, load_mnist
from .setup import BenchmarkConfig, ExtendedConfig


def stratified_indices(
    labels: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    selected: list[int] = []
    per_class = count // 10
    remainder = count % 10
    for label in range(10):
        candidates = np.flatnonzero(labels == label)
        take = per_class + int(label < remainder)
        selected.extend(rng.choice(candidates, size=take, replace=False).tolist())
    result = np.asarray(selected, dtype=np.int64)
    rng.shuffle(result)
    return result


def prepare_benchmark_splits(
    config: BenchmarkConfig,
    seed: int,
) -> tuple[MNISTBundle, MNISTBundle]:
    full = load_mnist(normalize=True, flatten=False)
    rng = np.random.default_rng(seed)
    train_indices = stratified_indices(full.y_train, config.train_size, rng)
    remaining_mask = np.ones(len(full.y_train), dtype=bool)
    remaining_mask[train_indices] = False
    remaining_indices = np.flatnonzero(remaining_mask)
    validation_local = stratified_indices(
        full.y_train[remaining_indices], config.validation_size, rng
    )
    validation_indices = remaining_indices[validation_local]
    test_indices = stratified_indices(full.y_test, config.test_size, rng)
    optimization = MNISTBundle(
        x_train=full.x_train[train_indices],
        y_train=full.y_train[train_indices],
        x_test=full.x_train[validation_indices],
        y_test=full.y_train[validation_indices],
    )
    final_test = MNISTBundle(
        x_train=optimization.x_train,
        y_train=optimization.y_train,
        x_test=full.x_test[test_indices],
        y_test=full.y_test[test_indices],
    )
    return optimization, final_test


def prepare_extended_bundle(config: ExtendedConfig) -> MNISTBundle:
    return load_mnist(
        normalize=True,
        flatten=False,
        train_limit=config.train_size,
        test_limit=config.monitor_size,
    )
