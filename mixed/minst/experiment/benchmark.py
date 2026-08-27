from __future__ import annotations

import numpy as np

from ..minst_model import SimpleCNN, save_model
from .manifest import build_benchmark_manifest
from .methods import evaluate_model, robustness_rows, train_evolution, train_gradient
from .output import OutputPaths, save_frame, save_json
from .plots import METHODS, plot_benchmark
from .setup import BenchmarkConfig
from .splits import prepare_benchmark_splits


def finalize_benchmark(
    config: BenchmarkConfig,
    paths: OutputPaths,
    *,
    generated_from_training: bool,
) -> dict[str, object]:
    figures = plot_benchmark(paths)
    manifest = build_benchmark_manifest(
        config,
        paths,
        generated_from_training=generated_from_training,
        figures=figures,
    )
    save_json(paths.root / "manifest.json", manifest)
    return manifest


def rebuild_saved_benchmark() -> dict[str, object]:
    config = BenchmarkConfig()
    paths = OutputPaths.from_root(config.output_directory)
    paths.prepare()
    return finalize_benchmark(config, paths, generated_from_training=False)


def run_benchmark() -> dict[str, object]:
    config = BenchmarkConfig()
    paths = OutputPaths.from_root(config.output_directory)
    paths.prepare()
    summary_rows: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    robustness: list[dict[str, object]] = []

    for seed_index, seed in enumerate(config.seeds):
        print(f"[{seed_index + 1}/{len(config.seeds)}] seed={seed}", flush=True)
        optimization, final_test = prepare_benchmark_splits(config, seed)
        gd_model, gd_trace, gd_time = train_gradient(
            optimization,
            seed=seed,
            epochs=config.gd_epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
        )
        evo_random, evo_trace, evo_time = train_evolution(
            optimization,
            seed=seed,
            config=config,
            initial_model=None,
            mutation_scope="all",
            mutation_std=config.random_mutation_std,
        )
        gd_continued, continued_trace, continued_time = train_gradient(
            optimization,
            seed=seed + 500,
            epochs=config.gd_extra_epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            initial_model=gd_model,
        )
        hybrid, hybrid_trace, hybrid_time = train_evolution(
            optimization,
            seed=seed,
            config=config,
            initial_model=gd_model,
            mutation_scope="classifier",
            mutation_std=config.warm_mutation_std,
        )
        models: dict[str, tuple[SimpleCNN, list[dict[str, float]], float]] = {
            "GD": (gd_model, gd_trace, gd_time),
            "EVO-random": (evo_random, evo_trace, evo_time),
            "GD+GD": (gd_continued, continued_trace, gd_time + continued_time),
            "GD+EVO": (hybrid, hybrid_trace, gd_time + hybrid_time),
        }
        confusion_payload: dict[str, np.ndarray] = {}
        prediction_payload: dict[str, np.ndarray] = {
            "labels": final_test.y_test[:120],
            "images": final_test.x_test[:120],
        }

        for method, (model, trace, training_time) in models.items():
            result = evaluate_model(model, final_test, config.batch_size)
            summary_rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "test_accuracy": result["test_accuracy"],
                    "test_loss": result["test_loss"],
                    "mean_confidence": result["mean_confidence"],
                    "training_time": training_time,
                }
            )
            for digit, accuracy in enumerate(result["class_accuracy"]):
                class_rows.append(
                    {"seed": seed, "method": method, "digit": digit, "accuracy": accuracy}
                )
            for point in trace:
                trace_rows.append(
                    {
                        "seed": seed,
                        "method": method,
                        "step": point["step"],
                        "progress": point["step"] / max(1, len(trace)),
                        "loss": point["loss"],
                        "validation_accuracy": point["validation_accuracy"],
                        "elapsed": point["elapsed"],
                    }
                )
            robustness.extend(robustness_rows(model, final_test, method, seed))
            confusion_payload[method] = result["confusion"]
            prediction_payload[f"pred_{method}"] = result["predictions"][:120]
            prediction_payload[f"conf_{method}"] = result["confidences"][:120]
            checkpoint_name = method.replace("+", "_").replace("-", "_")
            save_model(model, paths.checkpoints / f"{checkpoint_name}_{seed}.npz")

        np.savez(paths.root / f"confusion_seed_{seed}.npz", **confusion_payload)
        np.savez_compressed(
            paths.root / f"predictions_seed_{seed}.npz", **prediction_payload
        )

    save_frame(paths.root / "summary.csv", summary_rows)
    save_frame(paths.root / "traces.csv", trace_rows)
    save_frame(paths.root / "class_accuracy.csv", class_rows)
    save_frame(paths.root / "robustness.csv", robustness)
    return finalize_benchmark(config, paths, generated_from_training=True)
