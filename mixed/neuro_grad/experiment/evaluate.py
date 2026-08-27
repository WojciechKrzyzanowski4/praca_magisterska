from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ExperimentFrames:
    runs: pd.DataFrame
    traces: pd.DataFrame
    paired: pd.DataFrame
    summary: pd.DataFrame


def aggregate_paired(runs: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "scenario",
        "variant",
        "replicate",
        "method",
        "target_function",
        "noise_std",
        "depth",
        "width",
        "parameter_count",
    ]
    numeric = [
        "objective",
        "train_mse",
        "clean_mse",
        "test_mse",
        "violation_rate",
        "training_time_s",
        "objective_evaluations",
        "acceptance_rate",
    ]
    return runs.groupby(keys, dropna=False, as_index=False)[numeric].mean()


def summarize(paired: pd.DataFrame) -> pd.DataFrame:
    return (
        paired.groupby(["scenario", "variant", "method"], as_index=False)
        .agg(
            n=("replicate", "count"),
            test_mse_mean=("test_mse", "mean"),
            test_mse_std=("test_mse", "std"),
            objective_mean=("objective", "mean"),
            violation_rate_mean=("violation_rate", "mean"),
            training_time_mean_s=("training_time_s", "mean"),
        )
    )


def build_frames(
    runs: list[dict[str, object]],
    traces: list[dict[str, object]],
) -> ExperimentFrames:
    runs_frame = pd.DataFrame(runs)
    traces_frame = pd.DataFrame(traces)
    paired_frame = aggregate_paired(runs_frame)
    return ExperimentFrames(
        runs=runs_frame,
        traces=traces_frame,
        paired=paired_frame,
        summary=summarize(paired_frame),
    )


def load_frames(output_directory) -> ExperimentFrames:
    runs = pd.read_csv(output_directory / "runs.csv")
    traces = pd.read_csv(output_directory / "traces.csv")
    paired = pd.read_csv(output_directory / "paired_runs.csv")
    summary = pd.read_csv(output_directory / "tables" / "summary.csv")
    return ExperimentFrames(runs, traces, paired, summary)
