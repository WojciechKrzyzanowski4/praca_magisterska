from __future__ import annotations

import json
import time

from .evaluate import ExperimentFrames, build_frames, load_frames
from .execute import run_changed_objective, run_standard_job
from .jobs import standard_jobs
from .manifest import build_manifest
from .output import OutputPaths, save_frame, save_json
from .plots import make_plots
from .setup import StaticConfig


def save_frames(frames: ExperimentFrames, paths: OutputPaths) -> None:
    save_frame(paths.root / "runs.csv", frames.runs)
    save_frame(paths.root / "traces.csv", frames.traces)
    save_frame(paths.root / "paired_runs.csv", frames.paired)
    save_frame(paths.tables / "summary.csv", frames.summary)


def finalize(
    config: StaticConfig,
    paths: OutputPaths,
    frames: ExperimentFrames,
    *,
    duration_seconds: float | None,
    generated_from_training: bool,
) -> dict[str, object]:
    figure_paths = make_plots(frames, paths)
    artifact_files = sorted(
        str(path.relative_to(paths.root))
        for path in paths.root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    manifest = build_manifest(
        config,
        frames,
        duration_seconds=duration_seconds,
        generated_from_training=generated_from_training,
        figure_paths=figure_paths,
        artifact_files=artifact_files,
    )
    save_json(paths.root / "manifest.json", manifest)
    return manifest


def rebuild_saved_artifacts() -> dict[str, object]:
    """Odtwarza agregacje, wykresy i manifest z zapisanych wyników."""

    config = StaticConfig()
    paths = OutputPaths.from_config(config)
    paths.prepare()
    frames = load_frames(paths.root)
    return finalize(
        config,
        paths,
        frames,
        duration_seconds=None,
        generated_from_training=False,
    )


def run_experiment() -> dict[str, object]:
    """Wykonuje pełny statyczny eksperyment regresji od inicjalizacji modeli."""

    config = StaticConfig()
    paths = OutputPaths.from_config(config)
    paths.prepare()
    all_runs: list[dict[str, object]] = []
    all_traces: list[dict[str, object]] = []
    started = time.perf_counter()
    jobs = standard_jobs(config)

    for index, job in enumerate(jobs, start=1):
        print(f"[{index}/{len(jobs) + 1}] {job.scenario}/{job.variant}", flush=True)
        runs, traces = run_standard_job(
            job,
            paths,
            trace_interval=config.trace_interval,
        )
        all_runs.extend(runs)
        all_traces.extend(traces)

    print(f"[{len(jobs) + 1}/{len(jobs) + 1}] objective/threshold", flush=True)
    runs, traces = run_changed_objective(config, paths)
    all_runs.extend(runs)
    all_traces.extend(traces)

    frames = build_frames(all_runs, all_traces)
    save_frames(frames, paths)
    return finalize(
        config,
        paths,
        frames,
        duration_seconds=time.perf_counter() - started,
        generated_from_training=True,
    )


def main() -> None:
    manifest = run_experiment()
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
