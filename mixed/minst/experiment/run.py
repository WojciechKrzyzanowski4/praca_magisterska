from __future__ import annotations

import json

from .benchmark import run_benchmark
from .extended import run_extended


def run_experiment() -> dict[str, object]:
    """Wykonuje główny benchmark, a następnie pojedynczy przebieg rozszerzony."""

    return {
        "benchmark": run_benchmark(),
        "extended": run_extended(),
    }


def main() -> None:
    print(json.dumps(run_experiment(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
