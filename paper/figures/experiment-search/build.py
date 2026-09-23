"""Build experiment-search design-lesson figure."""

from __future__ import annotations

import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
from search import (  # noqa: E402
    DEFAULT_PHASE95_BENCHMARK,
    DEFAULT_PHASE95_SEED_METRICS,
    build_experiment_search,
    _set_theme,
)


def main() -> None:
    _set_theme()
    for path in build_experiment_search(
        seed_metrics_path=DEFAULT_PHASE95_SEED_METRICS,
        benchmark_path=DEFAULT_PHASE95_BENCHMARK,
        output_dir=HERE,
        dpi=300,
        only_stems={"experiment-search"},
    ):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
