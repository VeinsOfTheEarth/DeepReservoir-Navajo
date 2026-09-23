"""Build combined NIIP reflex design-lesson figure."""

from __future__ import annotations

import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
from reflex import (  # noqa: E402
    DEFAULT_NIIP_REFLEX_ROLLOUT,
    DEFAULT_PAIRED_REFLEX,
    build_niip_reflex_design_lesson,
    _set_theme,
)


def main() -> None:
    _set_theme()
    for path in build_niip_reflex_design_lesson(
        rollout_path=DEFAULT_NIIP_REFLEX_ROLLOUT,
        paired_reflex_path=DEFAULT_PAIRED_REFLEX,
        output_dir=HERE,
        dpi=300,
    ):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
