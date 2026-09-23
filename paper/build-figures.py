"""Rebuild the final programmatic paper figures, preserving manual exports."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "paper" / "figure_catalog.json"


def main() -> int:
    figures = json.loads(CATALOG.read_text(encoding="utf-8"))["figures"]
    automated = [
        figure for figure in figures
        if figure.get("status") == "paper"
        and figure["source"] == "repo_script"
    ]
    manual = [figure["slug"] for figure in figures if figure not in automated]
    print("Leaving manual figures untouched: " + ", ".join(manual), flush=True)
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    failed = []
    for index, figure in enumerate(automated, start=1):
        print(f"\n[{index}/{len(automated)}] {figure['slug']}", flush=True)
        result = subprocess.run(
            [sys.executable, "-B", str(ROOT / figure["script"])],
            cwd=ROOT,
            env=environment,
            check=False,
        )
        missing = [
            output for output in figure["outputs"]
            if not (ROOT / output).is_file() or (ROOT / output).stat().st_size == 0
        ]
        if result.returncode or missing:
            failed.append(figure["slug"])
            if missing:
                print("Missing or empty output: " + ", ".join(missing), flush=True)
    if failed:
        print("\nFailed: " + ", ".join(failed), file=sys.stderr)
        return 1
    print(f"\nRebuilt all {len(automated)} programmatic figures.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
