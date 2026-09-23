from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "paper" / "build-figures.py"
SPEC = importlib.util.spec_from_file_location("build_paper_figures", SCRIPT)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def figure(slug: str, source: str = "repo_script", **updates) -> dict:
    return {
        "slug": slug,
        "status": "paper",
        "source": source,
        "script": f"paper/figures/{slug}/build.py",
        "outputs": [],
        **updates,
    }


class TestPaperFigureBuild(unittest.TestCase):
    def run_build(self, figures: list[dict], returncodes: list[int]):
        with (
            patch.object(builder, "CATALOG") as catalog,
            patch.object(builder.subprocess, "run") as run,
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            catalog.read_text.return_value = json.dumps({"figures": figures})
            run.side_effect = [subprocess.CompletedProcess([], code) for code in returncodes]
            return builder.main(), run.call_args_list

    def test_skips_manual_and_nonpaper_figures(self):
        status, calls = self.run_build(
            [
                figure("plot"),
                figure("edited-map", "repo_script_and_manual_assets"),
                figure("flowchart", "manual_assets", script=None),
                figure("experiment", status="development"),
            ],
            [0],
        )
        self.assertEqual(status, 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].args[0][-1], str(builder.ROOT / "paper/figures/plot/build.py"))
        self.assertEqual(calls[0].kwargs["env"]["MPLBACKEND"], "Agg")

    def test_reports_failure_and_continues_other_builds(self):
        status, calls = self.run_build([figure("first"), figure("second")], [1, 0])
        self.assertEqual(status, 1)
        self.assertEqual(len(calls), 2)

    def test_missing_output_is_a_failure(self):
        with patch.object(Path, "is_file", return_value=False):
            status, _ = self.run_build([figure("plot", outputs=["plot.pdf"])], [0])
        self.assertEqual(status, 1)

    def test_empty_output_is_a_failure(self):
        with patch.object(Path, "is_file", return_value=True), patch.object(Path, "stat") as stat:
            stat.return_value.st_size = 0
            status, _ = self.run_build([figure("plot", outputs=["plot.pdf"])], [0])
        self.assertEqual(status, 1)


if __name__ == "__main__":
    unittest.main()
