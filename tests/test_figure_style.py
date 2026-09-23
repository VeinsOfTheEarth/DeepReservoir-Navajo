from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from fontTools.ttLib import TTFont


SUPPORT = Path(__file__).resolve().parents[1] / "paper" / "figure-support"
sys.path.insert(0, str(SUPPORT))
from figurestyle import OBJECTIVE_COLORS, PANEL_FONT, TEXT, add_panel_label


class TestFigureStyle(unittest.TestCase):
    def test_panel_font_is_static_inter_bold(self):
        with TTFont(PANEL_FONT) as font:
            self.assertNotIn("fvar", font)
            self.assertEqual(font["OS/2"].usWeightClass, 700)
            self.assertEqual(font["name"].getDebugName(6), "Inter-Bold")

    def test_panel_letters_do_not_depend_on_body_font(self):
        with plt.rc_context({"font.family": "DejaVu Sans", "font.style": "italic"}):
            fig, ax = plt.subplots()
            try:
                first = add_panel_label(ax, "A")
                second = add_panel_label(ax, "B", x=0.98, ha="right", fontsize=11)
                for text in (first, second):
                    self.assertEqual(text.get_fontproperties().get_file(), str(PANEL_FONT))
                    self.assertEqual(text.get_fontweight(), "bold")
                    self.assertEqual(text.get_fontstyle(), "normal")
                    self.assertEqual(to_hex(text.get_color()), TEXT.lower())
                    self.assertIsNone(text.get_bbox_patch())
                self.assertEqual(second.get_position(), (0.98, 0.98))
                self.assertEqual(second.get_fontsize(), 11)
            finally:
                plt.close(fig)

    def test_panel_letter_size_scales_with_figure_width(self):
        for width in (7.55, 8.4, 8.6, 10.4, 11.6, 12.9):
            with self.subTest(width=width):
                fig, ax = plt.subplots(figsize=(width, 4))
                try:
                    text = add_panel_label(ax, "A")
                    self.assertAlmostEqual(text.get_fontsize() / width, 12.5 / 8.4)
                    override = add_panel_label(ax, "B", fontsize=11)
                    self.assertEqual(override.get_fontsize(), 11)
                finally:
                    plt.close(fig)

    def test_objective_colors_are_distinct(self):
        self.assertEqual(len(OBJECTIVE_COLORS), 7)
        self.assertEqual(len(set(OBJECTIVE_COLORS.values())), 7)
        self.assertEqual(OBJECTIVE_COLORS["niip"], "#2CA25F")
        self.assertEqual(OBJECTIVE_COLORS["spr"], "#7C3AED")

    def test_approved_schematic_palette(self):
        self.assertEqual(OBJECTIVE_COLORS["storage"], "#1D4ED8")
        self.assertEqual(OBJECTIVE_COLORS["esa"], "#0891B2")
        self.assertEqual(OBJECTIVE_COLORS["hydropower"], "#D97706")

    def test_builders_use_the_shared_objective_palette(self):
        root = SUPPORT.parent / "figures"

        def load(slug):
            name = "style_test_" + slug.replace("-", "_")
            spec = importlib.util.spec_from_file_location(name, root / slug / "build.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module

        training = load("training-reward-signals")
        keys = {
            "dam_safety": "dam_safety", "flood_safety": "flood",
            "niip_delivery": "niip", "hydropower": "hydropower",
            "esa_min_flow": "esa", "storage": "storage",
            "spring_peak_release": "spr",
        }
        for component in training.COMPONENTS:
            self.assertEqual(component["color"], OBJECTIVE_COLORS[keys[component["key"]]])

        stress = load("stress-testing")
        for (label, _, color), key in zip(
            stress.METRIC_SPECS, ("dam_safety", "storage", "esa", "flood", "spr", "hydropower", "niip")
        ):
            self.assertEqual(color, OBJECTIVE_COLORS[key], label)

        response = load("policy-response")
        for (label, _, color), key in zip(
            response.OBJECTIVES, ("dam_safety", "storage", "esa", "flood", "hydropower", "niip")
        ):
            self.assertEqual(color, OBJECTIVE_COLORS[key], label)

        architecture = load("architecture-network-nodes")
        for key in ("storage", "esa", "niip", "spr"):
            self.assertEqual(architecture.COLORS[key], OBJECTIVE_COLORS[key])

        release = load("release-attribution")
        for component in release.ALL_RELEASE_COMPONENTS:
            if component.key == "unassigned":
                continue
            key = "dam_safety" if component.key == "spill" else component.key
            self.assertEqual(component.color, OBJECTIVE_COLORS[key])


if __name__ == "__main__":
    unittest.main()
