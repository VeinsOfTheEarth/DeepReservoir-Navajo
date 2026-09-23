"""Create the bundled static panel font from the repository's Inter font."""

from pathlib import Path

from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


if __name__ == "__main__":
    source = ROOT / "assets/fonts/inter/Inter-VariableFont_opsz,wght.ttf"
    with TTFont(source) as variable:
        bold = instantiateVariableFont(
            variable, {"opsz": 14, "wght": 700}, updateFontNames=True
        )
        bold.save(HERE / "Inter-Bold.ttf")
