# Panel Font

`Inter-Bold.ttf` is a static instance of the repository's bundled Inter variable
font, with weight 700 and optical size 14. It ensures that Matplotlib embeds
the same genuinely bold typeface for every panel letter rather than silently
using the regular variable-font instance.

The original font is in `assets/fonts/inter/`; its copyright and license
metadata are preserved in the static instance. To regenerate
the static font with fontTools, run `python -B paper/figure-support/fonts/build.py`.
Routine figure builds use the retained TTF and do not regenerate it.
