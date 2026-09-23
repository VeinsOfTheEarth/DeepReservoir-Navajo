# Shared Paper Figure Helpers

`figurestyle.py` is the lightweight source of objective colors and panel-letter
formatting. Every programmatic multi-panel figure uses its `add_panel_label`;
all objective-comparison plots use `OBJECTIVE_COLORS`. Panel letters use the
bundled static Inter Bold font in `fonts/`, so body-font fallback cannot change
their appearance. Intentional seed, scenario, and map encodings are separate.

`paperstyle.py` re-exports the shared visual conventions and supplies Matplotlib
styling, selected rollout/benchmark accessors, and reusable comparison panels.
It was moved out
of the old `selected_policy_comparison` grouping so final figures need not
depend on an obsolete figure directory. Styling edits do not change the metric
calculations or their denominators.

Figure-specific generators and additional experiment data live in their own
`paper/figures/<figure-name>/` directories. The retained reward/priority
figures use this module and their own bundled data; they do not depend on
the locally archived development figures.
