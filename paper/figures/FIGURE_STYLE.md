# Paper Figure Style Guide

This file is the visual and terminology contract for paper-facing figures in
this repository. It records the conventions used by the figures in the current
WRR draft, the September 2026 implementation audit, and deliberate exceptions.

The executable palette and panel-letter style are defined in
`paper/figure-support/figurestyle.py`. Do not duplicate objective hex codes in
individual builders; import the shared mapping instead.

## Current Paper Figure Set

The agreed final paper set includes the following technical figures:

- `reward-functions-diagrams`
- `spr-flow`
- `architecture-network-nodes`
- `historic-timeseries`
- `selected-policy-comparison-with-historic`
- `stress-testing`
- `release-attribution`
- `seed-variability`
- `experiment-search`
- `niip-reflex`
- `training-reward-signals`
- `policy-response`

The paper also includes `study-system-map` and the manually refined
`objective-locations-schematic`. Their editable sources must be preserved.
The environment diagram is excluded from the agreed set. It remains in the
September 8 Overleaf ZIP as Figure 3 and should be removed there on refresh.
The reward/priority investigation contributes only `training-reward-signals`
and `policy-response`. Concentration-index plots and earlier priority-audit
alternatives are excluded and archived locally outside the paper figures.
See [README.md](README.md) for the authoritative folder index. The confirmed
manual sources are `study-system-map/study-system-map.pptx` and
`spr-flow/spr-flow.pptx`.

Component reward plots, alternate architecture mockups, seed-group summaries,
and intermediate design-lesson plots are not part of the current paper set.
They are archived locally under the ignored `.figure-archive/` directory and
do not need visual alignment unless promoted into the manuscript.

## Encoding Precedence

Color can encode either a management strategy, an objective family, or a
continuous variable. Use the following order to avoid assigning two meanings
to one mark:

1. In direct strategy comparisons, strategy identity takes precedence:
   historic management is grey and the selected policy is blue.
2. In plots comparing several objectives within one strategy or stress run,
   objective-family colors take precedence.
3. In a single-objective plot, use that objective's family color even when the
   data come from the selected policy.
4. Continuous scenario variables may use an appropriate sequential or
   diverging colormap with a labeled colorbar.
5. Maps and logic diagrams may use their own cartographic or logical palette,
   provided the legend makes that separate meaning explicit.

Never rely on color alone when two strategies or important states are being
compared. Pair color with line style, marker shape, fill, or annotation.

## Canonical Palette

| Concept | Primary | Light fill | Notes |
| --- | --- | --- | --- |
| Selected policy | `#2B6CB0` | `#BBD7F0` | Strategy encoding; solid line |
| Storage | `#1D4ED8` | pale blue | Royal blue from the approved objective schematic |
| Historic management | `#8B95A5` | `#E5E7EB` | Short-dotted line where possible |
| NIIP | `#2CA25F` | `#BBEBCF` | Demand remains grey; delivery is green |
| SPR | `#7C3AED` | `#DDD6FE` | Physical SPR thresholds may be grey |
| Hydropower | `#D97706` | `#FED7AA` | Amber from the approved schematic; use `Hydropower`, not `Hydro` |
| ESA minimum flow | `#0891B2` | pale teal | Teal from the approved schematic; distinct from storage blue |
| Flood safety | `#B91C1C` | pale red | Do not substitute hydropower orange |
| Dam safety / spill / spill-free days | `#111827` | grey | Use consistently for dam-safety reward components and outcome metrics |
| Spill-pressure input feature | `#C44536` | pale red | Architecture input group, not the dam-safety objective color |
| Discretionary release action | `#D97706` | pale orange | Same action color in architecture and policy response; not an independent objective |
| Text | `#111827` | - | |
| Muted text | `#6B7280` | - | |
| Grid | `#E2E8F0` | - | |
| Axes spines | `#94A3B8` | - | |
| Plot background | `#F4F7FA` | - | |

Accepted exceptions:

- Selected-policy blue is a strategy color, distinct from royal-blue storage.
- Training-path gradients encode training progress; retain the existing blue
  and purple sequential scales rather than replacing them with flat colors.
- The flood-safety reward panel uses dark red for Farmington and light red for
  Bluff, with different line styles and matching threshold colors.
- The historic SPR threshold bars use four purple shades to distinguish
  threshold levels within the same objective family.
- The historic-timeseries figure uses blue (`#2B6CB0`) for the training period
  and orange (`#D97706`) for the evaluation period. In that figure only, these
  hues encode time periods rather than management strategy or objective
  family. Direct labels and vertically aligned shading make that contextual
  meaning explicit.
- The selected-policy SPR time series uses separate Animas and Navajo fills.
- The SPR flowchart uses teal for decisions/credits and orange for penalties.
  These are logic meanings, not objective-family colors.
- The map uses gold headwater boundaries, a cyan reservoir, and coral USGS
  gage symbols and labels with dark outlines.
- Seed identity may use a qualitative palette because the colors identify
  seeds rather than objectives.
- The seed-variability selected-policy line is black for contrast against the
  other seed colors. The experiment-search selected policy is a gold star,
  following the author's choice; neither is an objective-color encoding.

## Fonts And Final-Size Scale

- The primary paper font is `Inter`.
- Register the bundled files in `assets/fonts/inter` with
  `matplotlib.font_manager.addfont` before setting `font.sans-serif`.
- Use `Source Sans 3`, `IBM Plex Sans`, and `DejaVu Sans` only as fallbacks.
- Panel letters use `paper/figure-support/fonts/Inter-Bold.ttf` explicitly.
  This is a static weight-700 instance of the bundled variable font, because
  Matplotlib otherwise resolves requested bold weights to Inter Regular.
- Use `figurestyle.add_panel_label` for all programmatic panel letters. Keep
  the existing figure-specific positions, but not local font rules. By default,
  the helper scales panel letters with canvas width: 12.5 pt at 8.4 inches.
  This keeps their apparent size approximately constant when different-width
  figures are inserted at the same manuscript width. An explicit `fontsize`
  overrides this only when a different publication width requires it.
- PowerPoint-authored figures should also use Inter before export.
- Export PDFs with editable TrueType text using `pdf.fonttype = 42` and
  `ps.fonttype = 42`.

Target sizes after the figure has been reduced to its final manuscript width:

| Element | Final target |
| --- | --- |
| Axis label | 8.5-9.5 pt |
| Tick label | 7.5-8.5 pt |
| Legend / annotation | 7.5-8.5 pt |
| Compact reward-panel title | 9-10 pt |
| Panel letter | 10.5-12 pt, bold |

Source font size alone is not a valid consistency check. If a 12.9-inch-wide
Matplotlib canvas is inserted at the same `\textwidth` as an 8.4-inch-wide
canvas, its text will be about 35% smaller. Prefer authoring full-width figures
at a common width of roughly 7.5-8.5 inches. When a wider canvas is needed,
scale all source font sizes by the same ratio and inspect the compiled PDF at
100% zoom. No final figure text should fall below about 7 pt.

The current selected-policy comparison, stress-testing, NIIP-reflex, and
architecture figures need this final-size check most urgently because their
source canvases are 10.4-12.9 inches wide.

## Panel Letters

- Use uppercase `A`, `B`, `C`, and so on.
- Use the same static Inter Bold (weight 700), no surrounding white box, at axes coordinates approximately
  `(0.02, 0.98)`.
- Place letters in the upper-left by default.
- The reward figure's `C` and `H`, experiment-search panel `A`, selected-policy
  comparison panel `E`, and policy-response panel `A` use the upper-right to
  avoid upper-left legends or data.
- Use the same final-size letter weight and size in every figure.
- Captions must refer to panels as `(A)`, `(B)`, and so on, matching the image.
  The current reward-function caption uses lowercase panel references and
  should be corrected.

The September 11 PDF audit confirmed Inter Bold, upright uppercase letters,
and `#111827` in all nine lettered figures. The inconsistency was scale, not
font family: fixed point sizes looked larger in narrower exports such as
training-reward-signals than in the wider stress and policy-response figures.
The shared width-based sizing now corrects this; local fixed-size overrides
were removed from NIIP-reflex and release-attribution. The three manual figures,
architecture-network-nodes, and seed-variability have no panel letters.

## Background, Grid, And Axes

- Use a white figure canvas.
- Use `#F4F7FA` for quantitative plot axes by default.
- White axes are appropriate for architecture diagrams, flowcharts, and maps.
- Dense scatterplots and parallel-coordinate plots also use the pale-grey
  background. Retain white only if a documented visibility problem remains
  after adjusting mark opacity and grid weight.
- Use grid color `#E2E8F0`, width about `0.6`, alpha about `0.7`, and draw it
  behind the data.
- Use a solid light grid by default. A dotted grid is acceptable for dense
  seed or experiment plots if it reduces interference with the marks.
- Hide top and right spines unless the plot is a heatmap or framed diagram.
- Use `#94A3B8`, about `0.9` pt, for visible left and bottom spines.
- Zero-reward lines should be darker than the grid and visually consistent
  across all reward panels.

All programmatic quantitative figures now use the canonical pale-grey axes.
The architecture's white canvas and pale alternating row bands are deliberate.

## Lines, Markers, And References

- Selected policy: solid line.
- Historic management: grey short-dotted line.
- Benchmark at 1.0: dark-grey dashed line.
- Physical limits such as dead pool, spill level, and SPR thresholds:
  grey dashed lines.
- Put grids behind bars, fills, and trajectories.
- Use markers as well as colors for multi-objective response curves when
  curves overlap or the figure must remain interpretable in greyscale.
- Use black crosses for annual SPR target attainment and define them once in
  the legend.
- Use uncertainty bands only when their statistic is named explicitly, such as
  `IQR`.

## Terminology

Use these forms in axes, legends, annotations, and captions:

| Use | Avoid or restrict |
| --- | --- |
| `historic management` | `historical management` |
| `selected policy` | `best policy`, `final policy` |
| `selected-family mean` in prose | `average policy` |
| `Family mean (n=16)` in compact tables | longer forms when space is limited |
| `training period`, `evaluation period` | `train`, `test`, `eval` in paper text |
| `San Juan` in prose | `SJ` in prose before definition |
| `SJ` in compact axes and legends | spelling out `San Juan` when it causes wrapping |
| `San Juan at Farmington` | `San Juan @ Farmington` |
| `SPR target` | `threshold` when referring to the agent's choice |
| `flow threshold` | `target` when referring only to a physical discharge line |
| `ESA minimum flow` | `ESA min` except in very compact labels |
| `Hydropower` | `Hydro` |
| `NIIP demand proxy` on first use | implying the series is an independent demand survey |
| `NIIP demand` in compact labels after definition | switching among demand, target, and historic NIIP |
| `Navajo Reservoir release` | bare `Navajo` where it could mean the basin |

Preferred compact objective labels:

- `Storage`
- `Spill-free days`
- `Flood safety`
- `ESA min flow`
- `SPR 10k/5d`, `SPR 8k/10d`, `SPR 5k/21d`, `SPR 2.5k/10d`
- `Hydropower`
- `NIIP volume`
- `NIIP daily delivery`

Use `historic` in on-figure labels throughout. Captions should also prefer
`historic record`, `historic management`, and `historic benchmark` to match the
author's terminology preference.

## Units And Number Formatting

- Discharge: `cfs`.
- Reservoir storage: `MAF`.
- Volume: `kAF`, not `thousand acre-ft`.
- Hydropower generation: `MWh/day`.
- Percentages: `%`.
- Days: `d` only in compact SPR labels; spell out `days` in prose.
- Use commas for full discharge values: `10,000 cfs`.
- Compact threshold labels may use `10k`, `8k`, `5k`, and `2.5k` after the
  units and convention have been defined.
- Ratios should name their denominator: `Selected policy / historic`,
  `Stress run / historic`, or `Ratio to benchmark or target`.
- Paper-facing objective comparisons use the corresponding historic-management
  metric as the denominator. A value of one matches historic management, and
  values above one meet or exceed it. Do not mix these ratios with stress-run /
  nominal-policy "retention" ratios in the same paper.
- `Spill-free days` is the explicit target-normalized exception: plot
  `1 - fraction of simulated spill days`, so one means no spill. A reliable
  historic spill denominator is unavailable.

## Titles, Legends, And Annotations

- Do not use panel titles in result and context figures.
- Reward-function panels may retain compact titles because the title identifies
  the function being shown.
- Keep legends inside unused axes space when they do not cover data.
- Use white legend backgrounds with `#D6DEE6` borders for quantitative plots.
- Frameless legends are acceptable in dense design/seed figures.
- Use `Historic management` and `Selected policy` consistently in legends.
- When plotting climatological summaries, use explicit labels such as
  `Selected policy median`, `Selected policy IQR`, and `Historic median`.
- Use sentence case for callouts and annotations.

## Export, Naming, And Overleaf

- Keep one final figure directory and one authoritative build script per paper
  figure. Helper modules are acceptable, but the directory's `build.py` must
  reproduce the included output.
- Match each directory name to its final output stem, using dashes instead of
  spaces or underscores. Keep figure-specific helpers beside the builder and
  extra experiment inputs in that figure's `data/` folder. Shared selected-policy
  artifacts and physical-environment data need not be duplicated.
- Common plotting/benchmark code is in `paper/figure-support/paperstyle.py`;
  objective colors and panel typography are in `figurestyle.py` beside it.
- For manually edited figures, preserve the editable source and identify its
  exported final explicitly. Do not claim that the earlier Python mockup
  reproduces subsequent PowerPoint edits.
- Track both PDF and PNG outputs.
- Use kebab-case for final filenames.
- Prefer PDF in Overleaf for plots, diagrams, and flowcharts so text and lines
  remain sharp. Use PNG when the figure is fundamentally raster, such as an
  imagery basemap.
- If PNG is required, export at no less than 300 dpi at final physical size.
- Preserve the repository filename when copying an artifact to Overleaf.
  The draft currently references `reward_functions_diagrams.png`, while the
  tracked output is `reward-functions-diagrams.png`.
- Avoid absolute machine paths in figure scripts. External source locations may
  be documented as provenance, but routine builds should use tracked inputs.

`spr-flow/spr-flow.pptx` is the authoritative editable source, with
`spr-flow/spr-flow.pdf` retained for the paper. The stale PNG and older variants
are archived locally; re-export from PowerPoint when a raster version is
needed. For subsequent exports:

- use Inter throughout;
- retain teal solid paths for credits/valid decisions and orange dashed paths
  for penalties/rejected decisions;
- keep credit and penalty equations explicit in the terminal boxes;
- preserve consistent diamond and terminal-box sizes;
- avoid adding a panel title inside the artwork; and
- prefer PDF, or use a PNG exported at 300 dpi or greater at final page width.

## Included-Figure Audit

The September implementation pass checked all fourteen final exports and their
builders or PowerPoint sources. Corrected programmatic inconsistencies:

- Reward-function font registration used a nonexistent `paper/assets/fonts`
  path and silently produced DejaVu Sans. It now uses the bundled Inter fonts.
- All panel letters use a single static Inter Bold font, including the
  externally positioned policy-response C. Existing placements are retained.
- Objective colors now come from one shared mapping. Architecture input groups
  and release-attribution components use the same family colors as the
  comparison, stress-testing, training-reward, and policy-response plots.
- Dam-safety reward components are charcoal; flood-safety curves are red;
  hydropower's reward surface uses a sequential ramp to its objective color.
- Historic SPR threshold bars use purple shades instead of unrelated hues.
- `ESA min flow` is used in the comparison summary; seed SPR labels include
  their durations; selected-policy median/IQR labels are explicit; the NIIP
  reflex legend now says `Median NIIP release`.

Verification: all eleven programmatic PNG/PDF pairs rebuilt successfully;
`pdffonts` confirms Inter Regular in every programmatic PDF and Inter Bold in
all nine multi-panel PDFs. Before/after contact sheets of all fourteen figures
were inspected for label fit and visual consistency. Nine figure-build/style
tests pass, including shared-palette checks across the main builders. Only
rendering and labels changed; metric definitions and input data did not.

### Manual Export Follow-up

The three manually refined figures were inspected but not overwritten by the
programmatic rebuild:

| Figure | Finding | Follow-up |
| --- | --- | --- |
| `study-system-map` | PowerPoint specifies Inter, but the retained PDF embeds Calibri variants | Re-export with Inter available and verify the resulting PDF fonts; preserve the current label positions |
| `objective-locations-schematic` | PowerPoint specifies Inter; all seven objective colors now match the shared palette | Author approved the schematic palette after reviewing the training-reward-signals preview; no PowerPoint color edits needed |
| `spr-flow` | PowerPoint uses Arial; its teal/orange colors represent logic, not objectives | Switch to Inter and re-export from PowerPoint after checking text wrapping; preserve the credit/penalty color meanings |

The author approved the schematic palette in September 2026 after reviewing a
separate training-reward-signals preview. Storage, ESA minimum flow, and
hydropower now use that palette throughout the programmatic objective plots.
No manual text, connector, or label placement was changed.

| Figure | Already consistent | Follow-up for the implementation pass |
| --- | --- | --- |
| `study-system-map` | Programmatic provenance; paper font registration; unobscured imagery; gold headwater outlines; uniform coral USGS gages and labels; paired area/discharge inset | Check final label size in the compiled paper |
| `reward-functions-diagrams` | Verified embedded Inter Regular/Bold; canonical objective palette; pale plot background; consistent zero lines | Correct lowercase caption panel references; verify the very tall figure fits with its caption |
| `spr-flow` | Clear credit/penalty/decision palette; no unnecessary panel title; confirmed source and PDF share one directory | Preserve Inter when editing; export a PNG only if needed; finish caption |
| `architecture-network-nodes` | Inter body text; strong feature-group colors; clear actor/critic separation; final `build.py` reproduces the output | Check effective final text size; mathematical glyphs still use Matplotlib's DejaVu math font |
| `historic-timeseries` | Canonical background; shared Inter Bold panel letters; purple SPR thresholds; no panel titles; shared time axis | Retain blue/orange as the documented local time-period encoding |
| `selected-policy-comparison-with-historic` | Shared objective palette and Inter Bold panels; explicit selected-policy median/IQR; consistent ESA name | Check final-size type in the compiled paper; replace `San Juan @ Farmington` and `eval` in caption |
| `stress-testing` | Objective colors match comparison panel G; Inter panel letters; shared ratio framing; canonical objective names | Check final-size type in the compiled paper; add marker-shape redundancy if needed |
| `release-attribution` | Inter; canonical background; canonical NIIP/SPR/hydropower/ESA colors; `kAF` volume units | Standardize compact San Juan/SJ labels if the current legend is too wide at final size |
| `seed-variability` | Inter; canonical background; SPR labels include durations; selected policy emphasized without competing with seed colors | Seed colors intentionally identify seeds, not objectives; retain the established axis order |
| `experiment-search` | Inter; canonical background; neutral pass-count scale; selected policy and historic markers are distinct; phase medians are labeled | Define composite scores in the caption or methods; compact `Historic` remains acceptable in this crowded legend |
| `niip-reflex` | Shared Inter Bold panels; canonical background; median NIIP release explicitly labeled; NIIP green and SPR-window purple; grey base reward | Verify final-size type in the compiled paper |
| `training-reward-signals` | Inter; canonical objective colors and background; signed credits and penalties remain distinct | Define the rolling normalization in the caption and do not interpret reward share as an operator-priority weight |
| `policy-response` | Canonical objective colors; compact action/allocation panels and performance-change heatmap | Define updated actions versus the saved action schedule and distinguish action changes from outcome changes |

## Resolved Decisions

The July 2026 consistency review established the following:

1. Use `Flood safety` everywhere in paper-facing text, including the reward
   function name. Internal data keys may retain older names.
2. Use `Spill-free days` for the no-spill objective.
3. Retain distinct blue and orange training/evaluation encodings in the
   historic-timeseries figure as a local time-period encoding.
4. Use pale-grey axes for all quantitative plots, including dense diagnostic
   figures.
5. Use a neutral slate sequential scale for experiment-search objective pass
   counts so green remains reserved for NIIP.
6. In NIIP reflex, use grey for the base reward and green for the banded NIIP
   reward.

The September 2026 palette review additionally established:

7. Match the objective schematic: storage `#1D4ED8`, ESA minimum flow
   `#0891B2`, and hydropower `#D97706`. Retain the other four objective colors.
   Strategy, seed, training-progress, and scenario encodings remain separate.
