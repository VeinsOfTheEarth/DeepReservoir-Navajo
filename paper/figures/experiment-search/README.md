# Experiment Search

Run `python -B paper/figures/experiment-search/build.py` from the repository
root to generate `experiment-search.png` and `experiment-search.pdf`.
`search.py` contains the original composite-score and plotting calculations.

## Bundled Inputs

- `data/archived-search-metrics.csv`: metrics for the earlier experiments,
  consolidated from 445 source files. The table contains 3,320 raw archive
  records, of which 2,740 meet the figure's comparison rules below.
- `data/phase95-seed-metrics-with-pass-flags.csv`: phase-95 seed metrics and
  pass flags, originally from `jon/phase95/phase95_seed_metrics_with_pass_flags.csv`.
- `data/phase95-historic-benchmark.csv`: corresponding historic benchmarks,
  originally from `jon/phase95/phase95_historic_benchmark.csv`.
- `data/archive-provenance.json`: original source paths, file hashes, and counts.

Routine builds do not read `jon/` or `runs/report_html/`. The consolidated
table preserves the original per-source spill-pass rules, including failure
when neither spill metric was available. For sources lacking total spill, the
synthetic `total_spill_af` field encodes the original zero-spill test as zero
(pass) or infinity (fail); it must not be interpreted as a measured volume.
See the provenance note.

The figure includes archive rows only when they represent a policy or checkpoint
evaluation over the retrospective test interval, contain all nine nonspill
metrics, and retain a no-spill screen result. Phase-78 training-window
evaluations are excluded, as are repeated
Phase-79 historical rows, rows missing any of the nine nonspill metrics, and
rows without a recorded spill-screen result. A synthetic infinity is retained as
a recorded no-spill failure. Phase medians and distributions give each retained
record equal weight. These rules retain 2,740 earlier records; with the 128
Phase-95 seeds, the plotted population is 2,868 records. Three meet
all ten screens: one earlier record and two Phase-95 seeds.

The composite-score normalization, weighting, phase order, and pass thresholds
remain unchanged. Because phases contain unequal mixtures of designs,
checkpoints, and seeds, the plotted frequencies describe this development
archive and are not estimates of a general PPO success probability.
