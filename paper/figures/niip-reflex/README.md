# NIIP Reflex

Run `python -B paper/figures/niip-reflex/build.py` from the repository root.
It writes `niip-reflex.png` and `niip-reflex.pdf`. `reflex.py` contains the
figure-specific climatology and paired-comparison calculations.

Both extra inputs are bundled in `data/`:

- `phase52-highreq10-niipband-tight-003-seed-002-eval-rollout.csv`, originally
  `runs/report_inputs/phase52/reward_jon_p52_highreq10_niipband_tight_003/seed_002/eval__holdout_2014_2024_08_17/eval_rollout.csv`.
- `phase52-split-paired-reflex-reduction.csv`, reconstructed from all 36
  paired-treatment rollouts used by
  `jon/phase52/phase52_split_paired_reflex_reduction.csv`.

The paired reflex table matches policies by SPR design, numerical seed, and NIIP
reward. Within each pair only the shared versus split actor-head architecture
changes. Reflex is the absolute difference in mean NIIP release bias between
dates with and without a requested controlled San Juan release above 1 cfs.
Both samples require NIIP demand of at least 100 cfs and day of year 50--300.
There are 18 nominal pairs; one has no qualifying no-request dates and therefore
has an undefined shared-head reflex, leaving 17 pairs in the plotted panel.

No archive run folders or old design-lesson directory are needed to rebuild.
