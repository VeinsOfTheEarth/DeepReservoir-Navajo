# Policy Response to Changed Inflow

This figure shows what changes when the selected policy can update its actions
in response to altered inflow rather than following a previously saved action
schedule.

For each reservoir-inflow perturbation from -50% to +50%, the selected policy is
evaluated twice:

1. **Updated actions:** the policy observes the evolving reservoir state and
   chooses new actions each day.
2. **Saved action schedule:** the normalized action vector from the corresponding
   day of the unperturbed deterministic evaluation is used instead. Environment
   controllers, release limits, and physical feasibility checks remain active.

The paired difference therefore estimates the effect of updating neural-policy
actions as conditions change, conditional on the model's existing controller
structure. It does **not** remove structural protections such as the SPR target
bridge or the state-dependent mapping from the ESA action to a required
baseflow release.

The final `policy-response.png` and `policy-response.pdf` contain:

- Panel A: mean change in each normalized action head from
  the saved schedule. Positive values mean the policy requested more on
  average when allowed to update its actions.
- Panel B: cumulative release and spill changes over
  2014--2024, plus the change in ending storage, caused by updating actions.
- Panel C: objective-performance change from updating
  actions, expressed as a fraction of the corresponding historic benchmark.
  SPR is omitted because this diagnostic deliberately sets aside its known
  structural priority.

The default builder no longer emits standalone copies of the three panels.
Their earlier exports are preserved only in the ignored local
`.figure-archive/priority-explorations/` directory, outside the paper figures.

The paired cumulative water accounting closes to within 1.63 kAF in every run,
less than 0.07% of the largest plotted reallocation. The small residual occurs
only in severe dry cases, where the environment's zero-storage floor can make
the imposed evaporation loss supply-limited.

Routine builds use the paired summaries and metadata already in `data/`:

```powershell
python -B paper/figures/policy-response/build.py
```

Only rerun the diagnostics when intentionally replacing those source data:

```powershell
python paper\figures\policy-response\run_diagnostics.py
python paper\figures\policy-response\build.py
```
