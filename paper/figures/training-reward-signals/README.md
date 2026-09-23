# Training Reward Signals

This figure separates aggregate reward evolution from the frequency-magnitude
paths followed by two contrasting objectives during training.

- Panel A shows centered 25-update means of the signed reward from each
  objective and the total reward.
- Panel B shows centered 25-update means of each objective's share of total
  absolute reward mass. The component shares sum to 100% within each update.
- Panel C follows the percentage of positive SPR-reward days and the mean SPR
  reward on those days through training.
- Panel D gives the corresponding path for storage, a frequent reward signal
  that provides a useful contrast with event-driven SPR rewards.

All training panels use the same-configuration rich diagnostic retraining of
the selected configuration and seed. Single-threaded CPU arithmetic caused
that trajectory to diverge numerically from the archived selected-policy run,
so it is used to diagnose learning mechanisms rather than to replace reported
selected-policy performance. Stars in Panels C and D show the archived
selected policy's deterministic 2014--2024 evaluation.

Suggested caption:

> Reward signals during same-configuration diagnostic retraining of the
> selected policy design. (A) Mean signed reward per step for each objective
> and their total; thin lines show individual updates and bold lines show
> centered 25-update means. (B) Centered 25-update mean shares of total
> absolute reward mass. (C-D) Training paths relating the frequency of
> positive rewards to their conditional mean magnitude for spring peak
> release (SPR) and storage, respectively. Color denotes training progress,
> diamonds mark the end of training, and stars show the
> archived selected policy's deterministic 2014--2024 evaluation. The
> diagnostic retraining used the selected configuration and seed but is not a
> bit-for-bit reproduction of the archived selected-policy trajectory.

The raw update metrics and component diagnostics are bundled in `data/` as
`train-update-metrics.csv` and `train-update-component-diagnostics.csv`.
Their original source is
`runs/selected_policy_rich_diagnostics/seed_004_cpu1/`. The evaluation daily
rewards are also bundled, so no priority-hints folder is required for refresh.

Refresh the compact tracked extracts from these local snapshots when needed:

```powershell
python paper\figures\training-reward-signals\build.py --refresh-data
```

Routine rebuild:

```powershell
python paper\figures\training-reward-signals\build.py
```
