# Training Reward Signals

This figure uses the saved training trace for the corrected selected policy
(`reward_jon_p95_oishift875to90_heff`, seed 13, recovery task 105).

- Panel A shows centered 25-update means of the signed reward from each objective and the total reward.
- Panel B shows the spring-peak objective's share of total absolute reward mass, nonzero-reward days, and positive-reward days.
- Panel C relates the frequency and mean magnitude of positive spring-peak rewards through training. The star is the deterministic 2014--2024 selected-policy evaluation.

Rebuild with:

```powershell
python paper/figures/training-reward-signals/build.py
```
