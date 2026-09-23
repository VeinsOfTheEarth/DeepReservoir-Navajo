> **Prepublication artifact:** This figure uses the archived pre-correction Phase-95 policy or reward set and is included for provenance. Refresh it after final corrected-policy selection.

# Reward Functions Diagrams

Run `python -B paper/figures/reward-functions-diagrams/build.py` from the
repository root. It writes `reward-functions-diagrams.png` and
`reward-functions-diagrams.pdf`. The six non-SPR reward panels are defined
directly in the builder, with the original palette/axes helpers in
`rewardstyle.py`. No extra experiment data are needed. SPR is documented by
the separate, manually edited `spr-flow` figure.
