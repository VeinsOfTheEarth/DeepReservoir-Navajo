# Release Attribution

`release-attribution.pdf` and `release-attribution.png` are the final paper
exports. Rebuild from the repository root with the project's Python environment:

```powershell
python -B paper/figures/release-attribution/build.py
```

The builder uses the shared selected-policy rollout and physical-environment
configuration. No additional experiment data or archived figure is required.
Treat the release components as a diagnostic attribution, not causal intent.
