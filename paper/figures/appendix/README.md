# Appendix figures

These directories contain the six programmatically generated figures used by
Appendices A-C of the Navajo Reservoir manuscript. Each builder resolves the
repository root from its own location, uses only bundled runtime data, and can
be run from any working directory. PDF files are the manuscript assets; PNG
files are convenient previews. Each build also refreshes a `summary.json` with
numerical checks and SHA-256 input hashes.

From the repository root, rebuild all six figures with:

```powershell
python paper/figures/appendix/a-data-environment/build.py
python paper/figures/appendix/b-hydropower/build.py
python paper/figures/appendix/c-hydrologic-simplifications/build.py
```

| Directory | Figures |
| --- | --- |
| `a-data-environment/` | Elevation-capacity audit |
| `b-hydropower/` | Model response; monthly calibration fit |
| `c-hydrologic-simplifications/` | Farmington infill; tributary context; travel-time evidence |

The historical source PDF and analysis slide decks/scripts are provenance
artifacts rather than runtime dependencies. The section READMEs document the
optional arguments for hashing local copies of those files.
