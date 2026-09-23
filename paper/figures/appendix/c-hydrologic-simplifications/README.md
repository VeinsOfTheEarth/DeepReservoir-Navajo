# Appendix C figures and evidence

`build.py` generates the three PDF figures used by Appendix C, PNG previews,
and `summary.json`. It reads the archived daily gage records, original
Farmington patch inputs, and shared figure style bundled in this repository.
No external scientific-code checkout or slide deck is needed.

From the repository root, rebuild with:

```powershell
python paper/figures/appendix/c-hydrologic-simplifications/build.py
```

| Figure | PDF | Evidence and limits |
| --- | --- | --- |
| C1 | `farmington-infill.pdf` | Reproduces the archived patch of the **San Juan at Farmington** record from original observations, Archuleta and Animas daily records, and `patched_rows.csv`. The 46.68-cfs bias and fit statistics use the full observed overlap, including evaluation years, and are descriptive rather than held-out validation. |
| C2 | `tributary-context.pdf` | Recomputes annual mean and annual daily peak discharge for selected gages on years with at least 330 valid daily values. It updates the comparison shown on slide 7; gage coverage differs, and episodic peaks remain important. |
| C3 | `travel-time-evidence.pdf` | Shows the approximate reach estimates from slides 8–10 and their approximately 48-hour Navajo Dam-to-Bluff total, used to explain the retained two-day flood-proxy delay. The original subdaily event-selection code was not recovered. |

`summary.json` records the numerical results and SHA-256 checksums for every
bundled input. A local copy of the historical slide deck can be supplied with
`--slide-deck`; it is then hashed as optional provenance and is not read by the
calculation. The reported 10-hour dam-to-Farmington estimate was extrapolated
from the 8.6-hour Archuleta-to-Farmington estimate.
Slide 11 also presents a separate observed-overlap regression,
`Farmington = 0.971 × (Archuleta + Animas) + 5.4 cfs`; the archived gap patch
used the additive-bias formula instead.
The author retained the current two-day flood-proxy delay because the Bluff
criterion concerns high-flow peak arrival and slide 10 estimates approximately
48 hours from Navajo Dam to Bluff. The prior broad-record daily correlation
check was removed from this figure and is not used to choose the flood lag.
The code delays the combined mainstem-outlet-plus-Animas quantity by two days;
this also shifts the Animas contribution by 48 hours, while the slide-derived
Farmington-to-Bluff estimate is approximately 38 hours. The figure illustrates
a daily approximation, not event-specific routing or validation.
The PDFs are regenerated with fixed PDF metadata so their checksums are stable
across otherwise identical builds.
