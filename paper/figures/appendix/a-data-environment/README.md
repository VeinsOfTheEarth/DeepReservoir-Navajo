# Appendix A elevation--capacity figure

`build.py` verifies the active elevation-capacity interpolation against the
processed 0.01-ft Reclamation table, then builds the two-panel audit figure used
in Appendix A. The processed table, interpolation pickle, historic reservoir
export, and shared figure style are all bundled in this repository.

From the repository root, rebuild with:

```powershell
python paper/figures/appendix/a-data-environment/build.py
```

Panel A shows the processed relation derived from the [Navajo Reservoir 2019
Area and Capacity Tables](https://www.usbr.gov/tsc/techreferences/reservoir/NavajoReservoir2019AreaCapacityTables_final508VI.pdf)
and the elevation--storage pairs published in the historic reservoir export.
Panel B plots reported storage minus capacity computed from reported elevation
with the 2019 table. The abrupt change on October 1, 2021, supports the inference
that the historic export began using the 2019 relation on that date: elevation fell only
0.05 ft from the prior day, while reported storage fell 47,500 acre-feet and
became equal to the 2019 curve within rounding. No official source located in
this audit states the adoption date. The [2019 sedimentation survey
report](https://www.usbr.gov/tsc/techreferences/reservoir/NavajoReservoir2019SedimentationSurvey_final508VI.pdf)
explains that the preceding 2002 table represented the original reservoir and
that the 2019 survey reduced live capacity by about 56,470 acre-feet through a
combination of sedimentation and improved survey methods.

The official report spans 5,775.00--6,110.00 ft. The processed CSV and active
interpolation contain 33,200 rows from 5,775.00--6,106.99 ft and reproduce both
directions exactly at those knots. The legacy extraction omitted the final 301
rows of the official table, all above the modeled 6,085-ft spill elevation. At
that spillway crest the active interpolation gives 1,647,936.17 acre-feet above
the 5,775-ft deadpool reference. Surface area is
not used to calculate daily evaporation in this model; evaporation enters as a
reported daily volume.

`summary.json` records the numerical checks, date-boundary values, official
URLs, and SHA-256 hashes of the bundled data and style inputs. The official
area-capacity PDF is optional provenance: a bundled copy is recorded when
present, or another copy can be supplied with `--area-capacity-report`. The
figure calculation itself uses the processed table. The PDF omits volatile
date metadata so identical rebuilds have stable checksums.

The present builder verifies the processed CSV against the active interpolation object; it does not repeat extraction from the source PDF.
