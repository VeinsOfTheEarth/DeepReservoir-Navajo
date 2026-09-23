# Appendix B hydropower figures

`build.py` rebuilds the figures from the active single-efficiency model, the
archived daily Reclamation reservoir export, and the bundled Navajo-only
RectifHyd v1.3 extract. The builder checks that the runtime code's tailwater
knots match `tailwater-knots.csv`. No external analysis directory is needed.

From the repository root:

```powershell
python paper/figures/appendix/b-hydropower/build.py
```

Rebuilding requires NumPy, pandas, and Matplotlib; the script loads
`paper/figure-support/figurestyle.py` for paper fonts and colors. An original
analysis directory can be supplied with `--analysis-root`; when present, the
historical parameter pickle, calibration script, and slide deck are hashed in
`summary.json` but are never used in the calculation.

| Figure | PDF | Evidence and limits |
| --- | --- | --- |
| B1 | `model-response.pdf` | The **implemented approximate** tailwater rating curve through 8,000 cfs (the first five of nine knots), with the historic calibration-flow range shaded, and modeled daily energy against release at the 25th, 50th, and 75th percentiles of historic reservoir elevation. These are model responses, not measured generation or a reproduction of the original plate. |
| B2 | `monthly-fit.pdf` | The fitted model against **RectifHyd monthly estimates** for January 2001–December 2022 (264 months), as a time series and one-to-one scatter comparison. This is an in-sample calibration diagnostic, not held-out validation. |

The model uses one refitted efficiency (`eta_eff = 0.6635421508537132` in the active parameter file), a 1,300-cfs generating-flow cap, and a configured 32-MW power ceiling. `TAILWATER-DIGITIZATION.md` documents the nine approximate knots read from the slide-2 copy of [Plate 7-4 (Tailwater) of the draft USACE Navajo Dam and Reservoir water control manual, revised August 2010](https://water.usace.army.mil/cda/documents/wc/2560/Navajo_WCM_Draft_8-5-10Redacted.pdf). The plate credits USBR drawing 711-D-38. The old calibration script's placeholder curve is preserved in the original analysis folder. [EIA's 2000 inventory](https://www.eia.gov/electricity/archive/009500.pdf) reports two 15-MW nameplate and two 16-MW net seasonal ratings, supporting the historical 32-MW model setting; [current Reclamation records](https://www.usbr.gov/power/data/faclferc.html) list 30 MW. Neither source verifies that 32 MW remains the current achievable output.

`refit_eta.py` reconstructs the same 264 paired monthly estimates and optimizes
the single efficiency over 0.3--0.95 using the audited runtime curve. By default
it only prints the proposed fit:

```powershell
python paper/figures/appendix/b-hydropower/refit_eta.py
```

To update the bundled runtime parameter pickle after reviewing that output,
pass its exact path with `--output`. The optimized efficiency moved from
0.66171254 to 0.66354215; in-sample RMSE remained about 2,782 MWh/month.

RectifHyd's Navajo rows all use `RectifHyd_method = release`. They share
release-timing information with the fitted model and do not independently
validate monthly variation. `summary.json` records the paired-month fit,
source-data flags, model constants, and SHA-256 hashes of all runtime inputs.
PDF date metadata is suppressed for repeatable exports.
