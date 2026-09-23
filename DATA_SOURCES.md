# Data sources and provenance

This repository includes compact source-data snapshots and project-derived
products needed to reproduce the Navajo Reservoir model and its paper figures.
Any license applied to the project software does not replace the terms, notices, or citation
requirements of the upstream data and assets listed here. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for license and attribution
text.

Some legacy exports did not retain an exact download timestamp. In those cases,
the coverage date and timestamp embedded in the filename are reported rather
than inventing an access date. The files are reproducibility snapshots; users
should query the authoritative source for current or revised observations.

## Bureau of Reclamation water operations

| Repository files | Upstream source | Coverage and processing |
| --- | --- | --- |
| `data/navajo_reservoir_historic/Clipped_NAVAJORESERVOIR08-18-2024T16.48.23.csv` | [Upper Colorado Region Water Operations Historic Data](https://www.usbr.gov/rsvrWater/HistoricalApp.html); current records are also cataloged in [Reclamation Information Sharing Environment](https://data.usbr.gov/catalog/2392/item/613) | Daily Navajo Reservoir elevation, storage, evaporation, inflow, and total release, June 7, 1967--August 17, 2024. The repository preserves the downloaded, date-clipped CSV. Reclamation describes these operational data as provisional and subject to revision. |
| `data/niip/NAVAJOINDIANIRRIGATIONPROJECT07-17-2025T13.21.47.csv` | [Upper Colorado Region HydroData](https://www.usbr.gov/uc/water/hydrodata/) (site: Navajo Indian Irrigation Project) | Daily reported project flow, October 1, 1975--July 16, 2025. The model uses this historical delivery record as its daily demand/delivery proxy. |
| `data/elevation_area_storage_relationships/elevation_storage_area_2019.csv` and `2019_elevation_area_capacity.pkl` | Bureau of Reclamation, *Navajo Reservoir (New Mexico) Sedimentation Survey ACAP Table 2019*, [RISE item 133891](https://data.usbr.gov/catalog/8069/item/133891); see also the [Technical Service Center reservoir surveys index](https://www.usbr.gov/tsc/techreferences/reservoir.html) | The CSV transcribes the 2019 elevation--area--capacity table. The pickle stores project-generated interpolation objects derived from that table. Runtime preprocessing uses piecewise-linear interpolation with endpoint clamping. The source PDF is not redistributed in this repository. |

The daily reservoir export changes its reported elevation--storage relationship
on October 1, 2021. `src/deepreservoir/data/storage_datum.py` documents and
implements the optional remapping of all reported elevations through the 2019
area-capacity relationship.

## U.S. Geological Survey streamflow

The files in `data/daily_flows/` are processed daily mean discharge series
(parameter 00060, cubic feet per second) from [USGS Water Data for the
Nation](https://waterdata.usgs.gov/). The stable database citation is U.S.
Geological Survey, *USGS Water Data for the Nation*,
[https://doi.org/10.5066/F7P55KJN](https://doi.org/10.5066/F7P55KJN).

| File | USGS site | File coverage |
| --- | --- | --- |
| `daily_sj_archuleta.csv` | [09355500, San Juan River near Archuleta, New Mexico](https://waterdata.usgs.gov/monitoring-location/USGS-09355500/) | 1954-12-01 to 2025-11-07 |
| `daily_sj_farmington.csv` | [09365000, San Juan River at Farmington, New Mexico](https://waterdata.usgs.gov/monitoring-location/USGS-09365000/) | 1930-10-01 to 2025-11-07; contains the documented gap filling below |
| `daily_sj_fourcorners.csv` | [09371010, San Juan River at Four Corners, Colorado](https://waterdata.usgs.gov/monitoring-location/USGS-09371010/) | 1977-10-01 to 2025-11-07 |
| `daily_sj_shiprock.csv` | [09368000, San Juan River at Shiprock, New Mexico](https://waterdata.usgs.gov/monitoring-location/USGS-09368000/) | 1934-10-01 to 2025-11-07 |
| `daily_sj_bluff.csv` | [09379500, San Juan River near Bluff, Utah](https://waterdata.usgs.gov/monitoring-location/USGS-09379500/) | 1914-10-30 to 2025-11-07 |
| `daily_animas_farmington.csv` | [09364500, Animas River at Farmington, New Mexico](https://waterdata.usgs.gov/monitoring-location/USGS-09364500/) | 1913-10-01 to 2025-11-07 |
| `daily_chaco_waterflow.csv` | [09367950, Chaco River near Waterflow, New Mexico](https://waterdata.usgs.gov/monitoring-location/USGS-09367950/) | 1975-11-01 to 1994-10-11 |
| `daily_chinle_mexicanwater.csv` | [09379200, Chinle Creek near Mexican Water, Arizona](https://waterdata.usgs.gov/monitoring-location/USGS-09379200/) | 1964-10-01 to 2025-11-07 |
| `daily_laplata_farmington.csv` | [09367500, La Plata River near Farmington, New Mexico](https://waterdata.usgs.gov/monitoring-location/USGS-09367500/) | 1938-03-01 to 2025-11-07 |
| `daily_mancos_towaoc.csv` | [09371000, Mancos River near Towaoc, Colorado](https://waterdata.usgs.gov/monitoring-location/USGS-09371000/) | 1921-04-01 to 2025-11-07 |

`data/patch_sanjuan_at_farmington/` retains the original Farmington series,
scripts, diagnostics, and machine-readable summaries for the project's gap
filling. The processed Farmington file replaces 822 missing days with the sum
of San Juan near Archuleta and Animas at Farmington: 811 consecutive days from
August 20, 2020 through November 8, 2022, plus eleven isolated days through
March 5, 2025. `patch_summary.json` lists every affected range. These values are
project-derived estimates, not USGS observations.

## ERA5-Land snow-water equivalent

`data/snow_water_equivalent/Animas_swe_daily.csv` and
`UpperSJ_swe_daily.csv` are processed daily basin snapshots of the ERA5-Land
`snow_depth_water_equivalent` variable, in meters, covering January 1, 1950
through October 29, 2025. The source workflow sampled ERA5-Land over the Animas
and upper San Juan basin polygons through Google Earth Engine and reduced the
output to one value per basin and day. The original task files and complete
processing environment are not part of this release, so the two CSVs should be
treated as retained study inputs rather than as a stand-alone reproduction of
the ERA5-Land processing pipeline.

Source: Copernicus Climate Change Service, *ERA5-Land hourly data from 1950 to
present*, Climate Data Store,
[https://doi.org/10.24381/cds.e2161bac](https://doi.org/10.24381/cds.e2161bac).
The repository files are modified Copernicus products and are distributed with
the attribution and disclaimer in `THIRD_PARTY_NOTICES.md`.

## Hydropower data and calibration

| Repository files | Upstream source and processing |
| --- | --- |
| `data/hydropower/Navajo_Dam_monthly.csv` | U.S. Energy Information Administration Electricity Data Browser, plant 584 (Navajo Dam), monthly net generation: [plant page](https://www.eia.gov/electricity/data/browser/#/plant/584/?freq=M&pin=). The retained export spans January 2001--February 2025; recent entries may be blank. |
| `data/hydropower/RectifYhd_v1.3_Navajo.csv` and `RectifHyd_v_1.3_readme.txt` | Navajo Dam subset and upstream readme from Turner, Voisin, Nelson, and Bracken (2024), *RectifHyd*, version 1.3, [https://doi.org/10.5281/zenodo.11584567](https://doi.org/10.5281/zenodo.11584567). The subset contains 264 months, January 2001--December 2022. The filename's `RectifYhd` spelling is retained for reproducibility. |
| `data/hydropower/hydropower_parameters.pkl` | Project-generated single-efficiency calibration. The effective efficiency (`eta_eff = 0.6635421508537132`) was fit to the 264 RectifHyd monthly values using Reclamation daily elevation and release, a 1,300-cubic-feet-per-second turbine-flow cap, a 32-megawatt instantaneous capacity cap, and the tailwater curve below. The pickle stores the calibration window, constants, notes, and source hashes. |

The tailwater knots in `src/deepreservoir/define_env/hydropower_model.py` were
manually approximated from Plate 7-4, “Navajo Dam and Reservoir — Tailwater,”
in the U.S. Army Corps of Engineers' 2010 *Navajo Dam Water Control Manual*.
The plate identifies Bureau of Reclamation Drawing 711-D-38 as the source of
the plotted curve. The code linearly interpolates between the digitized knots.
The manual is available from the [Corps Water Management System document
archive](https://water.usace.army.mil/cda/documents/wc/2560/Navajo_WCM_Draft_8-5-10Redacted.pdf).

## Study-system map

The map support directory `paper/figures/study-system-map/data/` contains:

- compact GeoJSON snapshots prepared from the original project GeoPackages;
- USGS gage identifiers and locations in `gages.geojson`;
- `study-system-metadata.json`, containing project-computed polygon areas and
  flow-volume diagnostics; and
- a cached USGS Imagery Only tile mosaic, `usgs-imagery-z10.jpg`, with its
  request bounds, zoom, tile range, service URL, and attribution in
  `usgs-imagery-z10.json`.

The imagery service is the [USGS Imagery Only basemap](https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer)
from The National Map. The map itself credits “Imagery: USDA, USGS The National
Map.” USGS requests the acknowledgment “Map services and data available from
U.S. Geological Survey, National Geospatial Program.”

`watersheds.geojson` and `river-network.geojson` descend from MERIT Hydro
version 1.0.1 through the Veins of the Earth processing interface. Archived
project code records that the watershed layer was generated by calling VotE's
`delineate_basin` routine for the Lake Powell inflow, Animas, and Navajo reach
identifiers, and that the river layer was selected from a VotE river-network
export. The exact VotE package and database snapshot were not retained. The
release files select features and attributes from those project layers, add
display labels or flags, and reproject them to longitude/latitude coordinates.
`gages.geojson` combines public-domain USGS station information with
VotE/MERIT Hydro mapped reach identifiers.

MERIT Hydro is distributed under a choice of Creative Commons
Attribution-NonCommercial 4.0 or the Open Data Commons Open Database License
1.0. This repository selects ODbL 1.0 for `watersheds.geojson`,
`river-network.geojson`, and, conservatively, the mixed-origin `gages.geojson`.
The license URI, file-level notices, modification descriptions, and citation are
in [`paper/figures/study-system-map/data/MERIT-HYDRO-ODBL-NOTICE.md`](paper/figures/study-system-map/data/MERIT-HYDRO-ODBL-NOTICE.md).
The map is a Produced Work and the programmatic builder places a visible MERIT
Hydro attribution on newly generated map exports.

Citation: Yamazaki, D., Ikeshima, D., Sosa, J., Bates, P. D., Allen, G. H., and
Pavelsky, T. M. (2019), *MERIT Hydro: A high-resolution global hydrography map
based on latest topography dataset*, *Water Resources Research*, 55, 5053--5073,
[https://doi.org/10.1029/2019WR024873](https://doi.org/10.1029/2019WR024873).

The public package does not redistribute the project's earlier stand-alone
reservoir outline because adequate source and license provenance was not
recovered. The programmatic builder instead marks Navajo Dam at longitude
-107.53047 and latitude 36.85932 from the Bureau of Reclamation's
[RISE location page](https://data.usbr.gov/location/423). The retained
author-edited map exports are figure products, not reusable boundary data. The
compact GeoJSON files are not authoritative releases of their upstream data.

## San Juan and Animas diversion summaries

The two CSVs in `data/sanjuan_irrigation/` are project-derived summaries of
reported diversion information used during the hydrologic-simplification
analysis. Their source is Lyons, Farrington, Platania, and Gori (2016), *San
Juan and Animas Rivers Diversion Study: Final Report*, prepared for the Bureau
of Reclamation and the San Juan River Basin Recovery Implementation Program.
The contractor-authored report is not redistributed; its citation and official
public link are retained in `data/sanjuan_irrigation/SOURCE.md`.

## Project-derived data

Files below `artifacts/`, figure-specific `data/` directories, and the remaining
JSON, CSV, Parquet, and pickle files not named above are outputs or compact
intermediate products created by this project from the cited inputs. Their
provenance and rebuild path are documented in `ARTIFACT_MANIFEST.md`,
`REPRODUCIBILITY.md`, and the README beside each figure. Python pickle files
must be treated as trusted release artifacts; do not load modified or
untrusted pickle files.
