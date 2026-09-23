# Third-party notices

Any license applied to the project's original code does not relicense the third-party data,
fonts, or source material listed below. Copyright and trademark notices are
included for attribution and do not imply endorsement of this project by any
source organization.

## United States government data and imagery

### U.S. Geological Survey

The daily discharge snapshots and stream-gage metadata originate from U.S.
Geological Survey Water Data for the Nation. The study-system map also contains
a cached mosaic from the USGS Imagery Only service in The National Map.

USGS works prepared by federal employees as part of their official duties are
generally in the public domain in the United States. USGS explains its data
licensing policy at <https://www.usgs.gov/data-management/data-licensing>.
USGS states that The National Map services and data are free and in the public
domain and requests this acknowledgment:

> Map services and data available from U.S. Geological Survey, National
> Geospatial Program.

The paper map additionally carries the service attribution “Imagery: USDA,
USGS The National Map.” Source files and station identifiers are listed in
`DATA_SOURCES.md`. USGS observations may be provisional and subject to
revision; use of agency names does not imply endorsement.

### Bureau of Reclamation and U.S. Army Corps of Engineers

Repository inputs derived from Bureau of Reclamation sources include Navajo
Reservoir operations, Navajo Indian Irrigation Project deliveries, the 2019
elevation-area-capacity table, and the Bureau drawing credited by the tailwater
rating plate. The digitized tailwater knots come from Plate 7-4 of the U.S. Army
Corps of Engineers' 2010 *Navajo Dam Water Control Manual*, which attributes
the curve to Bureau of Reclamation Drawing 711-D-38.

These federal materials are cited in `DATA_SOURCES.md`. No warranty is made for
the source data, and operational records should be checked against the current
authoritative source before operational use. The contractor-authored diversion
report cited in `data/sanjuan_irrigation/SOURCE.md` is not redistributed.

### U.S. Energy Information Administration

`data/hydropower/Navajo_Dam_monthly.csv` contains a download of monthly net
generation for Navajo Dam (plant 584). The U.S. Energy Information
Administration states that its government publications and website data are in
the public domain and may be used or distributed with acknowledgment:
<https://www.eia.gov/about/copyrights_reuse.php>.

Suggested acknowledgment: “Source: U.S. Energy Information Administration.”
The EIA name and data source attribution do not imply endorsement, and the EIA
logo is not included or licensed here.

## MERIT Hydro and Veins of the Earth map derivatives

The following compact databases contain information derived from MERIT Hydro
version 1.0.1 through the Veins of the Earth processing interface:

- `paper/figures/study-system-map/data/watersheds.geojson`
- `paper/figures/study-system-map/data/river-network.geojson`
- `paper/figures/study-system-map/data/gages.geojson`

The third file also contains public-domain USGS station metadata, but is
conservatively offered under the same terms because it retains MERIT Hydro
mapped reach identifiers. MERIT Hydro offers a choice of Creative Commons
Attribution-NonCommercial 4.0 or the Open Data Commons Open Database License
1.0. For these derived databases, this project selects the Open Data Commons
Open Database License 1.0 (SPDX identifier `ODbL-1.0`):
<https://opendatacommons.org/licenses/odbl/1-0/>. Publicly used adaptations of
these databases must follow the ODbL attribution and share-alike terms.

> Contains information from MERIT Hydro v1.0.1, made available under the Open
> Data Commons Open Database License (ODbL) 1.0.

MERIT Hydro copyright is held by its developers (2019). The authoritative data
policy and version information are at
<https://global-hydrodynamics.github.io/MERIT_Hydro/>.

Citation: Yamazaki, D., Ikeshima, D., Sosa, J., Bates, P. D., Allen, G. H., and
Pavelsky, T. M. (2019), *MERIT Hydro: A high-resolution global hydrography map
based on latest topography dataset*, *Water Resources Research*, 55, 5053--5073,
<https://doi.org/10.1029/2019WR024873>.

The GeoJSON files contain embedded license, source, and modification fields, and
an adjacent notice is retained at
`paper/figures/study-system-map/data/MERIT-HYDRO-ODBL-NOTICE.md`. The
programmatic map builder places the corresponding notice on new map exports.
No stand-alone reservoir-boundary vector is redistributed. The programmatic map
uses the Bureau of Reclamation RISE location for its Navajo Dam marker, as
documented in `DATA_SOURCES.md` and the map-data README.

## Copernicus ERA5-Land

The following files contain modified Copernicus Climate Change Service
information:

- `data/snow_water_equivalent/Animas_swe_daily.csv`
- `data/snow_water_equivalent/UpperSJ_swe_daily.csv`

Source: Copernicus Climate Change Service (2022), *ERA5-Land hourly data from
1950 to present*, Copernicus Climate Change Service Climate Data Store,
<https://doi.org/10.24381/cds.e2161bac>.

The Climate Data Store identifies the dataset license as Creative Commons
Attribution 4.0; see <https://creativecommons.org/licenses/by/4.0/>. The files
in this repository are basin-aggregated daily derivatives rather than
unmodified ERA5-Land downloads.

Required attribution and disclaimer:

> Contains modified Copernicus Climate Change Service information 2025.
> Neither the European Commission nor ECMWF is responsible for any use that
> may be made of the Copernicus information or data it contains.

## RectifHyd

`data/hydropower/RectifYhd_v1.3_Navajo.csv` is a Navajo Dam subset of
RectifHyd version 1.3, and `data/hydropower/RectifHyd_v_1.3_readme.txt` is its
upstream readme.

Citation: Turner, S., Voisin, N., Nelson, K., and Bracken, C. (2024),
*RectifHyd* (version 1.3) [data set], Zenodo,
<https://doi.org/10.5281/zenodo.11584567>.

The authors' official source repository is
<https://github.com/IMMM-SFA/turner_voisin_nelson_2022_scientific_data>.
Its `LICENSE` file supplies the BSD 2-Clause notice reproduced verbatim in
`data/hydropower/LICENSE-RectifHyd.txt`. The upstream notice labels itself
“RectifHyd v1.0”; that wording is preserved rather than silently altered for
the version 1.3 data snapshot.

## Bundled fonts

The font binaries remain under their own SIL Open Font License, Version 1.1.
The complete notices are stored beside the families:

| Font software | Bundled paths | Upstream | License notice |
| --- | --- | --- | --- |
| Inter 4.001 | `assets/fonts/inter/*.ttf`; derived static instance `paper/figure-support/fonts/Inter-Bold.ttf` | [Google Fonts distribution](https://github.com/google/fonts/tree/main/ofl/inter); [typeface project](https://github.com/rsms/inter) | `assets/fonts/inter/OFL.txt` |
| IBM Plex Sans 3.201 | `assets/fonts/ibmplexsans/*.ttf` | [Google Fonts distribution](https://github.com/google/fonts/tree/main/ofl/ibmplexsans); [typeface project](https://github.com/IBM/plex) | `assets/fonts/ibmplexsans/OFL.txt` |
| Source Sans 3.052 | `assets/fonts/sourcesans3/*.ttf` | [Google Fonts distribution](https://github.com/google/fonts/tree/main/ofl/sourcesans3); [typeface project](https://github.com/adobe-fonts/source-sans) | `assets/fonts/sourcesans3/OFL.txt` |

The static Inter Bold file was generated from the bundled Inter variable font
for deterministic figure rendering and remains covered by the same license.
