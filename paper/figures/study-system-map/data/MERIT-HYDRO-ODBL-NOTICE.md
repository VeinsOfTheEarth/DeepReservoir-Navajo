# MERIT Hydro derivative-data notice

The following compact databases contain information from MERIT Hydro version
1.0.1, processed through the Veins of the Earth interface:

- `watersheds.geojson`
- `river-network.geojson`
- `gages.geojson`

They are made available under the **Open Data Commons Open Database License
(ODbL), version 1.0**:
<https://opendatacommons.org/licenses/odbl/1-0/>.

> Contains information from MERIT Hydro v1.0.1, made available under the Open
> Data Commons Open Database License (ODbL) 1.0.

MERIT Hydro copyright is held by its developers (2019). The authoritative data
policy and version information are at
<https://global-hydrodynamics.github.io/MERIT_Hydro/>.

Citation: Yamazaki, D., Ikeshima, D., Sosa, J., Bates, P. D., Allen, G. H., and
Pavelsky, T. M. (2019), *MERIT Hydro: A high-resolution global hydrography map
based on latest topography dataset*, *Water Resources Research*, 55, 5053--5073,
<https://doi.org/10.1029/2019WR024873>.

## Project modifications

- `watersheds.geojson`: selected three polygons from a ten-basin project layer
  delineated through VotE; retained area attributes, added display labels, and
  reprojected the result to EPSG:4326.
- `river-network.geojson`: selected the San Juan network; retained reach
  identifier, downstream drainage area, reach length, and project mainstem flag;
  and reprojected the result to EPSG:4326.
- `gages.geojson`: selected six USGS gages; retained their VotE/MERIT Hydro
  mapped reach identifiers; added display labels and project roles; and
  reprojected the result to EPSG:4326. USGS-origin station identifiers, names,
  drainage areas, and locations remain public-domain U.S. government data. The
  file is conservatively offered under ODbL because it includes the mapped reach
  identifiers.

The ODbL selection applies to these database contents and their adaptations.
Figures rendered from them are Produced Works; they should retain the notice
above or an equivalent MERIT Hydro/ODbL attribution. The independently sourced
Navajo Dam point marker is documented in [`README.md`](README.md).
