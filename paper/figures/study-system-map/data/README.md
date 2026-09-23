# Map Data Provenance

These compact snapshots were prepared from the GIS layers used in the original
QGIS study-system map:

- `trib_basins.gpkg` -> `watersheds.geojson`
- `powell_rn.gpkg` -> `river-network.geojson`
- `sanjuan_gages.gpkg` and the Animas gage from `all_gages.gpkg` ->
  `gages.geojson`

## MERIT Hydro-derived databases

The watershed and river layers descend from MERIT Hydro version 1.0.1 through
the Veins of the Earth processing interface. Archived project code records use
of VotE's `delineate_basin` routine for the watershed polygons and a VotE river
network export for the reach layer. The exact VotE package and database snapshot
were not retained. The public snapshots select features and attributes, add
project display labels or flags, and reproject the layers to EPSG:4326.

`gages.geojson` combines public-domain USGS station identifiers, names, drainage
areas, and locations with VotE/MERIT Hydro mapped reach identifiers. Because of
those reach identifiers, it is conservatively distributed under the same ODbL
terms as the watershed and river files.

`watersheds.geojson`, `river-network.geojson`, and `gages.geojson` each embed a
source, license, attribution, scope, and modification record. They are made
available under ODbL 1.0. See
[`MERIT-HYDRO-ODBL-NOTICE.md`](MERIT-HYDRO-ODBL-NOTICE.md) for the license URI
and the complete local notice.

## Navajo Dam location

The programmatic builder marks Navajo Dam at longitude -107.53047 and latitude
36.85932, as listed on the Bureau of Reclamation's [RISE location page]
(https://data.usbr.gov/location/423). No stand-alone reservoir-boundary vector is
redistributed because the project's earlier outline lacked adequate source and
license provenance. The retained author-edited map exports remain figure products,
rather than reusable boundary data.

## Derived diagnostics and imagery

`study-system-metadata.json` records source-layer hashes, polygon-derived
drainage areas, and flow diagnostics used by the inset. `build.py --prepare-data`
recomputes the Farmington proxy statistics directly from the bundled Archuleta,
Animas, and original unpatched Farmington daily CSVs. It uses the original
Farmington record so the proxy is not evaluated against values filled by that
same proxy.

The cached basemap uses the public `USGSImageryOnly` tile service from The
National Map. Its request bounds, zoom level, service URL, and attribution are
recorded in the accompanying JSON file.
