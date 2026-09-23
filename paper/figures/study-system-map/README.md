# Study System Map

## Authoritative Source

`study-system-map.pptx` is the author-confirmed editable source, renamed from
`study-system-map-editable-all-elements.pptx` without changing its contents.
The existing `study-system-map.pdf` and `study-system-map.png` are retained
unchanged. Use the PowerPoint for final label placement and subsequent exports;
the Python builder recreates the programmatic layout, not later manual edits.
Earlier PowerPoints are preserved only in the ignored local archive.
The older programmatic SVG is archived as well, because it does not capture
the subsequent manual edits. Background images and donut SVGs remain here as
supporting assets for `export-editable-powerpoint.ps1`, not alternative finals.

This directory contains the programmatic study-system context map for the
paper. The map combines the San Juan basin boundary, the Animas and Navajo
headwater basins, the river network, the Navajo Dam and reservoir location, and
the streamflow gages used to define the modeled system.

The paired-circle inset compares the combined Animas and Navajo drainage area
with their contribution to San Juan River discharge near Bluff. The
drainage-area percentage is calculated directly from the watershed polygons.
The discharge percentage is the ratio of summed common-record daily volume
from San Juan near Archuleta plus Animas at Farmington to observed San Juan
near Bluff volume (1954-12-01 through 2025-11-07).

The north-arrow PNG and SVG in `data/` are simple repository-created geometric assets. The builder uses the PNG while the SVG is retained as the vector source.

## Data attribution

The watershed and river-network snapshots contain information derived from
MERIT Hydro version 1.0.1 through Veins of the Earth. The gage snapshot mixes
public-domain USGS station data with MERIT Hydro mapped reach identifiers. All
three are made available under ODbL 1.0; see
[`data/MERIT-HYDRO-ODBL-NOTICE.md`](data/MERIT-HYDRO-ODBL-NOTICE.md) for the
license URI, modification descriptions, and Yamazaki et al. (2019) citation.
New programmatic exports carry a visible MERIT Hydro notice alongside the USGS
imagery credit. The retained author-edited PowerPoint, PDF, and PNG were not
re-exported solely to add that line; this adjacent README and the paper citation
provide the release notice for those preserved artifacts.

The earlier stand-alone reservoir outline is omitted from the public data package
because adequate source and license provenance was not recovered. Future
programmatic builds instead use the official Navajo Dam coordinates listed on the
Bureau of Reclamation [RISE location page](https://data.usbr.gov/location/423).
The retained author-edited PowerPoint, PDF, and PNG remain the authoritative
figure exports and are not offered as reusable boundary data.

## Programmatic Rebuild

These commands regenerate the original programmatic design and overwrite its
PNG/PDF/SVG exports. They are not needed to use or edit the final PowerPoint.

With the prepared data and imagery cache already present:

```powershell
python -B paper\figures\study-system-map\build.py
```

To refresh the prepared GIS snapshots from the source GeoPackages:

```powershell
python -B paper\figures\study-system-map\build.py `
  --prepare-data `
  --source-gis-dir <path-to-source-geopackages> `
  --prepare-only
```

Add `--refresh-basemap` to redownload the public USGS imagery tiles. Routine
builds use the tracked imagery cache and therefore do not require network
access.

Outputs:

- `study-system-map.png`
- `study-system-map.pdf`
- `study-system-map.svg` (editable text and vector overlays)

## Editable PowerPoint

Generate the furniture-free PNG/JPEG backgrounds and vector donut assets:

```powershell
python -B paper\figures\study-system-map\build.py --export-editable
```

Then create a new PowerPoint for comparison, without overwriting the edited
`study-system-map.pptx`:

```powershell
powershell -ExecutionPolicy Bypass -File paper\figures\study-system-map\export-editable-powerpoint.ps1 `
  -OutputPath paper\figures\study-system-map\study-system-map-rebuilt.pptx
```

Only the satellite basemap and mapped GIS features are fixed in the
PowerPoint background. Geographic labels, Farmington leader lines, imagery
attribution, north arrow, scale bar, map legend, and the complete headwater
contribution inset are named PowerPoint objects. The legend, scale bar, and
inset are grouped for convenient movement while their component shapes and
text remain editable within each group. The two donut rings are independent
SVG objects generated from the same metadata values used by the Python plot.

The exporter defaults to `study-system-map.pptx` for a fresh project, but
refuses to overwrite any existing output. Specify a new `-OutputPath` to
regenerate a comparison version; it does not replace the author-edited source.
