"""Build the programmatic study-system context map.

The default build uses compact GIS snapshots and a cached USGS imagery mosaic
stored beside this script. Use ``--prepare-data`` to recreate those snapshots
from the source GeoPackages and ``--refresh-basemap`` to refresh the imagery.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

# Conda environments on Windows keep GDAL/PROJ data under Library/share. Setting these
# before importing geopandas avoids fragile machine-level configuration.
os.environ.setdefault("GDAL_DATA", str(Path(sys.prefix) / "Library" / "share" / "gdal"))
os.environ.setdefault("PROJ_LIB", str(Path(sys.prefix) / "Library" / "share" / "proj"))

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image, ImageEnhance
import numpy as np


OUTPUT_STEM = "study-system-map"
WEB_MERCATOR_HALF_WORLD = 20037508.342789244
BASEMAP_ZOOM = 10
BASEMAP_URL = (
    "https://basemap.nationalmap.gov/arcgis/rest/services/"
    "USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"
)
BASEMAP_ATTRIBUTION = "Imagery: USDA, USGS The National Map"
MERIT_HYDRO_VERSION = "1.0.1"
MERIT_HYDRO_URL = "https://global-hydrodynamics.github.io/MERIT_Hydro/"
MERIT_HYDRO_DOI = "https://doi.org/10.1029/2019WR024873"
MERIT_HYDRO_LICENSE = "ODbL-1.0"
MERIT_HYDRO_LICENSE_URL = "https://opendatacommons.org/licenses/odbl/1-0/"
USBR_RISE_NAVAJO_DAM_URL = "https://data.usbr.gov/location/423"
NAVAJO_DAM_LONGITUDE = -107.53047
NAVAJO_DAM_LATITUDE = 36.85932
MERIT_HYDRO_ATTRIBUTION = (
    "Contains information from MERIT Hydro v1.0.1, made available under "
    "the Open Data Commons Open Database License (ODbL) 1.0."
)
MAP_ATTRIBUTION = (
    f"{BASEMAP_ATTRIBUTION}\n"
    "Hydrography contains information from MERIT Hydro v1.0.1 (ODbL 1.0)\n"
    "Navajo Dam location: Bureau of Reclamation RISE"
)

TEXT = "#111827"
MUTED = "#64748B"
BLUE = "#2B6CB0"
BLUE_LIGHT = "#7DD3FC"
HEADWATER = "#FACC15"
BOUNDARY = "#F8FAFC"
RESERVOIR = "#38BDF8"
GAGE = "#FB7185"
OTHER_BASINS = "#CBD5E1"

WATERSHED_PATH = DATA_DIR / "watersheds.geojson"
RIVER_PATH = DATA_DIR / "river-network.geojson"
GAGE_PATH = DATA_DIR / "gages.geojson"
NORTH_ARROW_PATH = DATA_DIR / "north-arrow-white.png"
METADATA_PATH = DATA_DIR / "study-system-metadata.json"
BASEMAP_PATH = DATA_DIR / f"usgs-imagery-z{BASEMAP_ZOOM}.jpg"
BASEMAP_META_PATH = DATA_DIR / f"usgs-imagery-z{BASEMAP_ZOOM}.json"

REQUIRED_DATA = [
    WATERSHED_PATH,
    RIVER_PATH,
    GAGE_PATH,
    NORTH_ARROW_PATH,
    METADATA_PATH,
]

GAGE_NAMES = {
    "09355500": "SJ nr\nArchuleta",
    "09364500": "Animas at\nFarmington",
    "09365000": "SJ at Farmington",
    "09368000": "SJ at Shiprock",
    "09371010": "SJ at Four Corners",
    "09379500": "SJ nr Bluff",
}

# Keep portable data labels free of plot-specific abbreviations and line breaks.
# The map itself continues to use GAGE_NAMES for compact label placement.
GAGE_DISPLAY_NAMES = {
    "09355500": "SJ near Archuleta",
    "09364500": "Animas at Farmington",
    "09365000": "SJ at Farmington",
    "09368000": "SJ at Shiprock",
    "09371010": "SJ at Four Corners",
    "09379500": "SJ near Bluff",
}

KEY_GAGES = {"09355500", "09364500", "09365000", "09379500"}

GAGE_LABEL_AXES = {
    "09355500": (0.743789, 0.471000),
    "09364500": (0.572144, 0.570000),
    "09365000": (0.556131, 0.433000),
    "09368000": (0.456662, 0.486667),
    "09371010": (0.361802, 0.573000),
    "09379500": (0.213812, 0.622333),
}

GAGE_LEADER_AXES = {
    "09364500": ((0.617188, 0.483381), (0.579343, 0.539614)),
    "09365000": ((0.614007, 0.447047), (0.586007, 0.477047)),
}


def _register_fonts() -> None:
    font_dir = REPO_ROOT / "assets" / "fonts"
    if font_dir.is_dir():
        for font_path in font_dir.rglob("*.ttf"):
            try:
                fm.fontManager.addfont(str(font_path))
            except Exception:
                pass
    preferred = ["Inter", "Source Sans 3", "IBM Plex Sans", "DejaVu Sans"]
    available: list[str] = []
    for name in preferred:
        try:
            fm.findfont(fm.FontProperties(family=name), fallback_to_default=False)
            available.append(name)
        except Exception:
            pass
    if "DejaVu Sans" not in available:
        available.append("DejaVu Sans")
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": available,
            "font.size": 11.0,
            "axes.labelsize": 11.0,
            "legend.fontsize": 10.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def _write_geojson(
    frame: object,
    path: Path,
    *,
    merit_hydro_modifications: str | None = None,
    license_scope: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    frame.to_file(path, driver="GeoJSON", index=False)

    if merit_hydro_modifications is None:
        return

    # GeoJSON permits foreign members. Keep the ODbL notice inside each derived
    # database so it remains attached when the file is copied independently.
    raw = path.read_bytes().decode("utf-8")
    newline = "\r\n" if "\r\n" in raw else "\n"
    marker = '"features": ['
    if raw.count(marker) != 1:
        raise RuntimeError(f"Unexpected GeoJSON structure in {path}")
    notice = {
        "attribution": MERIT_HYDRO_ATTRIBUTION,
        "license": MERIT_HYDRO_LICENSE,
        "license_url": MERIT_HYDRO_LICENSE_URL,
        "source": {
            "dataset": "MERIT Hydro",
            "version": MERIT_HYDRO_VERSION,
            "url": MERIT_HYDRO_URL,
            "citation_doi": MERIT_HYDRO_DOI,
            "processing_interface": "Veins of the Earth (VotE)",
        },
        "modifications": merit_hydro_modifications,
        "license_scope": license_scope
        or "Applies to the database contents in this file.",
    }
    insertion = "".join(
        f"{json.dumps(key)}: "
        f"{json.dumps(value, ensure_ascii=False, separators=(',', ':'))},{newline}"
        for key, value in notice.items()
    )
    path.write_bytes(raw.replace(marker, insertion + marker, 1).encode("utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prepare_data(source_gis_dir: Path) -> None:
    """Create compact, paper-facing GIS snapshots from the source layers."""

    try:
        import geopandas as gpd
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "Preparing the GIS snapshots requires geopandas. Install it in the active environment."
        ) from exc

    source_gis_dir = source_gis_dir.resolve()
    expected = {
        "watersheds": source_gis_dir / "trib_basins.gpkg",
        "rivers": source_gis_dir / "powell_rn.gpkg",
        "gages": source_gis_dir / "sanjuan_gages.gpkg",
        "all_gages": source_gis_dir / "all_gages.gpkg",
    }
    missing = [str(path) for path in expected.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing source GIS files:\n" + "\n".join(missing))

    watersheds = gpd.read_file(expected["watersheds"])
    watersheds = watersheds[
        watersheds["name"].isin(["Powell Inflow", "Animus River", "Navajo Outflow"])
    ][["id_basin", "name", "area_km2", "geometry"]].copy()
    watersheds["display_name"] = watersheds["name"].replace(
        {
            "Powell Inflow": "San Juan basin above Lake Powell",
            "Animus River": "Animas basin",
            "Navajo Outflow": "Navajo basin",
        }
    )
    _write_geojson(
        watersheds.to_crs(4326),
        WATERSHED_PATH,
        merit_hydro_modifications=(
            "Selected three polygons from a ten-basin project layer delineated "
            "through VotE; retained area attributes, added display labels, and "
            "reprojected the result to EPSG:4326."
        ),
    )

    rivers = gpd.read_file(expected["rivers"])[
        ["id_reach", "drainarea_ds_km2", "length_km", "san_juan", "geometry"]
    ].copy()
    rivers["san_juan"] = rivers["san_juan"].fillna(False).astype(bool)
    _write_geojson(
        rivers.to_crs(4326),
        RIVER_PATH,
        merit_hydro_modifications=(
            "Selected the San Juan network and retained reach identifier, "
            "downstream drainage area, reach length, and project mainstem flag; "
            "reprojected the result to EPSG:4326."
        ),
    )

    gages = gpd.read_file(expected["gages"])
    gages = gages[gages["id_source"].notna()][
        [
            "id_gage",
            "id_source",
            "station_name",
            "drainarea_km2_source",
            "mapped_id_reach",
            "geometry",
        ]
    ].copy()

    all_gages = gpd.read_file(
        expected["all_gages"],
        columns=[
            "id_gage",
            "id_source",
            "station_name",
            "drainarea_km2_source",
            "mapped_id_reach",
        ],
    )
    animas = all_gages[all_gages["id_source"].astype(str).eq("09364500")].copy()
    gages = pd.concat([gages, animas], ignore_index=True)
    gages = gpd.GeoDataFrame(gages, geometry="geometry", crs=4326)
    gages["id_source"] = gages["id_source"].astype(str)
    gages = gages[gages["id_source"].isin(GAGE_DISPLAY_NAMES)].drop_duplicates("id_source")
    gages["display_name"] = gages["id_source"].map(GAGE_DISPLAY_NAMES)
    gages["role"] = np.where(gages["id_source"].isin(KEY_GAGES), "key", "supporting")
    _write_geojson(
        gages.to_crs(4326),
        GAGE_PATH,
        merit_hydro_modifications=(
            "Selected six USGS gages, retained their VotE/MERIT Hydro mapped "
            "reach identifiers, added project display labels and roles, and "
            "reprojected the result to EPSG:4326."
        ),
        license_scope=(
            "Applied conservatively to this mixed-origin database because it "
            "contains MERIT Hydro-derived mapped reach identifiers; USGS-origin "
            "station identifiers, names, drainage areas, and locations remain "
            "public-domain U.S. government data."
        ),
    )

    areas = watersheds.set_index("name")["area_km2"].astype(float)
    productive_area = float(areas["Animus River"] + areas["Navajo Outflow"])
    total_area = float(areas["Powell Inflow"])
    area_share_pct = 100.0 * productive_area / total_area

    daily_flow_dir = REPO_ROOT / "data" / "daily_flows"

    def load_daily_flow(path: Path, column: str) -> object:
        frame = pd.read_csv(path)
        frame["time"] = pd.to_datetime(frame["time"], errors="raise")
        frame[column] = pd.to_numeric(frame["value"], errors="coerce")
        return frame[["time", column]].dropna()

    # Recompute the proxy diagnostics from bundled daily series. Use the
    # retained original Farmington record so gap-filled proxy values are never
    # compared against themselves.
    farmington_original_path = (
        REPO_ROOT
        / "data"
        / "patch_sanjuan_at_farmington"
        / "daily_sj_farmington_original_backup.csv"
    )
    farmington_proxy = (
        load_daily_flow(daily_flow_dir / "daily_sj_archuleta.csv", "archuleta_cfs")
        .merge(
            load_daily_flow(
                daily_flow_dir / "daily_animas_farmington.csv", "animas_cfs"
            ),
            on="time",
            how="inner",
        )
        .merge(
            load_daily_flow(farmington_original_path, "farmington_cfs"),
            on="time",
            how="inner",
        )
    )
    farmington_columns = ["archuleta_cfs", "animas_cfs", "farmington_cfs"]
    farmington_proxy[farmington_columns] = farmington_proxy[
        farmington_columns
    ].clip(lower=0.0)
    farmington_proxy["proxy_cfs"] = (
        farmington_proxy["archuleta_cfs"] + farmington_proxy["animas_cfs"]
    )
    daily_error = farmington_proxy["proxy_cfs"] - farmington_proxy["farmington_cfs"]
    daily_sst = float(
        (
            farmington_proxy["farmington_cfs"]
            - farmington_proxy["farmington_cfs"].mean()
        ).pow(2).sum()
    )
    daily_r2 = 1.0 - float(daily_error.pow(2).sum()) / daily_sst
    mean_proxy_pct = 100.0 * float(farmington_proxy["proxy_cfs"].mean()) / float(
        farmington_proxy["farmington_cfs"].mean()
    )

    cubic_feet_per_second_day_to_acre_feet = 1.983471074
    annual_proxy = farmington_proxy.assign(
        year=farmington_proxy["time"].dt.year,
        observed_af=(
            farmington_proxy["farmington_cfs"]
            * cubic_feet_per_second_day_to_acre_feet
        ),
        proxy_af=(
            farmington_proxy["proxy_cfs"]
            * cubic_feet_per_second_day_to_acre_feet
        ),
        day_count=1,
    ).groupby("year").agg(
        observed_af=("observed_af", "sum"),
        proxy_af=("proxy_af", "sum"),
        n_days=("day_count", "sum"),
    )
    expected_days = annual_proxy.index.to_series().map(
        lambda year: (
            366
            if pd.Timestamp(year=int(year), month=12, day=31).is_leap_year
            else 365
        )
    )
    annual_proxy = annual_proxy.loc[annual_proxy["n_days"] == expected_days].copy()
    annual_error = annual_proxy["proxy_af"] - annual_proxy["observed_af"]
    annual_sst = float(
        (annual_proxy["observed_af"] - annual_proxy["observed_af"].mean())
        .pow(2)
        .sum()
    )
    annual_r2 = 1.0 - float(annual_error.pow(2).sum()) / annual_sst
    annual_mean_bias_pct = 100.0 * float(annual_error.mean()) / float(
        annual_proxy["observed_af"].mean()
    )

    bluff_contribution = (
        load_daily_flow(daily_flow_dir / "daily_sj_archuleta.csv", "archuleta_cfs")
        .merge(
            load_daily_flow(
                daily_flow_dir / "daily_animas_farmington.csv", "animas_cfs"
            ),
            on="time",
            how="inner",
        )
        .merge(
            load_daily_flow(daily_flow_dir / "daily_sj_bluff.csv", "bluff_cfs"),
            on="time",
            how="inner",
        )
    )
    valid = (
        (bluff_contribution["archuleta_cfs"] >= 0.0)
        & (bluff_contribution["animas_cfs"] >= 0.0)
        & (bluff_contribution["bluff_cfs"] > 0.0)
    )
    bluff_contribution = bluff_contribution.loc[valid].copy()
    bluff_contribution["headwater_cfs"] = (
        bluff_contribution["archuleta_cfs"] + bluff_contribution["animas_cfs"]
    )
    bluff_share_pct = 100.0 * float(
        bluff_contribution["headwater_cfs"].sum()
    ) / float(bluff_contribution["bluff_cfs"].sum())

    metadata = {
        "source_layers": {
            "watersheds": "trib_basins.gpkg",
            "rivers": "powell_rn.gpkg",
            "mainstem_gages": "sanjuan_gages.gpkg",
            "animas_gage": "all_gages.gpkg",
        },
        "source_layer_sha256": {
            "watersheds": _sha256_file(expected["watersheds"]),
            "rivers": _sha256_file(expected["rivers"]),
            "mainstem_gages": _sha256_file(expected["gages"]),
            "animas_gage": _sha256_file(expected["all_gages"]),
        },
        "geospatial_provenance": {
            "merit_hydro": {
                "version": MERIT_HYDRO_VERSION,
                "source_url": MERIT_HYDRO_URL,
                "citation_doi": MERIT_HYDRO_DOI,
                "processing_interface": "Veins of the Earth (VotE)",
                "exact_vote_database_snapshot_retained": False,
                "derived_files": [
                    "watersheds.geojson",
                    "river-network.geojson",
                    "gages.geojson",
                ],
                "license": MERIT_HYDRO_LICENSE,
                "license_url": MERIT_HYDRO_LICENSE_URL,
                "notice": MERIT_HYDRO_ATTRIBUTION,
            },
            "navajo_dam_location": {
                "longitude": NAVAJO_DAM_LONGITUDE,
                "latitude": NAVAJO_DAM_LATITUDE,
                "source_url": USBR_RISE_NAVAJO_DAM_URL,
                "description": (
                    "Official Navajo Dam location used for the programmatic map "
                    "marker. No stand-alone reservoir-boundary vector is "
                    "redistributed."
                ),
            },
        },
        "areas_km2": {
            "san_juan_above_powell": total_area,
            "animas": float(areas["Animus River"]),
            "navajo": float(areas["Navajo Outflow"]),
            "animas_plus_navajo": productive_area,
        },
        "animas_plus_navajo_area_share_pct": area_share_pct,
        "bluff_headwater_contribution": {
            "definition": "San Juan near Archuleta plus Animas at Farmington",
            "observed": "San Juan near Bluff",
            "overlap_start": bluff_contribution["time"].min().date().isoformat(),
            "overlap_end": bluff_contribution["time"].max().date().isoformat(),
            "n_days": int(len(bluff_contribution)),
            "volume_share_pct": bluff_share_pct,
        },
        "farmington_proxy": {
            "definition": "San Juan near Archuleta plus Animas at Farmington",
            "observed": "San Juan at Farmington",
            "observed_source": (
                "data/patch_sanjuan_at_farmington/"
                "daily_sj_farmington_original_backup.csv"
            ),
            "overlap_start": farmington_proxy["time"].min().date().isoformat(),
            "overlap_end": farmington_proxy["time"].max().date().isoformat(),
            "n_days": int(len(farmington_proxy)),
            "daily_r2": daily_r2,
            "mean_proxy_pct_of_observed": mean_proxy_pct,
            "annual_complete_calendar_years": {
                "n_years": int(len(annual_proxy)),
                "start_year": int(annual_proxy.index.min()),
                "end_year": int(annual_proxy.index.max()),
            },
            "annual_r2": annual_r2,
            "annual_mean_bias_pct_of_observed": annual_mean_bias_pct,
        },
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _lon_to_tile_x(lon: float, zoom: int) -> int:
    return int(math.floor((lon + 180.0) / 360.0 * (2**zoom)))


def _lat_to_tile_y(lat: float, zoom: int) -> int:
    lat_rad = math.radians(max(min(lat, 85.05112878), -85.05112878))
    value = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0
    return int(math.floor(value * (2**zoom)))


def _tile_bounds_mercator(
    x_min: int,
    x_max: int,
    y_min: int,
    y_max: int,
    zoom: int,
) -> tuple[float, float, float, float]:
    n = float(2**zoom)
    left = (2.0 * x_min / n - 1.0) * WEB_MERCATOR_HALF_WORLD
    right = (2.0 * (x_max + 1) / n - 1.0) * WEB_MERCATOR_HALF_WORLD
    top = (1.0 - 2.0 * y_min / n) * WEB_MERCATOR_HALF_WORLD
    bottom = (1.0 - 2.0 * (y_max + 1) / n) * WEB_MERCATOR_HALF_WORLD
    return left, right, bottom, top


def _download_basemap(
    bounds_lonlat: tuple[float, float, float, float],
    *,
    zoom: int,
) -> None:
    import requests

    west, south, east, north = bounds_lonlat
    x_min = _lon_to_tile_x(west, zoom)
    x_max = _lon_to_tile_x(east, zoom)
    y_min = _lat_to_tile_y(north, zoom)
    y_max = _lat_to_tile_y(south, zoom)

    width_tiles = x_max - x_min + 1
    height_tiles = y_max - y_min + 1
    mosaic = Image.new("RGB", (256 * width_tiles, 256 * height_tiles))
    session = requests.Session()
    session.headers.update({"User-Agent": "DeepReservoir paper figure builder"})

    for tile_y in range(y_min, y_max + 1):
        for tile_x in range(x_min, x_max + 1):
            url = BASEMAP_URL.format(z=zoom, y=tile_y, x=tile_x)
            response = session.get(url, timeout=30)
            response.raise_for_status()
            tile = Image.open(io.BytesIO(response.content)).convert("RGB")
            offset = ((tile_x - x_min) * 256, (tile_y - y_min) * 256)
            mosaic.paste(tile, offset)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    mosaic.save(BASEMAP_PATH, quality=92, optimize=True)
    extent = _tile_bounds_mercator(x_min, x_max, y_min, y_max, zoom)
    BASEMAP_META_PATH.write_text(
        json.dumps(
            {
                "service_url": BASEMAP_URL,
                "attribution": BASEMAP_ATTRIBUTION,
                "zoom": zoom,
                "requested_bounds_lonlat": list(bounds_lonlat),
                "tile_range": {
                    "x_min": x_min,
                    "x_max": x_max,
                    "y_min": y_min,
                    "y_max": y_max,
                },
                "extent_epsg3857": list(extent),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _outlined_text(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    color: str = "white",
    fontsize: float = 8.5,
    fontweight: str = "normal",
    fontstyle: str = "normal",
    ha: str = "center",
    va: str = "center",
    rotation: float = 0.0,
    zorder: int = 20,
    transform: object | None = None,
    stroke_width: float = 2.2,
    stroke_color: str = "#111827",
    stroke_alpha: float = 0.82,
) -> mpl.text.Text:
    artist = ax.text(
        x,
        y,
        text,
        color=color,
        fontsize=fontsize,
        fontweight=fontweight,
        fontstyle=fontstyle,
        ha=ha,
        va=va,
        rotation=rotation,
        zorder=zorder,
        transform=ax.transData if transform is None else transform,
    )
    if stroke_width > 0.0:
        artist.set_path_effects(
            [
                path_effects.withStroke(
                    linewidth=stroke_width,
                    foreground=stroke_color,
                    alpha=stroke_alpha,
                )
            ]
        )
    return artist


def _project_lonlat(lon: float, lat: float) -> tuple[float, float]:
    lat = max(min(float(lat), 85.05112878), -85.05112878)
    x = WEB_MERCATOR_HALF_WORLD * float(lon) / 180.0
    y = WEB_MERCATOR_HALF_WORLD * math.log(
        math.tan(math.pi / 4.0 + math.radians(lat) / 2.0)
    ) / math.pi
    return x, y


def _load_features(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload["features"])


def _line_parts(geometry: dict[str, object]) -> list[list[list[float]]]:
    geom_type = str(geometry["type"])
    coordinates = geometry["coordinates"]
    if geom_type == "LineString":
        return [coordinates]  # type: ignore[list-item]
    if geom_type == "MultiLineString":
        return list(coordinates)  # type: ignore[arg-type]
    raise ValueError(f"Expected line geometry, received {geom_type}")


def _polygon_parts(geometry: dict[str, object]) -> list[list[list[list[float]]]]:
    geom_type = str(geometry["type"])
    coordinates = geometry["coordinates"]
    if geom_type == "Polygon":
        return [coordinates]  # type: ignore[list-item]
    if geom_type == "MultiPolygon":
        return list(coordinates)  # type: ignore[arg-type]
    raise ValueError(f"Expected polygon geometry, received {geom_type}")


def _feature_coordinates(feature: dict[str, object]) -> list[tuple[float, float]]:
    geometry = feature["geometry"]
    geom_type = str(geometry["type"])  # type: ignore[index]
    if geom_type == "Point":
        lon, lat = geometry["coordinates"]  # type: ignore[index]
        return [(float(lon), float(lat))]
    if geom_type in {"LineString", "MultiLineString"}:
        return [
            (float(lon), float(lat))
            for part in _line_parts(geometry)  # type: ignore[arg-type]
            for lon, lat in part
        ]
    if geom_type in {"Polygon", "MultiPolygon"}:
        return [
            (float(lon), float(lat))
            for polygon in _polygon_parts(geometry)  # type: ignore[arg-type]
            for ring in polygon
            for lon, lat in ring
        ]
    raise ValueError(f"Unsupported geometry type: {geom_type}")


def _features_bounds(
    features: list[dict[str, object]],
    *,
    projected: bool,
) -> tuple[float, float, float, float]:
    coords = [coord for feature in features for coord in _feature_coordinates(feature)]
    if projected:
        coords = [_project_lonlat(lon, lat) for lon, lat in coords]
    xs = [coord[0] for coord in coords]
    ys = [coord[1] for coord in coords]
    return min(xs), min(ys), max(xs), max(ys)


def _draw_polygon_features(
    ax: plt.Axes,
    features: list[dict[str, object]],
    *,
    facecolor: str,
    edgecolor: str,
    alpha: float,
    linewidth: float,
    zorder: int,
) -> None:
    for feature in features:
        geometry = feature["geometry"]
        for polygon in _polygon_parts(geometry):  # type: ignore[arg-type]
            exterior = polygon[0]
            projected = [_project_lonlat(float(lon), float(lat)) for lon, lat in exterior]
            xs = [point[0] for point in projected]
            ys = [point[1] for point in projected]
            if facecolor != "none":
                ax.fill(
                    xs,
                    ys,
                    facecolor=facecolor,
                    edgecolor="none",
                    alpha=alpha,
                    zorder=zorder,
                )
            ax.plot(
                xs,
                ys,
                color=edgecolor,
                alpha=max(alpha, 0.94) if facecolor == "none" else 1.0,
                linewidth=linewidth,
                zorder=zorder + 0.1,
            )


def _add_scale_bar(
    ax: plt.Axes,
    *,
    latitude: float,
    length_km: float = 100.0,
) -> list[mpl.artist.Artist]:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    start_x = x0 + 0.045 * (x1 - x0)
    start_y = y0 + 0.052 * (y1 - y0)
    projected_length = length_km * 1000.0 / math.cos(math.radians(latitude))
    end_x = start_x + projected_length

    scale_line = ax.plot(
        [start_x, end_x],
        [start_y, start_y],
        color="white",
        linewidth=4.5,
        solid_capstyle="butt",
        zorder=30,
        path_effects=[path_effects.withStroke(linewidth=6.4, foreground=TEXT, alpha=0.8)],
    )[0]
    scale_label = _outlined_text(
        ax,
        (start_x + end_x) / 2.0,
        start_y + 0.017 * (y1 - y0),
        f"{length_km:.0f} km",
        fontsize=9.8,
        fontweight="bold",
    )
    return [scale_line, scale_label]


def _add_north_arrow(ax: plt.Axes) -> plt.Axes:
    north_arrow = Image.open(NORTH_ARROW_PATH).convert("RGBA")
    arrow_ax = ax.inset_axes([0.027, 0.067, 0.058, 0.092], zorder=30)
    arrow_ax.imshow(north_arrow)
    arrow_ax.set_facecolor("none")
    arrow_ax.set_axis_off()
    return arrow_ax


def _add_hydrologic_inset(
    ax: plt.Axes,
    *,
    area_share_pct: float,
    discharge_share_pct: float,
) -> list[plt.Axes]:
    inset = ax.inset_axes([0.590, 0.040, 0.370, 0.300], zorder=40)
    inset.set_facecolor("white")
    for spine in inset.spines.values():
        spine.set_visible(False)
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_xlim(0.0, 1.0)
    inset.set_ylim(0.0, 1.0)

    inset.text(
        0.5,
        0.970,
        "Disproportionate headwater contribution",
        transform=inset.transAxes,
        color=TEXT,
        fontsize=11.2,
        fontweight="bold",
        ha="center",
        va="top",
    )

    legend_specs = (
        (0.125, HEADWATER, "Animas + Navajo"),
        (0.585, OTHER_BASINS, "Rest of San Juan basin"),
    )
    for x, color, label in legend_specs:
        inset.scatter(
            [x],
            [0.845],
            s=34,
            marker="s",
            color=color,
            edgecolor="none",
            transform=inset.transAxes,
            clip_on=False,
            zorder=5,
        )
        inset.text(
            x + 0.035,
            0.845,
            label,
            transform=inset.transAxes,
            color=TEXT,
            fontsize=8.8,
            ha="left",
            va="center",
        )

    circle_specs = (
        (
            [area_share_pct, 100.0 - area_share_pct],
            f"{area_share_pct:.1f}%",
            -0.5 * 3.6 * area_share_pct,
        ),
        (
            [discharge_share_pct, 100.0 - discharge_share_pct],
            f"{discharge_share_pct:.1f}%",
            180.0 - 0.5 * 3.6 * discharge_share_pct,
        ),
    )
    circle_axes = (
        inset.inset_axes([0.025, 0.190, 0.355, 0.580]),
        inset.inset_axes([0.620, 0.190, 0.355, 0.580]),
    )
    for circle_ax, (values, value_label, start_angle) in zip(circle_axes, circle_specs):
        circle_ax.set_facecolor("none")
        circle_ax.pie(
            values,
            colors=[HEADWATER, OTHER_BASINS],
            startangle=start_angle,
            counterclock=True,
            wedgeprops={"width": 0.31, "edgecolor": "white", "linewidth": 0.55},
        )
        circle_ax.text(
            0.5,
            0.52,
            value_label,
            transform=circle_ax.transAxes,
            color=TEXT,
            fontsize=11.5,
            fontweight="bold",
            ha="center",
            va="center",
        )
        circle_ax.set_aspect("equal")
        circle_ax.set_axis_off()

    inset.text(
        0.205,
        0.025,
        "Share of total\nSan Juan basin area",
        transform=inset.transAxes,
        color=TEXT,
        fontsize=8.5,
        ha="center",
        va="bottom",
        linespacing=1.02,
    )
    inset.text(
        0.795,
        0.025,
        "Contribution to\ndischarge at\nSJ nr Bluff",
        transform=inset.transAxes,
        color=TEXT,
        fontsize=8.5,
        ha="center",
        va="bottom",
        linespacing=1.02,
    )

    inset.annotate(
        "",
        xy=(0.590, 0.475),
        xytext=(0.410, 0.475),
        transform=inset.transAxes,
        arrowprops={
            "arrowstyle": "->",
            "color": MUTED,
            "linewidth": 1.35,
            "mutation_scale": 10,
            "shrinkA": 1.5,
            "shrinkB": 1.5,
        },
    )
    inset.text(
        0.5,
        0.545,
        "accounts for",
        transform=inset.transAxes,
        color=TEXT,
        fontsize=8.8,
        fontweight="bold",
        ha="center",
        va="bottom",
    )
    return [inset, *circle_axes]


def _write_donut_svg(
    path: Path,
    *,
    share_pct: float,
    center_angle_deg: float,
) -> None:
    """Write one PowerPoint-friendly vector donut with no embedded text."""

    share = max(0.0, min(float(share_pct), 100.0))
    sweep_angle = 3.6 * share
    start_angle = center_angle_deg - 0.5 * sweep_angle
    end_angle = center_angle_deg + 0.5 * sweep_angle

    def point(radius: float, angle_deg: float) -> tuple[float, float]:
        angle_rad = math.radians(angle_deg)
        return (
            50.0 + radius * math.cos(angle_rad),
            50.0 + radius * math.sin(angle_rad),
        )

    outer_radius = 43.0
    inner_radius = 29.0
    outer_start = point(outer_radius, start_angle)
    outer_end = point(outer_radius, end_angle)
    inner_end = point(inner_radius, end_angle)
    inner_start = point(inner_radius, start_angle)
    large_arc = 1 if sweep_angle > 180.0 else 0
    wedge_path = (
        f"M {outer_start[0]:.6f},{outer_start[1]:.6f} "
        f"A {outer_radius:.6f},{outer_radius:.6f} 0 {large_arc} 1 "
        f"{outer_end[0]:.6f},{outer_end[1]:.6f} "
        f"L {inner_end[0]:.6f},{inner_end[1]:.6f} "
        f"A {inner_radius:.6f},{inner_radius:.6f} 0 {large_arc} 0 "
        f"{inner_start[0]:.6f},{inner_start[1]:.6f} Z"
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <circle cx="50" cy="50" r="36" fill="none" stroke="{OTHER_BASINS}" stroke-width="14"/>
  <path d="{wedge_path}" fill="{HEADWATER}"/>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def _build_figure(dpi: int, *, export_editable: bool = False) -> list[Path]:
    watersheds = _load_features(WATERSHED_PATH)
    rivers = _load_features(RIVER_PATH)
    gages = _load_features(GAGE_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    basemap_meta = json.loads(BASEMAP_META_PATH.read_text(encoding="utf-8"))

    basemap = Image.open(BASEMAP_PATH).convert("RGB")
    basemap = ImageEnhance.Color(basemap).enhance(0.68)
    basemap = ImageEnhance.Contrast(basemap).enhance(0.90)
    basemap = ImageEnhance.Brightness(basemap).enhance(0.70)

    powell = [
        feature
        for feature in watersheds
        if feature["properties"]["name"] == "Powell Inflow"  # type: ignore[index]
    ]
    animas = [
        feature
        for feature in watersheds
        if feature["properties"]["name"] == "Animus River"  # type: ignore[index]
    ]
    navajo = [
        feature
        for feature in watersheds
        if feature["properties"]["name"] == "Navajo Outflow"  # type: ignore[index]
    ]

    minx, miny, maxx, maxy = _features_bounds(powell, projected=True)
    width = maxx - minx
    west_pad = 0.075 * width
    east_pad = 0.035 * width
    pad_y = 0.055 * (maxy - miny)
    basemap_left = float(basemap_meta["extent_epsg3857"][0])
    map_bounds = (
        max(minx - west_pad, basemap_left + 2500.0),
        miny - pad_y,
        maxx + east_pad,
        maxy + pad_y,
    )

    fig, ax = plt.subplots(figsize=(12.2, 7.0), constrained_layout=True)
    ax.imshow(
        np.asarray(basemap),
        extent=basemap_meta["extent_epsg3857"],
        origin="upper",
        interpolation="bilinear",
        zorder=0,
    )
    ax.set_xlim(map_bounds[0], map_bounds[2])
    ax.set_ylim(map_bounds[1], map_bounds[3])
    # Draw the full network first, with width increasing gently with drainage area.
    drainage_values = [
        max(float(feature["properties"].get("drainarea_ds_km2") or 0.0), 0.0)  # type: ignore[union-attr]
        for feature in rivers
    ]
    max_drainage = max(drainage_values) if drainage_values else 1.0
    for feature, drainage in zip(rivers, drainage_values):
        width = 0.25 + 0.72 * math.sqrt(drainage / max(max_drainage, 1.0))
        for part in _line_parts(feature["geometry"]):  # type: ignore[arg-type]
            projected = [_project_lonlat(float(lon), float(lat)) for lon, lat in part]
            ax.plot(
                [point[0] for point in projected],
                [point[1] for point in projected],
                color=BLUE_LIGHT,
                linewidth=width,
                alpha=0.46,
                zorder=3,
            )
            if bool(feature["properties"].get("san_juan")):  # type: ignore[union-attr]
                ax.plot(
                    [point[0] for point in projected],
                    [point[1] for point in projected],
                    color=BLUE,
                    linewidth=1.55,
                    alpha=0.98,
                    zorder=5,
                )

    _draw_polygon_features(
        ax,
        powell,
        facecolor="none",
        edgecolor=BOUNDARY,
        alpha=0.94,
        linewidth=1.15,
        zorder=6,
    )
    _draw_polygon_features(
        ax,
        animas,
        facecolor="none",
        edgecolor=HEADWATER,
        alpha=1.0,
        linewidth=1.9,
        zorder=7,
    )
    _draw_polygon_features(
        ax,
        navajo,
        facecolor="none",
        edgecolor=HEADWATER,
        alpha=1.0,
        linewidth=1.9,
        zorder=7,
    )

    dam_x, dam_y = _project_lonlat(NAVAJO_DAM_LONGITUDE, NAVAJO_DAM_LATITUDE)
    ax.plot(
        [dam_x],
        [dam_y],
        marker="D",
        markersize=7.5,
        markerfacecolor=RESERVOIR,
        markeredgecolor=TEXT,
        markeredgewidth=0.9,
        linestyle="none",
        zorder=11,
    )

    for feature in gages:
        properties = feature["properties"]
        lon, lat = feature["geometry"]["coordinates"]  # type: ignore[index]
        x, y = _project_lonlat(float(lon), float(lat))
        ax.plot(
            [x],
            [y],
            marker="o",
            markersize=7.0,
            markerfacecolor=GAGE,
            markeredgecolor=TEXT,
            markeredgewidth=0.9,
            linestyle="none",
            zorder=12,
        )

    map_label_artists: list[mpl.artist.Artist] = []

    # Axes-relative positions reproduce the manually refined PowerPoint layout.
    map_label_specs = (
        ("Lake\nPowell", 0.041075, 0.652000, "white", 11.2, "bold", "italic", 0.0),
        ("San Juan", 0.360111, 0.703500, "#93C5FD", 11.4, "bold", "italic", -8.0),
        ("River", 0.422519, 0.674687, "#93C5FD", 11.4, "bold", "italic", -34.1),
        (
            "Animas\nRiver\nbasin",
            0.706209,
            0.779853,
            HEADWATER,
            10.7,
            "bold",
            "normal",
            0.0,
        ),
        (
            "Navajo\nReservoir\nbasin",
            0.843376,
            0.723853,
            HEADWATER,
            10.7,
            "bold",
            "normal",
            0.0,
        ),
        (
            "Navajo\nReservoir",
            0.829589,
            0.548854,
            "white",
            10.8,
            "bold",
            "italic",
            0.0,
        ),
    )
    for label, x, y, color, fontsize, weight, style, rotation in map_label_specs:
        map_label_artists.append(
            _outlined_text(
                ax,
                x,
                y,
                label,
                color=color,
                fontsize=fontsize,
                fontweight=weight,
                fontstyle=style,
                rotation=rotation,
                transform=ax.transAxes,
            )
        )

    for gage_id, (start, end) in GAGE_LEADER_AXES.items():
        leader = ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            transform=ax.transAxes,
            color=GAGE,
            linewidth=0.95,
            zorder=18,
            path_effects=[
                path_effects.withStroke(linewidth=2.2, foreground=TEXT, alpha=0.82)
            ],
        )[0]
        map_label_artists.append(leader)

    for feature in gages:
        properties = feature["properties"]
        gage_id = str(properties["id_source"])  # type: ignore[index]
        label = GAGE_NAMES[gage_id]
        label_x, label_y = GAGE_LABEL_AXES[gage_id]
        map_label_artists.append(
            _outlined_text(
                ax,
                label_x,
                label_y,
                label,
                color=GAGE,
                fontsize=10.0,
                fontweight="medium",
                fontstyle="italic",
                transform=ax.transAxes,
                stroke_width=0.9,
                stroke_color="white",
                stroke_alpha=0.92,
            )
        )

    area_share = float(metadata["animas_plus_navajo_area_share_pct"])
    discharge_share = float(
        metadata["bluff_headwater_contribution"]["volume_share_pct"]
    )
    inset_axes = _add_hydrologic_inset(
        ax,
        area_share_pct=area_share,
        discharge_share_pct=discharge_share,
    )
    scale_bar_artists = _add_scale_bar(ax, latitude=36.75, length_km=100.0)
    north_arrow_axes = _add_north_arrow(ax)

    basin_boundary_handle = Line2D(
        [0],
        [0],
        color=BOUNDARY,
        linewidth=1.6,
        label="San Juan basin",
    )
    basin_boundary_handle.set_path_effects(
        [path_effects.withStroke(linewidth=3.0, foreground=MUTED, alpha=0.9)]
    )
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=HEADWATER,
            linewidth=1.9,
            label="Animas + Navajo basins",
        ),
        basin_boundary_handle,
        Line2D([0], [0], color=BLUE, linewidth=1.8, label="San Juan mainstem"),
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor=RESERVOIR,
            markeredgecolor=TEXT,
            markersize=7.0,
            label="Navajo Dam",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=GAGE,
            markeredgecolor=TEXT,
            markersize=7.0,
            label="USGS gage",
        ),
    ]
    legend = ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.018, 0.982),
        ncol=1,
        frameon=True,
        facecolor="white",
        edgecolor="#CBD5E1",
        framealpha=0.92,
        columnspacing=0.9,
        handlelength=1.8,
        borderpad=0.55,
    )
    for text in legend.get_texts():
        text.set_color(TEXT)

    attribution = _outlined_text(
        ax,
        map_bounds[2] - 0.015 * (map_bounds[2] - map_bounds[0]),
        map_bounds[1] + 0.012 * (map_bounds[3] - map_bounds[1]),
        MAP_ATTRIBUTION,
        fontsize=6.4,
        ha="right",
        va="bottom",
    )

    ax.set_aspect("equal")
    ax.set_axis_off()

    paths = [
        HERE / f"{OUTPUT_STEM}.png",
        HERE / f"{OUTPUT_STEM}.pdf",
        HERE / f"{OUTPUT_STEM}.svg",
    ]
    fig.savefig(paths[0], dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(paths[1], bbox_inches="tight", pad_inches=0.02)
    fig.savefig(paths[2], bbox_inches="tight", pad_inches=0.02)
    if export_editable:
        furniture_artists: list[object] = [
            legend,
            attribution,
            north_arrow_axes,
            *inset_axes,
            *scale_bar_artists,
        ]
        for artist in [*map_label_artists, *furniture_artists]:
            artist.set_visible(False)
        background_path = HERE / f"{OUTPUT_STEM}-editable-background.png"
        fig.savefig(background_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
        background_jpg_path = HERE / f"{OUTPUT_STEM}-editable-background.jpg"
        with Image.open(background_path) as background:
            background.convert("RGB").save(
                background_jpg_path,
                quality=95,
                subsampling=0,
                optimize=True,
            )
        area_donut_path = HERE / f"{OUTPUT_STEM}-editable-area-donut.svg"
        discharge_donut_path = HERE / f"{OUTPUT_STEM}-editable-discharge-donut.svg"
        _write_donut_svg(
            area_donut_path,
            share_pct=area_share,
            center_angle_deg=0.0,
        )
        _write_donut_svg(
            discharge_donut_path,
            share_pct=discharge_share,
            center_angle_deg=180.0,
        )
        paths.extend(
            [
                background_path,
                background_jpg_path,
                area_donut_path,
                discharge_donut_path,
            ]
        )
    plt.close(fig)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--prepare-data",
        action="store_true",
        help="Recreate compact GIS snapshots from --source-gis-dir.",
    )
    parser.add_argument(
        "--source-gis-dir",
        type=Path,
        help="Directory containing the source GeoPackages used with --prepare-data.",
    )
    parser.add_argument(
        "--refresh-basemap",
        action="store_true",
        help="Redownload the cached USGS imagery mosaic.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare GIS/basemap data without rendering the figure.",
    )
    parser.add_argument(
        "--export-editable",
        action="store_true",
        help="Also write a label-free PNG for the editable PowerPoint export.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _register_fonts()

    if args.prepare_data:
        if args.source_gis_dir is None:
            raise ValueError("--prepare-data requires --source-gis-dir")
        _prepare_data(args.source_gis_dir)

    missing = [path for path in REQUIRED_DATA if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing prepared figure data. Run with --prepare-data and "
            "--source-gis-dir first:\n" + "\n".join(str(path) for path in missing)
        )

    if args.refresh_basemap or not (BASEMAP_PATH.exists() and BASEMAP_META_PATH.exists()):
        watersheds = _load_features(WATERSHED_PATH)
        powell = [
            feature
            for feature in watersheds
            if feature["properties"]["name"] == "Powell Inflow"  # type: ignore[index]
        ]
        west, south, east, north = _features_bounds(powell, projected=False)
        pad_lon = 0.14
        pad_lat = 0.16
        _download_basemap(
            (west - pad_lon, south - pad_lat, east + pad_lon, north + pad_lat),
            zoom=BASEMAP_ZOOM,
        )

    if args.prepare_only:
        print(f"Prepared figure data in {DATA_DIR}")
        return

    for path in _build_figure(args.dpi, export_editable=args.export_editable):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
