"""Common-datum preprocessing for the Navajo Reservoir storage record."""

from __future__ import annotations

import pandas as pd
import pytest

from deepreservoir.data.loader import NavajoData
from deepreservoir.data.storage_datum import (
    apply_storage_datum_mode,
    normalize_storage_datum_mode,
)
from deepreservoir.drl import model


def test_mode_validation_preserves_reported_default() -> None:
    assert normalize_storage_datum_mode(None) == "reported"
    assert normalize_storage_datum_mode("legacy") == "reported"
    assert normalize_storage_datum_mode("2019") == "elevation_2019"
    with pytest.raises(ValueError, match="storage_datum_mode"):
        normalize_storage_datum_mode("mixed_tables")


def test_2019_mode_removes_export_table_break_and_preserves_source_values() -> None:
    index = pd.to_datetime(["2021-09-30", "2021-10-01"])
    reported = pd.DataFrame(
        {
            "elev_ft": [6024.10, 6024.05],
            "storage_af": [950_561.0, 903_061.0],
        },
        index=index,
    )

    legacy, legacy_meta = apply_storage_datum_mode(reported)
    pd.testing.assert_frame_equal(legacy, reported)
    assert legacy_meta["mode"] == "reported"
    assert "storage_reported_af" not in legacy.columns

    corrected, corrected_meta = apply_storage_datum_mode(
        reported,
        mode="elevation_2019",
    )
    assert corrected["storage_reported_af"].tolist() == [950_561.0, 903_061.0]
    assert corrected["storage_af"].tolist() == pytest.approx(
        [903_523.27, 903_060.07],
        abs=0.01,
    )
    assert corrected["storage_af"].diff().iloc[-1] == pytest.approx(-463.20)
    assert corrected_meta["mode"] == "elevation_2019"
    assert corrected_meta["n_converted"] == 2


def test_area_capacity_table_metadata_matches_bundled_csv() -> None:
    table = NavajoData().load_table("elev_area_storage_data")
    assert list(table.columns) == ["elev_ft", "area_ac", "capacity_af"]
    assert len(table) == 33_200
    assert table.iloc[0].to_dict() == {
        "elev_ft": 5775.0,
        "area_ac": 491.58,
        "capacity_af": 0.0,
    }
    assert table.iloc[-1]["elev_ft"] == pytest.approx(6106.99)


def test_full_record_common_datum_is_used_before_normalization() -> None:
    data = model.load_all_model_data(storage_datum_mode="elevation_2019")
    dates = pd.to_datetime(["2021-09-30", "2021-10-01"])
    rows = data["raw"].loc[
        dates,
        ["elev_ft", "storage_reported_af", "storage_af"],
    ]

    assert rows["storage_reported_af"].tolist() == [950_561.0, 903_061.0]
    assert rows["storage_af"].tolist() == pytest.approx(
        [903_523.27, 903_060.07],
        abs=0.01,
    )
    assert data["storage_datum_meta"]["n_converted"] == 20_892
    assert data["norm"]["storage_af"].mean() == pytest.approx(0.0, abs=1e-12)
    assert data["norm"]["storage_af"].std() == pytest.approx(1.0, abs=1e-12)


def test_legacy_full_record_remains_as_reported() -> None:
    data = model.load_all_model_data()
    rows = data["raw"].loc[
        pd.to_datetime(["2021-09-30", "2021-10-01"]),
        "storage_af",
    ]

    assert rows.tolist() == [950_561.0, 903_061.0]
    assert "storage_reported_af" not in data["raw"].columns
    assert data["storage_datum_meta"]["mode"] == "reported"
