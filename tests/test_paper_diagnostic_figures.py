from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUPPORT = ROOT / "paper" / "figure-support"
if str(SUPPORT) not in sys.path:
    sys.path.insert(0, str(SUPPORT))

import paperstyle  # noqa: E402


def _load_search_module():
    path = ROOT / "paper" / "figures" / "experiment-search" / "search.py"
    spec = importlib.util.spec_from_file_location("test_experiment_search", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_spr_success_markers_use_defined_may_june_window() -> None:
    index = pd.date_range("2020-03-01", "2020-07-31", freq="D")
    proxy = pd.Series(0.0, index=index)
    proxy.loc["2020-04-01":"2020-04-10"] = 2_500.0

    markers = paperstyle._spr_threshold_year_markers(
        index,
        proxy,
        threshold_specs=((2_500.0, 10, 0.80),),
    )
    assert markers == []

    proxy.loc["2020-05-09":"2020-05-18"] = 2_500.0
    markers = paperstyle._spr_threshold_year_markers(
        index,
        proxy,
        threshold_specs=((2_500.0, 10, 0.80),),
    )
    assert markers == [(pd.Timestamp("2020-05-22"), 2_500.0, 2_500)]


def test_archive_selection_keeps_only_comparable_policy_records() -> None:
    search = _load_search_module()
    raw_path = search.DATA_DIR / "archived-search-metrics.csv"
    raw = pd.read_csv(raw_path, float_precision="round_trip", low_memory=False)
    selected = search._select_comparable_archive_records(raw)

    assert len(raw) == 3_320
    assert len(selected) == 2_740
    assert set(selected.loc[selected["phase"].eq(78), "window"]) == {"holdout_2014_2024_08_17"}
    assert not selected.loc[selected["phase"].eq(79), "stage"].eq("historical").any()

    required = (
        selected.loc[:, search.ARCHIVE_REQUIRED_METRICS]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )
    assert np.isfinite(required).all()
    spill = pd.to_numeric(selected["total_spill_af"], errors="coerce")
    assert spill.notna().all()
    assert np.isinf(spill).sum() == 196


def test_niip_reflex_pairs_use_a_common_comparison_window() -> None:
    path = (
        ROOT
        / "paper"
        / "figures"
        / "niip-reflex"
        / "data"
        / "phase52-split-paired-reflex-reduction.csv"
    )
    pairs = pd.read_csv(path)
    complete = pairs.dropna(subset=["shared_reflex_abs_cfs", "split_reflex_abs_cfs"])

    assert len(pairs) == 18
    assert len(complete) == 17
    assert (complete["split_reflex_abs_cfs"] < complete["shared_reflex_abs_cfs"]).sum() == 13
    assert np.isclose(complete["shared_reflex_abs_cfs"].median(), 226.882068, atol=1e-6)
    assert np.isclose(complete["split_reflex_abs_cfs"].median(), 45.987385, atol=1e-6)

    undefined = pairs[pairs["shared_reflex_abs_cfs"].isna()]
    assert len(undefined) == 1
    assert int(undefined.iloc[0]["shared_nonrequest_active_days"]) == 0
