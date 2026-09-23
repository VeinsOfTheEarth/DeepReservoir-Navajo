import numpy as np
import pytest
from deepreservoir.drl.environs import (
    compute_esa_required_release_frac_signal, compute_spr_threshold_needed_frac_signal,
    normalize_action_scaling, obs_columns_for_context, spr_proxy_action_to_target,
)
from deepreservoir.drl.observations import SELECTED_OBS_COLUMNS


def test_only_checkpoint_observations_are_supported():
    assert obs_columns_for_context(None) == SELECTED_OBS_COLUMNS
    assert len(SELECTED_OBS_COLUMNS) == 20
    assert not any("fcst" in c or "swe" in c or "spring_oi" in c for c in SELECTED_OBS_COLUMNS)
    with pytest.raises(ValueError):
        obs_columns_for_context("base")
    with pytest.raises(ValueError):
        normalize_action_scaling("sj_power2")


def test_esa_input_is_current_release_requirement():
    result = compute_esa_required_release_frac_signal(
        animas_cfs=np.array([0., 250., 500., 1000.]), max_release_sj_main_cfs=5000.)
    np.testing.assert_allclose(result, [0.1, 0.05, 0., 0.])
    assert result.dtype == np.float32


def test_spr_input_respects_calendar_and_capacity():
    result = compute_spr_threshold_needed_frac_signal(
        animas_cfs=np.array([1000., 4000., 9000., 11000.]),
        spring_window_active=np.array([False, True, True, True]),
        max_release_sj_main_cfs=5000., threshold_cfs=10000.)
    np.testing.assert_allclose(result, [0., 1., 0.2, 0.])


@pytest.mark.parametrize("action,expected", [
    (-1., (0, 0.)), (-0.59, (1, 2500.)), (0., (2, 5000.)),
    (0.21, (3, 8000.)), (0.61, (4, 10000.)), (1., (4, 10000.)),
])
def test_spr_action_quantization(action, expected):
    assert spr_proxy_action_to_target(action) == expected
