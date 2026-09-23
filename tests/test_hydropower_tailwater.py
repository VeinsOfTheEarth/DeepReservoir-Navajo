"""Plate 7-4 tailwater digitization and scalar/vector model consistency."""

import numpy as np
import pytest

from deepreservoir.define_env.hydropower_model import (
    _TAILWATER_ELEV_FT,
    _TAILWATER_MODEL,
    _TAILWATER_Q_CFS,
    _tailwater_ft_scalar,
    navajo_power_generation_model,
    navajo_power_generation_scalar,
)


def test_digitized_plate_knots_and_low_flow_bend():
    assert _TAILWATER_Q_CFS.tolist() == [
        0, 600, 2000, 4000, 8000, 16000, 24000, 32000, 40000
    ]
    assert _TAILWATER_ELEV_FT.tolist() == [
        5711.7, 5713.0, 5713.3, 5713.7, 5714.4, 5716.1, 5717.8, 5719.2, 5720.4
    ]
    assert _tailwater_ft_scalar(1300) == pytest.approx(5713.15)


@pytest.mark.parametrize("release", [-100, 0, 224, 600, 1300, 5420, 40000, 45000])
def test_scalar_and_vector_tailwater_agree(release):
    assert float(_TAILWATER_MODEL([release])[0]) == pytest.approx(
        _tailwater_ft_scalar(release)
    )


@pytest.mark.parametrize("release", [0, 224, 600, 1300, 5420, 40000, 45000])
def test_scalar_and_vector_energy_agree(release):
    eta = 0.66
    elevation = 6050.0
    scalar = navajo_power_generation_scalar(release, elevation, eta_eff=eta)
    vector = navajo_power_generation_model(
        np.array([release, release]), np.array([elevation, elevation]), eta_eff=eta
    )
    assert vector == pytest.approx([scalar, scalar])
