from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from deepreservoir.drl import rewards


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = (
    REPO_ROOT
    / "config_files"
    / "reward_jon_phase95_corrected_recovery.json"
)


def _context(**info: float) -> rewards.RewardContext:
    return rewards.RewardContext(
        t=0,
        date=pd.Timestamp("2014-06-01"),
        obs=np.zeros(1, dtype=float),
        action=np.zeros(4, dtype=float),
        next_obs=np.zeros(1, dtype=float),
        info=dict(info),
    )


def test_every_corrected_phase95_reward_spec_resolves_through_registry() -> None:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))

    assert len(spec["experiments"]) == 8
    for experiment in spec["experiments"]:
        parsed = rewards.parse_objective_spec(experiment["reward_spec"])
        composite = rewards.build_composite_reward(parsed)

        assert len(composite.components) == 7
        assert {component.objective for component in composite.components} == {
            "dam_safety",
            "storage_control",
            "hydropower",
            "flooding",
            "niip",
            "esa_min_flow",
            "esa_spring_peak_release",
        }


def test_restored_phase95_storage_variants_match_their_definitions() -> None:
    plateau = rewards.storage_control_plateau90to96_concave0_softupper98
    assert plateau(_context(storage_af=900.0, max_storage_af=1_000.0)) == pytest.approx(1.0)
    assert plateau(_context(storage_af=960.0, max_storage_af=1_000.0)) == pytest.approx(1.0)
    assert plateau(_context(storage_af=970.0, max_storage_af=1_000.0)) == pytest.approx(0.75)
    assert plateau(_context(storage_af=1_000.0, max_storage_af=1_000.0)) == pytest.approx(-1.0)

    oi_shift = rewards.storage_control_target_oishift875to90_concave0_softupper98
    wet = oi_shift(
        _context(storage_af=875.0, max_storage_af=1_000.0, spring_oi=1.0)
    )
    scarce = oi_shift(
        _context(storage_af=875.0, max_storage_af=1_000.0, spring_oi=0.0)
    )
    assert wet == pytest.approx(1.0)
    assert scarce == pytest.approx(-1.0 + 2.0 * np.sqrt(875.0 / 900.0))


def test_restored_phase95_hydropower_variants_match_their_definitions() -> None:
    assert rewards.hydropower_positive_soft(
        _context(hydropower_mwh=384.0)
    ) == pytest.approx(0.5)
    assert rewards.hydropower_positive_efficiency(
        _context(hydropower_mwh=768.0, release_sj_main_cfs=1_300.0)
    ) == pytest.approx(1.0)
    assert rewards.hydropower_positive_efficiency(
        _context(hydropower_mwh=768.0, release_sj_main_cfs=2_600.0)
    ) == pytest.approx(0.5)
