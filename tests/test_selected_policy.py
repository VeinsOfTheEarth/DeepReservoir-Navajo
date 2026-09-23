from __future__ import annotations

import hashlib

import pytest

from deepreservoir.drl import selected_policy
from deepreservoir.drl.environs import (
    OBS_CONTEXT_SPECS,
    normalize_action_mode,
    normalize_spr_advice_mode,
)
from deepreservoir.drl.model import (
    normalize_policy_type,
    normalize_train_hydrology_transform,
)
from deepreservoir.drl.rewards import build_composite_reward, parse_objective_spec


def _sha256(path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def test_selected_policy_config_matches_metadata() -> None:
    cfg = selected_policy.load_selected_policy_config()
    exp = selected_policy.selected_experiment(cfg)
    train = cfg["base_train"]

    assert cfg["submission_name"] == selected_policy.SELECTED_PUBLIC_NAME
    assert cfg["seeds"] == [selected_policy.SELECTED_SEED]
    assert exp["source_family"] == selected_policy.SELECTED_SOURCE_FAMILY
    assert exp["reward_spec"] == selected_policy.SELECTED_REWARD_SPEC
    assert train["policy_type"] == selected_policy.SELECTED_POLICY_TYPE
    assert train["action_mode"] == selected_policy.SELECTED_ACTION_MODE
    assert train["obs_context"] == selected_policy.SELECTED_OBS_CONTEXT
    assert train["spr_advice_mode"] == selected_policy.SELECTED_SPR_ADVICE_MODE
    assert (
        train["train_hydrology_transform"]
        == selected_policy.SELECTED_TRAIN_HYDROLOGY_TRANSFORM
    )


def test_selected_public_config_uses_registered_modes() -> None:
    cfg = selected_policy.load_selected_policy_config()
    train = cfg["base_train"]

    assert (
        normalize_policy_type(train["policy_type"])
        == selected_policy.SELECTED_POLICY_TYPE
    )
    assert (
        normalize_action_mode(train["action_mode"])
        == selected_policy.SELECTED_ACTION_MODE
    )
    assert (
        normalize_spr_advice_mode(train["spr_advice_mode"])
        == selected_policy.SELECTED_SPR_ADVICE_MODE
    )
    assert (
        normalize_train_hydrology_transform(train["train_hydrology_transform"])
        == selected_policy.SELECTED_TRAIN_HYDROLOGY_TRANSFORM
    )
    assert train["obs_context"] in OBS_CONTEXT_SPECS
    assert (
        tuple(OBS_CONTEXT_SPECS[train["obs_context"]])
        == selected_policy.SELECTED_OBSERVATION_COLUMNS
    )


def test_selected_reward_spec_is_registered_and_buildable() -> None:
    parsed = parse_objective_spec(selected_policy.SELECTED_REWARD_SPEC)
    reward = build_composite_reward(parsed)

    assert len(reward.components) == 7
    assert [(c.objective, c.variant, c.alpha) for c in reward.components] == [
        ("dam_safety", "spill_guard_warn98", 1.0),
        ("storage_control", "target_oishift875to90_concave0_softupper98", 2.5),
        ("hydropower", "positive_efficiency", 1.5),
        ("flooding", "penalty_caps_archuleta_bluff", 0.25),
        ("niip", "delivery_match_historic_hardmeet_b10oversoft_neg1_logoff025_jon", 3.0),
        ("esa_min_flow", "green_logistic_jon", 2.5),
        (
            "esa_spring_peak_release",
            "farmington_thresholds_actionproxy_smart_ledger_hammer_histfreq_stop_excess_strong_calendar",
            2.0,
        ),
    ]


def test_selected_policy_model_artifact_is_loadable() -> None:
    model_path = selected_policy.selected_model_path()

    assert model_path.exists()
    assert model_path.name == "navajo_reservoir_selected_policy.zip"

    loaded = selected_policy.load_selected_policy_model(device="cpu")
    assert loaded.__class__.__name__ == "PPO"


def test_selected_policy_eval_artifacts_match_manifest() -> None:
    paths = (
        selected_policy.SELECTED_EVAL_METRICS_CSV_PATH,
        selected_policy.SELECTED_EVAL_METRICS_JSON_PATH,
        selected_policy.SELECTED_EVAL_ROLLOUT_PATH,
    )
    for path in paths:
        assert path.exists()
        assert _sha256(path) == selected_policy.SELECTED_ARTIFACT_SHA256[path.name]


def test_selected_policy_split_head_masks_match_expected_dimensions() -> None:
    loaded = selected_policy.load_selected_policy_model(device="cpu")
    extractor = loaded.policy.mlp_extractor

    assert loaded.observation_space.shape == (20,)
    assert loaded.action_space.shape == (4,)
    assert int(extractor.sj_branch.indices.numel()) == 6
    assert int(extractor.niip_branch.indices.numel()) == 5
    assert int(extractor.spr_branch.indices.numel()) == 20
    assert int(extractor.discretionary_sj_branch.indices.numel()) == 6
    assert int(extractor.vf_branch.indices.numel()) == 20


def test_public_spr_advice_default_is_operational() -> None:
    assert normalize_spr_advice_mode(None) == selected_policy.SELECTED_SPR_ADVICE_MODE
    assert normalize_spr_advice_mode("") == selected_policy.SELECTED_SPR_ADVICE_MODE


def test_hindsight_spr_modes_are_not_public_inputs() -> None:
    with pytest.raises(ValueError):
        normalize_spr_advice_mode("oracle_future")
    with pytest.raises(ValueError):
        normalize_spr_advice_mode("perfect_future")
    with pytest.raises(ValueError):
        normalize_action_mode("spr_target_proxy_3d_mask_perfect_lower")
    with pytest.raises(ValueError):
        normalize_action_mode("esa_base_spr_proxy_4d_max_mask_perfect_lower")
