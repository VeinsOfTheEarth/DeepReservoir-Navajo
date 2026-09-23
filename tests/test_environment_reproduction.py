"""Regression checks against the bundled paper evaluation."""
import numpy as np
import pandas as pd
import pytest
import torch

from deepreservoir.drl import model, selected_policy
from deepreservoir.drl.cli import _environment_options
from deepreservoir.drl.environs import CFS_TO_AF_PER_DAY


@pytest.fixture(scope="module")
def environment_inputs():
    data = model.load_all_model_data()
    config = selected_policy.load_selected_policy_config()
    window = config["base_eval"]["windows"][0]
    raw = data["raw"].loc[window["start"]:window["end"]].copy()
    return dict(data_raw=raw, data_norm=data["norm"].loc[raw.index],
                norm_stats=data["norm_stats"],
                reward_spec_str=selected_policy.SELECTED_REWARD_SPEC,
                **_environment_options(config))


@pytest.mark.xfail(
    strict=True,
    reason="Bundled Phase-95 rollout uses the archived tailwater curve and eta; "
           "current code has the Plate 7-4 correction pending policy retraining",
)
def test_selected_policy_reproduces_bundled_daily_results(environment_inputs):
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    env = model.make_env(**environment_inputs, is_eval=True, episode_length=None)
    try:
        agent = selected_policy.load_selected_policy_model(device="cpu")
        actual = model._run_rollout_env(agent=agent, eval_env=env)
    finally:
        env.close()
        torch.set_num_threads(threads)
    expected = pd.read_parquet(selected_policy.SELECTED_EVAL_ROLLOUT_PATH)
    assert actual.index.equals(expected.index)
    # The archived rollout includes zero-filled fields for an unused fifth action.
    retired_columns = {
        "raw_action_hydropower_sj_frac", "scaled_action_hydropower_sj_frac",
        "hydropower_sj_action_mode", "hydropower_sj_request_cfs",
    }
    assert (expected[list(retired_columns)] == 0).all().all()
    columns = [c for c in expected.select_dtypes("number") if c not in retired_columns]
    assert set(columns) <= set(actual.columns)
    np.testing.assert_allclose(actual[columns], expected[columns], rtol=1e-7, atol=1e-6,
                               equal_nan=True)
    np.testing.assert_allclose(
        actual["sj_at_archuleta_proxy_cfs"],
        actual["sj_main_flow_cfs"],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        actual["sj_at_bluff_proxy_cfs"],
        actual["sj_at_farmington_lag2_cfs"],
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )


@pytest.mark.parametrize("initial_storage", [0., 300000., 1000000., 1700000.])
def test_action_shapes_and_mass_balance(environment_inputs, initial_storage):
    env = model.make_env(**environment_inputs, is_eval=True, episode_length=None)
    try:
        obs, _ = env.reset(seed=4, options={"initial_storage_af": initial_storage})
        assert obs.shape == (20,)
        with pytest.raises(ValueError, match="action shape"):
            env.step(np.zeros(2))
        before = env.storage_af
        _, reward, _, _, info = env.step(np.ones(4, dtype=np.float32))
        assert np.isfinite(reward)
        row = environment_inputs["data_raw"].iloc[0]
        available = max(before + row.inflow_cfs * CFS_TO_AF_PER_DAY - row.evap_af, 0.)
        released = (info["release_sj_main_cfs"] + info["release_niip_cfs"]) * CFS_TO_AF_PER_DAY
        np.testing.assert_allclose(env.storage_af + released + info["spill_af"], available,
                                   rtol=1e-10, atol=1e-8)
        assert 0 <= env.storage_af <= env.max_storage_af
    finally:
        env.close()
