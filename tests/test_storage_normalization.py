"""Training-only storage scaling and frozen evaluation metadata."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from deepreservoir.drl import model


def _normalization_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    index = pd.date_range("2000-01-01", periods=5, freq="D")
    all_raw = pd.DataFrame(
        {
            "storage_af": [10.0, 20.0, 30.0, 80.0, 110.0],
            "inflow_cfs": np.ones(5),
            "evap_af": np.ones(5),
        },
        index=index,
    )
    train_raw = all_raw.iloc[:3].copy()
    full_stats = pd.DataFrame(
        {
            "mean": [50.0, 1.0, 1.0],
            "std": [40.0, 2.0, 3.0],
        },
        index=["storage_af", "inflow_cfs", "evap_af"],
    )
    return all_raw, train_raw, full_stats


def test_mode_validation_preserves_legacy_default() -> None:
    assert model.normalize_storage_normalization(None) == "full_record"
    assert model.normalize_storage_normalization("train_window") == "train_window"
    with pytest.raises(ValueError, match="storage_normalization"):
        model.normalize_storage_normalization("evaluation_window")


def test_train_window_mode_fits_only_training_storage() -> None:
    all_raw, train_raw, full_stats = _normalization_inputs()

    resolved, metadata = model._storage_normalization_for_training(
        train_raw=train_raw,
        all_raw=all_raw,
        full_record_norm_stats=full_stats,
        mode="train_window",
    )

    assert resolved.loc["storage_af", "mean"] == pytest.approx(20.0)
    assert resolved.loc["storage_af", "std"] == pytest.approx(10.0)
    assert resolved.loc["inflow_cfs", "std"] == full_stats.loc["inflow_cfs", "std"]
    assert full_stats.loc["storage_af", "mean"] == 50.0
    assert metadata == {
        "schema_version": 1,
        "mode": "train_window",
        "column": "storage_af",
        "source": "training_window",
        "mean": 20.0,
        "std": 10.0,
        "ddof": 1,
        "n_observations": 3,
        "fit_window": {"start": "2000-01-01", "end": "2000-01-03"},
    }


def test_full_record_mode_keeps_loader_statistics_exactly() -> None:
    all_raw, train_raw, full_stats = _normalization_inputs()

    resolved, metadata = model._storage_normalization_for_training(
        train_raw=train_raw,
        all_raw=all_raw,
        full_record_norm_stats=full_stats,
        mode="full_record",
    )

    pd.testing.assert_frame_equal(resolved, full_stats)
    assert metadata["source"] == "loader_full_record"
    assert metadata["mean"] == 50.0
    assert metadata["std"] == 40.0
    assert metadata["fit_window"] == {
        "start": "2000-01-01",
        "end": "2000-01-05",
    }


def test_evaluation_loads_frozen_training_stats_and_refuses_mismatch(tmp_path) -> None:
    all_raw, train_raw, full_stats = _normalization_inputs()
    _, metadata = model._storage_normalization_for_training(
        train_raw=train_raw,
        all_raw=all_raw,
        full_record_norm_stats=full_stats,
        mode="train_window",
    )
    artifact = tmp_path / model.STORAGE_NORMALIZATION_FILENAME
    model._write_storage_normalization_artifact(artifact, metadata)

    resolved, loaded = model._storage_norm_stats_for_evaluation(
        model_path=tmp_path / "last_model.zip",
        full_record_norm_stats=full_stats.assign(mean=[999.0, 1.0, 1.0]),
        mode="train_window",
    )

    assert loaded == metadata
    assert resolved.loc["storage_af", "mean"] == 20.0
    assert resolved.loc["storage_af", "std"] == 10.0
    with pytest.raises(ValueError, match="does not match"):
        model._storage_norm_stats_for_evaluation(
            model_path=tmp_path / "last_model.zip",
            full_record_norm_stats=full_stats,
            mode="full_record",
        )
    with pytest.raises(ValueError, match="Refusing to replace frozen"):
        model._write_storage_normalization_artifact(
            artifact,
            {**metadata, "mean": 21.0},
        )


def test_legacy_artifact_without_sidecar_uses_full_record_stats(tmp_path) -> None:
    _, _, full_stats = _normalization_inputs()

    resolved, metadata = model._storage_norm_stats_for_evaluation(
        model_path=tmp_path / "legacy_model.zip",
        full_record_norm_stats=full_stats,
        mode="full_record",
    )

    pd.testing.assert_frame_equal(resolved, full_stats)
    assert metadata is None
    with pytest.raises(FileNotFoundError, match="frozen artifact"):
        model._storage_norm_stats_for_evaluation(
            model_path=tmp_path / "corrected_model.zip",
            full_record_norm_stats=full_stats,
            mode="train_window",
        )


@pytest.mark.parametrize("schema_version", [None, 2, "1", True])
def test_evaluation_rejects_missing_or_unsupported_sidecar_schema(
    tmp_path,
    schema_version,
) -> None:
    all_raw, train_raw, full_stats = _normalization_inputs()
    _, metadata = model._storage_normalization_for_training(
        train_raw=train_raw,
        all_raw=all_raw,
        full_record_norm_stats=full_stats,
        mode="train_window",
    )
    if schema_version is None:
        metadata.pop("schema_version")
    else:
        metadata["schema_version"] = schema_version
    model._write_json(tmp_path / model.STORAGE_NORMALIZATION_FILENAME, metadata)

    with pytest.raises(ValueError, match="schema_version"):
        model._storage_norm_stats_for_evaluation(
            model_path=tmp_path / "last_model.zip",
            full_record_norm_stats=full_stats,
            mode="train_window",
        )


def test_run_rollout_data_validates_supplied_stats_and_mode_before_loading(
    tmp_path,
    monkeypatch,
) -> None:
    all_raw, train_raw, full_stats = _normalization_inputs()
    resolved, metadata = model._storage_normalization_for_training(
        train_raw=train_raw,
        all_raw=all_raw,
        full_record_norm_stats=full_stats,
        mode="train_window",
    )
    model._write_storage_normalization_artifact(
        tmp_path / model.STORAGE_NORMALIZATION_FILENAME,
        metadata,
    )
    monkeypatch.setattr(
        model,
        "make_env",
        lambda **_: pytest.fail("make_env ran before normalization validation"),
    )
    monkeypatch.setattr(
        model,
        "PPO",
        SimpleNamespace(
            load=lambda *_, **__: pytest.fail(
                "PPO.load ran before normalization validation"
            )
        ),
    )

    kwargs = {
        "model_path": tmp_path / "last_model.zip",
        "reward_spec": "dam_safety:spill_guard_warn98@1.0",
        "data_raw": train_raw,
        "data_norm": train_raw.copy(),
        "norm_stats": resolved.copy(),
        "storage_normalization": "train_window",
    }
    kwargs["norm_stats"].loc["storage_af", "mean"] = 21.0
    with pytest.raises(ValueError, match="do not match"):
        model.run_rollout_data(**kwargs)

    kwargs["norm_stats"] = resolved
    kwargs["storage_normalization"] = "full_record"
    with pytest.raises(ValueError, match="mode does not match"):
        model.run_rollout_data(**kwargs)


@pytest.mark.parametrize(
    ("mode", "write_sidecar"),
    [("train_window", True), ("full_record", False)],
)
def test_run_rollout_data_accepts_matching_and_legacy_normalization(
    tmp_path,
    monkeypatch,
    mode,
    write_sidecar,
) -> None:
    all_raw, train_raw, full_stats = _normalization_inputs()
    resolved, metadata = model._storage_normalization_for_training(
        train_raw=train_raw,
        all_raw=all_raw,
        full_record_norm_stats=full_stats,
        mode=mode,
    )
    if write_sidecar:
        model._write_storage_normalization_artifact(
            tmp_path / model.STORAGE_NORMALIZATION_FILENAME,
            metadata,
        )
    eval_env = object()
    agent = object()
    expected = pd.DataFrame({"release_cfs": [1.0]}, index=train_raw.index[:1])
    monkeypatch.setattr(model, "make_env", lambda **_: eval_env)
    monkeypatch.setattr(model, "PPO", SimpleNamespace(load=lambda *_, **__: agent))
    monkeypatch.setattr(
        model,
        "_run_rollout_env",
        lambda *, agent, eval_env, reset_options: expected,
    )

    actual = model.run_rollout_data(
        model_path=tmp_path / "last_model.zip",
        reward_spec="dam_safety:spill_guard_warn98@1.0",
        data_raw=train_raw,
        data_norm=train_raw.copy(),
        norm_stats=resolved,
        storage_normalization=mode,
    )

    assert actual is expected


def test_load_model_validates_source_sidecar_before_ppo_load(
    tmp_path,
    monkeypatch,
) -> None:
    all_raw, train_raw, full_stats = _normalization_inputs()
    resolved, metadata = model._storage_normalization_for_training(
        train_raw=train_raw,
        all_raw=all_raw,
        full_record_norm_stats=full_stats,
        mode="train_window",
    )
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    checkpoint = source_dir / "last_model.zip"
    checkpoint.write_bytes(b"fake")
    model._write_storage_normalization_artifact(
        source_dir / model.STORAGE_NORMALIZATION_FILENAME,
        {**metadata, "mean": 21.0},
    )
    load_calls: list[object] = []
    monkeypatch.setattr(
        model,
        "PPO",
        SimpleNamespace(load=lambda *args, **kwargs: load_calls.append((args, kwargs))),
    )
    run = object.__new__(model.DRLModel)
    run.logdir = tmp_path / "different_run"
    run.device = "cpu"
    run.train_env = object()
    run.norm_stats = resolved
    run.storage_normalization = "train_window"

    with pytest.raises(ValueError, match="do not match"):
        run.load_model(str(checkpoint))
    assert load_calls == []


def test_load_model_accepts_matching_source_sidecar(tmp_path, monkeypatch) -> None:
    all_raw, train_raw, full_stats = _normalization_inputs()
    resolved, metadata = model._storage_normalization_for_training(
        train_raw=train_raw,
        all_raw=all_raw,
        full_record_norm_stats=full_stats,
        mode="train_window",
    )
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    checkpoint = source_dir / "last_model.zip"
    checkpoint.write_bytes(b"fake")
    model._write_storage_normalization_artifact(
        source_dir / model.STORAGE_NORMALIZATION_FILENAME,
        metadata,
    )
    agent = object()
    calls: list[tuple[object, object, object]] = []

    def fake_load(path, *, env, device):
        calls.append((path, env, device))
        return agent

    monkeypatch.setattr(model, "PPO", SimpleNamespace(load=fake_load))
    run = object.__new__(model.DRLModel)
    run.logdir = tmp_path / "different_run"
    run.device = "cpu"
    run.train_env = object()
    run.norm_stats = resolved
    run.storage_normalization = "train_window"

    assert run.load_model(str(checkpoint)) is agent
    assert calls == [(checkpoint, run.train_env, "cpu")]


def test_load_model_preserves_sidecar_free_full_record_legacy_mode(
    tmp_path,
    monkeypatch,
) -> None:
    _, _, full_stats = _normalization_inputs()
    checkpoint = tmp_path / "legacy_model.zip"
    checkpoint.write_bytes(b"fake")
    agent = object()
    load_calls: list[Path] = []

    def fake_load(path, **_):
        load_calls.append(path)
        return agent

    monkeypatch.setattr(model, "PPO", SimpleNamespace(load=fake_load))
    run = object.__new__(model.DRLModel)
    run.logdir = tmp_path
    run.device = "cpu"
    run.train_env = object()
    run.norm_stats = full_stats
    run.storage_normalization = "full_record"

    assert run.load_model(str(checkpoint)) is agent
    assert load_calls == [checkpoint]

    run.storage_normalization = "train_window"
    with pytest.raises(FileNotFoundError, match="frozen artifact"):
        run.load_model(str(checkpoint))
    assert load_calls == [checkpoint]


class _FakeAgent:
    num_timesteps = 2

    def learn(self, **_: object) -> None:
        return None

    def save(self, path: str) -> None:
        Path(path).write_bytes(b"fake model")


def test_drlmodel_records_storage_normalization_in_run_artifacts(
    tmp_path,
    monkeypatch,
) -> None:
    all_raw, train_raw, full_stats = _normalization_inputs()
    all_norm = all_raw.copy()
    all_norm.loc[:, :] = 0.0
    loaded = {
        "raw": all_raw,
        "norm": all_norm,
        "norm_stats": full_stats,
        "storage_datum_meta": {
            "mode": "reported",
            "storage_column": "storage_af",
            "source": "reported reservoir storage",
        },
    }
    monkeypatch.setattr(model, "load_all_model_data", lambda **_: loaded)
    monkeypatch.setattr(model, "make_env", lambda **_: object())
    monkeypatch.setattr(model, "Monitor", lambda env: env)
    monkeypatch.setattr(model, "build_agent", lambda *_, **__: _FakeAgent())
    monkeypatch.setattr(
        model,
        "_apply_resolved_agent_config_to_manifest",
        lambda manifest, **_: manifest,
    )

    run = model.DRLModel(
        reward_spec="dam_safety:spill_guard_warn98@1.0",
        train_start=str(train_raw.index.min().date()),
        train_end=str(train_raw.index.max().date()),
        storage_normalization="train_window",
        episode_length_train=2,
        logdir=tmp_path,
    )
    run.train(total_timesteps=2, n_episodes=1, track_reward_components=False)

    artifact_path = tmp_path / model.STORAGE_NORMALIZATION_FILENAME
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert artifact["mean"] == 20.0
    assert artifact["std"] == 10.0
    assert manifest["config"]["storage_normalization"] == "train_window"
    assert manifest["config"]["storage_normalization_meta"] == artifact
    assert manifest["train_invocations"][-1]["requested_train_args"][
        "storage_normalization"
    ] == "train_window"
    assert manifest["artifacts"]["observation_normalization"] == str(
        artifact_path.resolve()
    )
