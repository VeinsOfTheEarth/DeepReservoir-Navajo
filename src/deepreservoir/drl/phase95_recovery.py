"""Plan, run, and summarize the corrected Phase 95 recovery search.

This runner performs the eight-family, sixteen-seed search with the approved
model corrections, records the full execution contract, and evaluates every
policy against the corrected historic benchmark.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shlex
import socket
import subprocess
import sys
import tempfile
import traceback
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from deepreservoir.drl import metrics as drl_metrics
from deepreservoir.drl import model
from deepreservoir.drl import rewards as drl_rewards


REPO_ROOT = Path(__file__).resolve().parents[3]
CFS_DAY_TO_AF = 86400.0 / 43560.0
OBJECTIVE_SCREEN_PROFILE = "corrected_phase95_historic_screen_v1"
OBJECTIVE_SCREEN_TOLERANCE = 1e-12
OBJECTIVE_METRICS = (
    "total_spill_af",
    "storage_frac_of_max_possible",
    "esa_min_flow_frac_days_met",
    "flooding_frac_days_met",
    "spr_freq_years_meeting_10000cfs_5d",
    "spr_freq_years_meeting_8000cfs_10d",
    "spr_freq_years_meeting_5000cfs_21d",
    "spr_freq_years_meeting_2500cfs_10d",
    "hydropower_frac_of_max_possible",
    "niip_annual_volume_frac_of_contract",
)

ENVIRONMENT_KEYS = (
    "obs_context",
    "action_scaling",
    "action_mode",
    "reward_balancing",
    "max_release_sj_main_cfs",
    "max_release_niip_cfs",
    "esa_min_flow_floor",
    "esa_baseflow_max_multiplier",
    "spr_proxy_priority_release",
    "spr_proxy_owns_sj_window",
    "spr_advice_mode",
    "decision_hydrology_timing",
    "niip_fallback_mode",
    "storage_datum_mode",
    "storage_normalization",
    "storage_budget_target_frac_of_max",
    "mask_incomplete_initial_spr",
)

REQUIRED_TRAIN_KEYS = (
    "train_start",
    "train_end",
    "train_timesteps",
    "episode_length",
    "batch_size",
    "gamma",
    "policy_type",
    "policy_net_arch",
    "train_hydrology_transform",
    *ENVIRONMENT_KEYS,
)

CORRECTED_SETTING_REQUIREMENTS: dict[str, object] = {
    "decision_hydrology_timing": "previous_day",
    "niip_fallback_mode": "training_only",
    "storage_datum_mode": "elevation_2019",
    "storage_normalization": "train_window",
    "storage_budget_target_frac_of_max": 0.875,
    "mask_incomplete_initial_spr": True,
}

CORRECTED_REWARD_TOKENS = (
    "flooding:penalty_caps_archuleta_bluff",
    "esa_spring_peak_release:"
    "farmington_thresholds_actionproxy_smart_ledger_hammer_"
    "histfreq_stop_excess_strong_calendar",
)


class Phase95SpecError(ValueError):
    """Raised when a Phase 95 recovery specification is unsafe or incomplete."""


@dataclass(frozen=True)
class Phase95Task:
    task_id: int
    experiment: str
    seed: int
    outdir: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise Phase95SpecError(f"Expected a JSON object in {source}.")
    return value


def _write_json(path: str | Path, value: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _write_csv(path: str | Path, frame: pd.DataFrame) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            frame.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _retire_stale_output(path: Path) -> Path | None:
    if not path.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    retired = path.with_name(f"{path.stem}_stale_{stamp}{path.suffix}")
    os.replace(path, retired)
    return retired


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: dict[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _finite_metric(mapping: dict[str, Any], key: str, *, source: str) -> float:
    if key not in mapping:
        raise Phase95SpecError(f"{source} is missing required metric {key!r}.")
    try:
        value = float(mapping[key])
    except (TypeError, ValueError) as exc:
        raise Phase95SpecError(
            f"{source} metric {key!r} is not numeric: {mapping[key]!r}."
        ) from exc
    if not np.isfinite(value):
        raise Phase95SpecError(
            f"{source} metric {key!r} must be finite; found {value!r}."
        )
    return value


def build_corrected_objective_screen(
    rollout: pd.DataFrame,
    policy_metrics: dict[str, Any],
    *,
    window_name: str,
    window_start: str,
    window_end: str,
    spec_sha256: str,
    git_commit: object,
) -> dict[str, Any]:
    """Build the corrected ten-objective historic screen for one rollout."""

    historic_summary = drl_metrics.compute_historic_summary_metrics(rollout)
    flood = drl_metrics.compute_flooding_comparison_metrics(
        rollout,
        profile=drl_metrics.FLOOD_COMPARISON_PROFILE_CORRECTED,
    )

    policy = {
        metric: _finite_metric(policy_metrics, metric, source="policy metrics")
        for metric in OBJECTIVE_METRICS
    }
    historic = {
        metric: _finite_metric(historic_summary, metric, source="historic metrics")
        for metric in OBJECTIVE_METRICS
    }
    policy["flooding_frac_days_met"] = _finite_metric(
        flood, "agent_flooding_frac_days_met", source="paired flood comparison"
    )
    historic["flooding_frac_days_met"] = _finite_metric(
        flood, "historic_flooding_frac_days_met", source="paired flood comparison"
    )

    pass_flags: dict[str, bool] = {}
    raw_deltas: dict[str, float] = {}
    margins: dict[str, float] = {}
    for metric in OBJECTIVE_METRICS:
        policy_value = policy[metric]
        historic_value = historic[metric]
        raw_deltas[metric] = float(policy_value - historic_value)
        if metric == "total_spill_af":
            pass_flags[metric] = bool(policy_value <= historic_value + 1e-6)
            margins[metric] = float(historic_value - policy_value)
        else:
            pass_flags[metric] = bool(
                policy_value >= historic_value - OBJECTIVE_SCREEN_TOLERANCE
            )
            margins[metric] = float(policy_value - historic_value)

    benchmark = {
        "schema_version": 1,
        "metric_profile": OBJECTIVE_SCREEN_PROFILE,
        "window": {
            "name": window_name,
            "start": window_start,
            "end": window_end,
        },
        "historic_metrics": historic,
        "flood_comparison": {
            "profile": flood["flooding_comparison_profile"],
            "comparison_days": int(flood["flooding_comparison_days"]),
            "historic_available_days": int(
                flood["historic_flooding_available_days"]
            ),
            "historic_safe_days": int(flood["historic_flooding_safe_days"]),
        },
    }
    benchmark_sha256 = _canonical_sha256(benchmark)
    pass_count = int(sum(pass_flags.values()))
    return {
        "schema_version": 1,
        "metric_profile": OBJECTIVE_SCREEN_PROFILE,
        "window": benchmark["window"],
        "spec_sha256": spec_sha256,
        "git_commit": git_commit,
        "policy_checkpoint": "last_model.zip",
        "policy_metrics": policy,
        "historic_benchmark": historic,
        "policy_minus_historic": raw_deltas,
        "criterion_margin": margins,
        "pass_flags": pass_flags,
        "pass_count": pass_count,
        "objective_count": len(OBJECTIVE_METRICS),
        "all_objectives_passed": bool(pass_count == len(OBJECTIVE_METRICS)),
        "flood_comparison": flood,
        "benchmark": benchmark,
        "benchmark_sha256": benchmark_sha256,
    }


def _safe_name(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return re.sub(r"_+", "_", text).strip("._-") or "item"


def _git_state() -> dict[str, object]:
    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "worktree_dirty": None if status is None else bool(status),
    }


def load_and_validate_spec(path: str | Path) -> dict[str, Any]:
    """Load a Phase 95 recovery spec and validate its task contract."""

    spec = _read_json(path)
    if not str(spec.get("submission_name", "")).strip():
        raise Phase95SpecError("submission_name must be nonempty.")
    if not isinstance(spec.get("base_train"), dict):
        raise Phase95SpecError("base_train must be a JSON object.")
    if not isinstance(spec.get("base_eval"), dict):
        raise Phase95SpecError("base_eval must be a JSON object.")

    seeds = spec.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise Phase95SpecError("seeds must be a nonempty list.")
    normalized_seeds = [int(seed) for seed in seeds]
    if len(normalized_seeds) != len(set(normalized_seeds)):
        raise Phase95SpecError("seeds contains duplicate values.")

    experiments = spec.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise Phase95SpecError("experiments must be a nonempty list.")
    names: list[str] = []
    for index, experiment in enumerate(experiments):
        if not isinstance(experiment, dict):
            raise Phase95SpecError(f"experiments[{index}] must be an object.")
        name = str(experiment.get("name", "")).strip()
        if not name:
            raise Phase95SpecError(f"experiments[{index}].name must be nonempty.")
        names.append(name)
        resolved = resolve_train_settings(spec, experiment)
        validate_corrected_settings(resolved, context=name)
    if len(names) != len(set(names)):
        raise Phase95SpecError("experiment names must be unique.")

    windows = spec["base_eval"].get("windows")
    if not isinstance(windows, list) or not windows:
        raise Phase95SpecError("base_eval.windows must be a nonempty list.")
    for index, window in enumerate(windows):
        if not isinstance(window, dict) or not window.get("start") or not window.get("end"):
            raise Phase95SpecError(
                f"base_eval.windows[{index}] must contain start and end."
            )
    return spec


def resolve_train_settings(
    spec: dict[str, Any], experiment: dict[str, Any]
) -> dict[str, Any]:
    """Merge base settings with one experiment's controlled overrides."""

    settings = deepcopy(spec.get("base_train", {}))
    overrides = experiment.get("train_overrides", {})
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, dict):
        raise Phase95SpecError(
            f"{experiment.get('name', '<unnamed>')}.train_overrides must be an object."
        )
    unknown = sorted(set(overrides) - set(settings))
    if unknown:
        raise Phase95SpecError(
            f"Unknown train override(s) for {experiment.get('name')}: {unknown}. "
            "Declare new settings in base_train first."
        )
    settings.update(deepcopy(overrides))
    if experiment.get("reward_spec") is not None:
        settings["reward_spec"] = str(experiment["reward_spec"])

    missing = [key for key in REQUIRED_TRAIN_KEYS if settings.get(key) is None]
    if missing:
        raise Phase95SpecError(
            f"Missing required train setting(s) for {experiment.get('name')}: {missing}."
        )
    if not str(settings.get("reward_spec", "")).strip():
        raise Phase95SpecError(f"{experiment.get('name')} has no reward_spec.")
    return settings


def validate_corrected_settings(settings: dict[str, Any], *, context: str) -> None:
    """Require the retraining corrections that distinguish this new lineage."""

    for key, expected in CORRECTED_SETTING_REQUIREMENTS.items():
        actual = settings.get(key)
        if isinstance(expected, float):
            matches = actual is not None and math.isclose(float(actual), expected)
        else:
            matches = actual == expected
        if not matches:
            raise Phase95SpecError(
                f"{context} must use corrected {key}={expected!r}; found {actual!r}."
            )

    reward_spec = str(settings.get("reward_spec", ""))
    for token in CORRECTED_REWARD_TOKENS:
        if token not in reward_spec:
            raise Phase95SpecError(
                f"{context} reward_spec is missing corrected component {token!r}."
            )

    try:
        parsed_reward = drl_rewards.parse_objective_spec(reward_spec)
        drl_rewards.build_composite_reward(parsed_reward)
    except (KeyError, TypeError, ValueError) as exc:
        raise Phase95SpecError(
            f"{context} reward_spec does not resolve through the current "
            f"reward registry: {exc}"
        ) from exc

def experiment_by_name(spec: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [exp for exp in spec["experiments"] if exp.get("name") == name]
    if len(matches) != 1:
        raise Phase95SpecError(f"Expected one experiment named {name!r}, found {len(matches)}.")
    return dict(matches[0])


def build_tasks(
    spec: dict[str, Any],
    *,
    out_root: str | Path | None = None,
    seeds: list[int] | None = None,
    experiment_names: list[str] | None = None,
) -> list[Phase95Task]:
    root_value = spec.get("runs_root") if out_root is None else out_root
    if not root_value:
        raise Phase95SpecError("runs_root must be set in the spec or supplied explicitly.")
    root = Path(root_value)
    if not root.is_absolute():
        root = REPO_ROOT / root
    root = root.resolve()

    available_seeds = [int(seed) for seed in spec["seeds"]]
    selected_seeds = available_seeds if seeds is None else [int(seed) for seed in seeds]
    if len(selected_seeds) != len(set(selected_seeds)):
        raise Phase95SpecError("Selected seeds contain duplicates.")
    unknown_seeds = sorted(set(selected_seeds) - set(available_seeds))
    if unknown_seeds:
        raise Phase95SpecError(f"Requested seed(s) are not in the spec: {unknown_seeds}.")
    if not selected_seeds:
        raise Phase95SpecError("At least one seed must be selected.")

    experiments_by_name = {
        str(experiment["name"]): experiment for experiment in spec["experiments"]
    }
    selected_names = (
        list(experiments_by_name)
        if experiment_names is None
        else [str(name) for name in experiment_names]
    )
    if len(selected_names) != len(set(selected_names)):
        raise Phase95SpecError("Selected experiments contain duplicates.")
    unknown_experiments = sorted(set(selected_names) - set(experiments_by_name))
    if unknown_experiments:
        raise Phase95SpecError(
            f"Requested experiment(s) are not in the spec: {unknown_experiments}."
        )
    if not selected_names:
        raise Phase95SpecError("At least one experiment must be selected.")

    tasks: list[Phase95Task] = []
    # Seed-major order keeps matched variants adjacent in task and Slurm logs.
    for seed in selected_seeds:
        for name in selected_names:
            task_id = len(tasks)
            outdir = root / _safe_name(name) / f"seed_{int(seed):03d}"
            tasks.append(
                Phase95Task(
                    task_id=task_id,
                    experiment=name,
                    seed=int(seed),
                    outdir=str(outdir),
                )
            )
    return tasks


def environment_options(settings: dict[str, Any]) -> dict[str, Any]:
    return {key: settings[key] for key in ENVIRONMENT_KEYS}


def _resolved_timesteps(settings: dict[str, Any], override: int | None) -> tuple[int, int]:
    requested = int(settings["train_timesteps"] if override is None else override)
    episode_length = int(settings["episode_length"])
    resolved = requested // episode_length * episode_length
    if resolved <= 0:
        raise Phase95SpecError(
            f"Training budget {requested} must cover one {episode_length}-step rollout."
        )
    return requested, resolved


def _new_task_directory(path: Path, *, retry_failed: bool = False) -> str | None:
    archived: str | None = None
    if path.exists() and any(path.iterdir()):
        if not retry_failed:
            raise FileExistsError(f"Refusing to overwrite nonempty task directory: {path}")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archive = path.with_name(f"{path.name}__failed_{stamp}")
        suffix = 1
        while archive.exists():
            archive = path.with_name(f"{path.name}__failed_{stamp}_{suffix:02d}")
            suffix += 1
        path.rename(archive)
        archived = str(archive)
    path.mkdir(parents=True, exist_ok=True)
    return archived


@contextmanager
def _task_lock(outdir: Path):
    outdir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = outdir.parent / f".{outdir.name}.phase95.lock"
    metadata = {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "started_at": _utc_now(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    payload = (json.dumps(metadata, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    handle = os.fdopen(descriptor, "r+b", buffering=0)
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            if handle.read(1) == b"":
                handle.write(b"\0")
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, BlockingIOError) as exc:
        handle.close()
        raise RuntimeError(f"Task is already locked by another process: {lock_path}") from exc
    try:
        # Windows holds a one-byte region, so keep byte zero allocated and put
        # metadata after it. POSIX flock covers the whole file.
        handle.seek(1 if os.name == "nt" else 0)
        handle.write(payload)
        handle.truncate()
        os.fsync(handle.fileno())
        yield
    finally:
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _lock_task_run(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        task = kwargs.get("task")
        if not isinstance(task, Phase95Task):
            raise TypeError("run_task requires a keyword Phase95Task argument.")
        with _task_lock(Path(task.outdir)):
            return function(*args, **kwargs)

    return wrapped


def _completed_record_mismatches(
    record: dict[str, Any],
    *,
    task: Phase95Task,
    spec_sha256: str,
    git_commit: object,
    requested_timesteps: int,
    resolved_timesteps: int,
    diagnostics: bool,
    expected_evaluations: int | None,
    plan_id: str | None,
) -> list[str]:
    mismatches: list[str] = []
    if record.get("task") != asdict(task):
        mismatches.append("task identity")
    if record.get("spec_sha256") != spec_sha256:
        mismatches.append("spec hash")
    if record.get("git", {}).get("commit") != git_commit:
        mismatches.append("Git commit")
    if plan_id is not None and record.get("plan_id") != plan_id:
        mismatches.append("plan ID")
    if int(record.get("requested_timesteps", -1)) != int(requested_timesteps):
        mismatches.append("requested timesteps")
    if int(record.get("resolved_timesteps", -1)) != int(resolved_timesteps):
        mismatches.append("resolved timesteps")
    if bool(record.get("rich_training_diagnostics", False)) != bool(diagnostics):
        mismatches.append("diagnostics mode")
    if expected_evaluations is not None and len(record.get("evaluations", [])) != int(
        expected_evaluations
    ):
        mismatches.append("evaluation windows")
    return mismatches


def _recovery_diagnostics(rollout: pd.DataFrame) -> dict[str, float | int]:
    """Return supplementary controller diagnostics for a recovery rollout."""

    def numeric(column: str) -> pd.Series:
        if column not in rollout:
            return pd.Series(dtype=float)
        return pd.to_numeric(rollout[column], errors="coerce")

    def bool_count(column: str) -> int:
        if column not in rollout:
            return 0
        return int(rollout[column].fillna(False).astype(bool).sum())

    spr_attributed_release = numeric("spr_attributed_sj_release_cfs")
    spr_marginal_request = numeric("spr_proxy_added_request_cfs")
    spr_full_need = numeric("spr_proxy_full_controller_need_cfs")
    spr_applied_need = numeric("spr_proxy_controller_need_cfs")
    niip_release = numeric("release_niip_cfs")
    sj_release = numeric("release_sj_main_cfs")
    target_hits = rollout.get("spr_proxy_target_hit")
    active = rollout.get("spr_proxy_window_active")
    target_cfs = numeric("spr_proxy_target_cfs")
    action_spr = numeric("action_2")
    window_mask = pd.Series(False, index=rollout.index, dtype=bool)
    active_hit_fraction = float("nan")
    if target_hits is not None and active is not None:
        window_mask = active.fillna(False).astype(bool)
        if bool(window_mask.any()):
            active_hit_fraction = float(
                target_hits.fillna(False).astype(bool).loc[window_mask].mean()
            )

    reachable = rollout.get("spr_proxy_target_reachable_at_decision")
    reachable_mask = pd.Series(False, index=rollout.index, dtype=bool)
    if reachable is not None:
        reachable_mask = window_mask & reachable.fillna(False).astype(bool)

    def masked_mean(values: pd.Series, mask: pd.Series) -> float:
        return float(values.loc[mask].mean()) if bool(mask.any()) else float("nan")

    target_fractions = {
        f"spr_target_{threshold}_fraction_window": masked_mean(
            (target_cfs == float(threshold)).astype(float), window_mask
        )
        for threshold in (0, 2500, 5000, 8000, 10000)
    }

    reward_columns = [column for column in rollout if str(column).startswith("rc_")]
    spr_reward_columns = [
        column
        for column in reward_columns
        if str(column).startswith("rc_esa_spring_peak_release.")
    ]

    def reward_abs_share(mask: pd.Series) -> float:
        if not reward_columns or not spr_reward_columns or not bool(mask.any()):
            return float("nan")
        all_abs = (
            rollout.loc[mask, reward_columns]
            .apply(pd.to_numeric, errors="coerce")
            .abs()
            .to_numpy()
            .sum()
        )
        spr_abs = (
            rollout.loc[mask, spr_reward_columns]
            .apply(pd.to_numeric, errors="coerce")
            .abs()
            .to_numpy()
            .sum()
        )
        return float(spr_abs / all_abs) if all_abs > 0.0 else float("nan")

    all_days_mask = pd.Series(True, index=rollout.index, dtype=bool)

    return {
        "n_days": int(len(rollout)),
        "spr_window_days": bool_count("spr_proxy_window_active"),
        "water_limited_days": int((numeric("release_phys_penalty") > 0.0).sum()),
        "spr_priority_scaled_days": bool_count("spr_proxy_priority_scaled"),
        "spr_target_hit_fraction_active": active_hit_fraction,
        "spr_priority_attributed_release_af": float(
            spr_attributed_release.fillna(0.0).sum() * CFS_DAY_TO_AF
        ),
        "spr_marginal_controller_request_af": float(
            spr_marginal_request.fillna(0.0).sum() * CFS_DAY_TO_AF
        ),
        "spr_full_bridge_need_af": float(spr_full_need.fillna(0.0).sum() * CFS_DAY_TO_AF),
        "spr_applied_bridge_need_af": float(
            spr_applied_need.fillna(0.0).sum() * CFS_DAY_TO_AF
        ),
        "controlled_sj_release_af": float(sj_release.fillna(0.0).sum() * CFS_DAY_TO_AF),
        "niip_release_af": float(niip_release.fillna(0.0).sum() * CFS_DAY_TO_AF),
        "mean_action_spr_all_days": float(action_spr.mean()),
        "mean_action_spr_window": masked_mean(action_spr, window_mask),
        "mean_action_spr_reachable_window": masked_mean(action_spr, reachable_mask),
        "spr_nonzero_target_fraction_window": masked_mean(
            (target_cfs > 0.0).astype(float), window_mask
        ),
        "spr_nonzero_target_fraction_off_window": masked_mean(
            (target_cfs > 0.0).astype(float), ~window_mask
        ),
        "spr_reward_abs_share_all_days": reward_abs_share(all_days_mask),
        "spr_reward_abs_share_window": reward_abs_share(window_mask),
        "mean_action_niip_all_days": float(numeric("action_1").mean()),
        "mean_action_discretionary_sj_all_days": float(numeric("action_3").mean()),
        **target_fractions,
    }


@_lock_task_run
def run_task(
    *,
    spec_path: str | Path,
    task: Phase95Task,
    timesteps_override: int | None = None,
    skip_eval: bool = False,
    diagnostics: bool = False,
    device: str = "cpu",
    torch_threads: int = 1,
    skip_complete: bool = True,
    retry_failed: bool = False,
    plan_id: str | None = None,
) -> dict[str, Any]:
    """Train and evaluate one materialized Phase 95 recovery task."""

    spec_path = Path(spec_path).resolve()
    spec = load_and_validate_spec(spec_path)
    experiment = experiment_by_name(spec, task.experiment)
    settings = resolve_train_settings(spec, experiment)
    requested, resolved = _resolved_timesteps(settings, timesteps_override)
    outdir = Path(task.outdir)
    result_path = outdir / "phase95-task.json"
    spec_sha256 = _sha256(spec_path)
    git = _git_state()
    if skip_complete and result_path.exists():
        try:
            prior = _read_json(result_path)
        except (OSError, json.JSONDecodeError, Phase95SpecError) as exc:
            if not retry_failed:
                raise Phase95SpecError(
                    f"Unreadable existing task record {result_path}; use "
                    "--retry-failed only after confirming the prior job ended."
                ) from exc
            prior = None
        if prior is not None and prior.get("status") == "completed":
            mismatches = _completed_record_mismatches(
                prior,
                task=task,
                spec_sha256=spec_sha256,
                git_commit=git.get("commit"),
                requested_timesteps=requested,
                resolved_timesteps=resolved,
                diagnostics=diagnostics,
                expected_evaluations=(
                    None if skip_eval else len(spec["base_eval"]["windows"])
                ),
                plan_id=plan_id,
            )
            if mismatches:
                raise FileExistsError(
                    f"Completed task output is incompatible ({', '.join(mismatches)}): "
                    f"{outdir}"
                )
            print(f"[phase95] task {task.task_id} already complete: {outdir}")
            return prior

    archived_attempt = _new_task_directory(outdir, retry_failed=retry_failed)
    record: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": _utc_now(),
        "task": asdict(task),
        "submission_name": spec["submission_name"],
        "plan_id": plan_id,
        "spec_path": str(spec_path),
        "spec_sha256": spec_sha256,
        "git": git,
        "experiment": experiment,
        "resolved_train_settings": settings,
        "requested_timesteps": requested,
        "resolved_timesteps": resolved,
        "device": device,
        "torch_threads": int(torch_threads),
        "rich_training_diagnostics": bool(diagnostics),
        "archived_failed_attempt": archived_attempt,
        "runtime": {
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
    }
    _write_json(result_path, record)

    try:
        import torch

        if int(torch_threads) < 1:
            raise Phase95SpecError("torch_threads must be positive.")
        torch.set_num_threads(int(torch_threads))
        torch.set_num_interop_threads(int(torch_threads))

        constructor_keys = (
            "gamma",
            "gae_lambda",
            "learning_rate",
            "clip_range",
            "ent_coef",
            "vf_coef",
            "max_grad_norm",
            "target_kl",
            "action_log_std_init",
        )
        constructor_options = {
            key: settings[key] for key in constructor_keys if settings.get(key) is not None
        }
        run = model.DRLModel(
            reward_spec=settings["reward_spec"],
            train_start=settings["train_start"],
            train_end=settings["train_end"],
            train_hydrology_transform=settings["train_hydrology_transform"],
            logdir=outdir,
            seed=int(task.seed),
            device=device,
            policy_type=settings["policy_type"],
            policy_net_arch=settings["policy_net_arch"],
            episode_length_train=int(settings["episode_length"]),
            launch_mode="phase95_corrected_recovery",
            **constructor_options,
            **environment_options(settings),
        )
        run.train(
            n_episodes=resolved // int(settings["episode_length"]),
            total_timesteps=resolved,
            n_steps=int(settings.get("ppo_n_steps", settings["episode_length"])),
            batch_size=int(settings["batch_size"]),
            n_epochs=int(settings.get("n_epochs", 10)),
            gamma=float(settings["gamma"]),
            gae_lambda=settings.get("gae_lambda"),
            learning_rate=settings.get("learning_rate"),
            clip_range=settings.get("clip_range"),
            ent_coef=settings.get("ent_coef"),
            vf_coef=settings.get("vf_coef"),
            max_grad_norm=settings.get("max_grad_norm"),
            target_kl=settings.get("target_kl"),
            action_log_std_init=settings.get("action_log_std_init"),
            policy_type=settings["policy_type"],
            policy_net_arch=settings["policy_net_arch"],
            track_reward_components=True,
            rich_training_diagnostics=bool(diagnostics),
        )

        evaluations: list[dict[str, object]] = []
        if not skip_eval:
            eval_settings = spec["base_eval"]
            for window in eval_settings["windows"]:
                name = _safe_name(window.get("name") or f"{window['start']}_{window['end']}")
                eval_outdir = outdir / f"eval__{name}"
                rollout, metrics = model.evaluate_model_window(
                    model_path=outdir / "last_model.zip",
                    reward_spec=settings["reward_spec"],
                    window_start=str(window["start"]),
                    window_end=str(window["end"]),
                    outdir=eval_outdir,
                    device=device,
                    which_metrics=str(eval_settings.get("which_metrics", "all")),
                    save_rollout=bool(eval_settings.get("save_rollout", True)),
                    save_metrics=bool(eval_settings.get("save_metrics", True)),
                    **environment_options(settings),
                )
                if metrics.empty:
                    raise Phase95SpecError(
                        f"Evaluation produced no metric row for {task.experiment} "
                        f"seed {task.seed}, window {name}."
                    )
                metric_row = metrics.iloc[0].to_dict()
                diagnostics_row = _recovery_diagnostics(rollout)
                _write_json(
                    eval_outdir / "recovery_diagnostics.json", diagnostics_row
                )
                screen = build_corrected_objective_screen(
                    rollout,
                    metric_row,
                    window_name=name,
                    window_start=str(window["start"]),
                    window_end=str(window["end"]),
                    spec_sha256=spec_sha256,
                    git_commit=git.get("commit"),
                )
                _write_json(
                    eval_outdir / "corrected_objective_screen.json", screen
                )
                _write_json(
                    eval_outdir / "corrected_historic_benchmark.json",
                    {
                        **screen["benchmark"],
                        "benchmark_sha256": screen["benchmark_sha256"],
                    },
                )
                evaluations.append(
                    {
                        "name": name,
                        "outdir": str(eval_outdir),
                        "n_days": int(len(rollout)),
                        "n_metric_rows": int(len(metrics)),
                        "recovery_diagnostics": diagnostics_row,
                        "objective_pass_count": screen["pass_count"],
                        "all_objectives_passed": screen[
                            "all_objectives_passed"
                        ],
                        "benchmark_sha256": screen["benchmark_sha256"],
                    }
                )

        record["status"] = "completed"
        record["ended_at"] = _utc_now()
        record["evaluations"] = evaluations
        _write_json(result_path, record)
        print(f"[phase95] completed task {task.task_id}: {outdir}")
        return record
    except BaseException as exc:
        record["status"] = "failed"
        record["ended_at"] = _utc_now()
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
        _write_json(result_path, record)
        raise


def materialize_plan(
    *,
    spec_path: str | Path,
    outdir: str | Path,
    out_root: str | Path | None = None,
    seeds: list[int] | None = None,
    experiment_names: list[str] | None = None,
    timesteps_override: int | None = None,
    skip_eval: bool = False,
    diagnostics: bool = False,
    require_clean: bool = True,
) -> dict[str, Any]:
    """Write a reviewed task matrix and Darwin Slurm scripts without submitting."""

    spec_path = Path(spec_path).resolve()
    spec = load_and_validate_spec(spec_path)
    control_dir = Path(outdir).resolve()
    if control_dir.exists() and any(control_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty plan directory: {control_dir}")
    tasks = build_tasks(
        spec,
        out_root=out_root,
        seeds=seeds,
        experiment_names=experiment_names,
    )
    if timesteps_override is not None:
        for experiment_name in {task.experiment for task in tasks}:
            settings = resolve_train_settings(
                spec, experiment_by_name(spec, experiment_name)
            )
            _resolved_timesteps(settings, timesteps_override)

    git = _git_state()
    if require_clean:
        if git.get("commit") is None:
            raise Phase95SpecError("Cannot resolve the Git commit for this plan.")
        if git.get("worktree_dirty") is not False:
            raise Phase95SpecError(
                "Refusing to materialize an HPC plan from a dirty worktree. "
                "Commit the reviewed experiment code and configuration first."
            )
    control_dir.mkdir(parents=True, exist_ok=True)

    slurm = dict(spec.get("slurm", {}))
    python_executable = str(slurm.get("python_executable", sys.executable))
    requested_packed_tasks = int(slurm.get("packed_tasks_per_node", 1))
    if requested_packed_tasks < 1:
        raise Phase95SpecError("slurm.packed_tasks_per_node must be positive.")
    packed_tasks = min(requested_packed_tasks, len(tasks))
    pack_count = int(math.ceil(len(tasks) / packed_tasks))
    parallelism = int(slurm.get("array_parallelism", pack_count))
    if parallelism < 1:
        raise Phase95SpecError("slurm.array_parallelism must be positive.")

    plan = {
        "schema_version": 1,
        "created_at": _utc_now(),
        "submission_name": spec["submission_name"],
        "spec_path": str(spec_path),
        "spec_sha256": _sha256(spec_path),
        "repo_root": str(REPO_ROOT.resolve()),
        "control_dir": str(control_dir),
        "python_executable": python_executable,
        "requested_packed_tasks_per_node": requested_packed_tasks,
        "packed_tasks_per_node": packed_tasks,
        "pack_count": pack_count,
        "selected_seeds": sorted({task.seed for task in tasks}),
        "selected_experiments": list(dict.fromkeys(task.experiment for task in tasks)),
        "timesteps_override": (
            None if timesteps_override is None else int(timesteps_override)
        ),
        "skip_eval": bool(skip_eval),
        "rich_training_diagnostics": bool(diagnostics),
        "tasks": [asdict(task) for task in tasks],
        "git": git,
    }
    plan["plan_id"] = _canonical_sha256(plan)
    plan_path = control_dir / "plan.json"
    _write_json(plan_path, plan)

    with (control_dir / "task_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("task_id", "experiment", "seed", "outdir")
        )
        writer.writeheader()
        writer.writerows(asdict(task) for task in tasks)

    slurm_logs = control_dir / "slurm_logs"
    packed_logs = control_dir / "packed_task_logs"
    slurm_logs.mkdir(exist_ok=True)
    packed_logs.mkdir(exist_ok=True)
    array_range = f"0-{pack_count - 1}%{min(parallelism, pack_count)}"
    job_name = _safe_name(slurm.get("job_name", spec["submission_name"]))
    setup = [
        "set -euo pipefail",
        f"cd {shlex.quote(str(REPO_ROOT.resolve()))}",
        f"export PYTHONPATH={shlex.quote(str(REPO_ROOT.resolve() / 'src'))}:${{PYTHONPATH:-}}",
        'export CUDA_VISIBLE_DEVICES=""',
        "export PYTHONUNBUFFERED=1",
        "export OMP_NUM_THREADS=1",
        "export MKL_NUM_THREADS=1",
        "export OPENBLAS_NUM_THREADS=1",
        "export NUMEXPR_NUM_THREADS=1",
        "export TORCH_NUM_THREADS=1",
    ]
    setup.extend(str(line) for line in slurm.get("setup_commands", []))
    extra_sbatch = [
        f"#SBATCH {line}" for line in slurm.get("extra_sbatch_lines", [])
    ]
    array_script = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --array={array_range}",
        f"#SBATCH --output={slurm_logs}/%x_%A_%a.out",
        f"#SBATCH --error={slurm_logs}/%x_%A_%a.err",
        f"#SBATCH --partition={slurm.get('partition', 'scaling')}",
        f"#SBATCH --cpus-per-task={packed_tasks}",
        f"#SBATCH --time={slurm.get('time', '09:30:00')}",
        "#SBATCH --no-requeue",
        *extra_sbatch,
        *setup,
        'PACK_ID="${SLURM_ARRAY_TASK_ID}"',
        (
            f"{shlex.quote(python_executable)} -B -m "
            "deepreservoir.drl.phase95_recovery run-pack "
            f"--plan {shlex.quote(str(plan_path))} --pack-id \"${{PACK_ID}}\""
        ),
    ]
    (control_dir / "batch_array.sbatch").write_text(
        "\n".join(array_script) + "\n", encoding="utf-8", newline="\n"
    )

    report_script = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}_report",
        f"#SBATCH --output={slurm_logs}/%x_%j.out",
        f"#SBATCH --error={slurm_logs}/%x_%j.err",
        f"#SBATCH --partition={slurm.get('partition', 'scaling')}",
        "#SBATCH --cpus-per-task=1",
        "#SBATCH --time=00:20:00",
        *extra_sbatch,
        *setup,
        (
            f"{shlex.quote(python_executable)} -B -m "
            "deepreservoir.drl.phase95_recovery collect "
            f"--plan {shlex.quote(str(plan_path))}"
        ),
    ]
    (control_dir / "batch_report.sbatch").write_text(
        "\n".join(report_script) + "\n", encoding="utf-8", newline="\n"
    )
    retry_script = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}_retry",
        f"#SBATCH --output={slurm_logs}/%x_%j.out",
        f"#SBATCH --error={slurm_logs}/%x_%j.err",
        f"#SBATCH --partition={slurm.get('partition', 'scaling')}",
        f"#SBATCH --cpus-per-task={packed_tasks}",
        f"#SBATCH --time={slurm.get('time', '09:30:00')}",
        "#SBATCH --no-requeue",
        *extra_sbatch,
        *setup,
        ': "${PACK_ID:?Submit with --export=ALL,PACK_ID=<failed-pack-id>}"',
        (
            f"{shlex.quote(python_executable)} -B -m "
            "deepreservoir.drl.phase95_recovery run-pack "
            f"--plan {shlex.quote(str(plan_path))} --pack-id \"${{PACK_ID}}\" "
            "--retry-failed"
        ),
    ]
    (control_dir / "retry_pack.sbatch").write_text(
        "\n".join(retry_script) + "\n", encoding="utf-8", newline="\n"
    )
    retry_submit_script = [
        "#!/bin/bash",
        "set -euo pipefail",
        'HERE="$(cd "$(dirname "$0")" && pwd)"',
        'PACK_ID="${1:-}"',
        f'if [[ ! "$PACK_ID" =~ ^[0-9]+$ ]] || (( PACK_ID >= {pack_count} )); then',
        f'  echo "Usage: $0 <pack-id 0..{pack_count - 1}>" >&2',
        "  exit 2",
        "fi",
        'if ! mkdir "$HERE/.retry-submission.lock" 2>/dev/null; then',
        '  echo "Another retry submission is in progress." >&2',
        "  exit 2",
        "fi",
        'trap \'rmdir "$HERE/.retry-submission.lock" 2>/dev/null || true\' EXIT',
        'STAMP=$(date -u +%Y%m%dT%H%M%SZ)',
        'RETRY_JOBID=$(sbatch --parsable --export=ALL,PACK_ID="${PACK_ID}" "$HERE/retry_pack.sbatch")',
        'JOB_TOKEN="${RETRY_JOBID%%;*}"',
        'RECORD="$HERE/retry_${STAMP}_job_${JOB_TOKEN}_pack_${PACK_ID}.json"',
        'printf \'{"pack_id":%s,"retry_job_id":"%s","report_job_id":null,"submitted_at":"%s","status":"retry_submitted"}\\n\' "${PACK_ID}" "${RETRY_JOBID}" "${STAMP}" > "${RECORD}"',
        'REPORT_JOBID=$(sbatch --parsable --dependency=afterany:${JOB_TOKEN} "$HERE/batch_report.sbatch")',
        'printf \'{"pack_id":%s,"retry_job_id":"%s","report_job_id":"%s","submitted_at":"%s","status":"submitted"}\\n\' "${PACK_ID}" "${RETRY_JOBID}" "${REPORT_JOBID}" "${STAMP}" > "${RECORD}.tmp"',
        'mv "${RECORD}.tmp" "${RECORD}"',
        'echo "Submitted retry job ${RETRY_JOBID} and report job ${REPORT_JOBID}."',
    ]
    (control_dir / "retry_failed_pack.sh").write_text(
        "\n".join(retry_submit_script) + "\n", encoding="utf-8", newline="\n"
    )
    submit_script = [
        "#!/bin/bash",
        "set -euo pipefail",
        'HERE="$(cd "$(dirname "$0")" && pwd)"',
        'if [[ -e "$HERE/submission.json" || -d "$HERE/.submission.lock" ]]; then',
        '  echo "Refusing duplicate submission; inspect $HERE/submission.json and Slurm state." >&2',
        "  exit 2",
        "fi",
        'mkdir "$HERE/.submission.lock"',
        'trap \'rmdir "$HERE/.submission.lock" 2>/dev/null || true\' EXIT',
        'ARRAY_JOBID=$(sbatch --parsable "$HERE/batch_array.sbatch")',
        'echo "Submitted Phase 95 array job: ${ARRAY_JOBID}"',
        'ARRAY_JOB_TOKEN="${ARRAY_JOBID%%;*}"',
        'SUBMITTED_AT=$(date --iso-8601=seconds)',
        'printf \'{"array_job_id":"%s","report_job_id":null,"submitted_at":"%s","status":"array_submitted"}\\n\' "${ARRAY_JOBID}" "${SUBMITTED_AT}" > "$HERE/submission.json"',
        'REPORT_JOBID=$(sbatch --parsable --dependency=afterany:${ARRAY_JOB_TOKEN} "$HERE/batch_report.sbatch")',
        'echo "Submitted Phase 95 report job: ${REPORT_JOBID}"',
        'printf \'{"array_job_id":"%s","report_job_id":"%s","submitted_at":"%s","status":"submitted"}\\n\' "${ARRAY_JOBID}" "${REPORT_JOBID}" "${SUBMITTED_AT}" > "$HERE/submission.json.tmp"',
        'mv "$HERE/submission.json.tmp" "$HERE/submission.json"',
    ]
    (control_dir / "submit_batch.sh").write_text(
        "\n".join(submit_script) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        f"[phase95] materialized {len(tasks)} tasks in {pack_count} pack(s): {control_dir}"
    )
    return plan


def _load_plan(path: str | Path) -> dict[str, Any]:
    plan = _read_json(path)
    recorded_plan_id = plan.get("plan_id")
    unsigned_plan = {key: value for key, value in plan.items() if key != "plan_id"}
    if not recorded_plan_id or _canonical_sha256(unsigned_plan) != recorded_plan_id:
        raise Phase95SpecError(
            "The materialized plan contents do not match its plan ID; create a new plan."
        )
    spec_path = Path(plan["spec_path"])
    actual_hash = _sha256(spec_path)
    if actual_hash != plan.get("spec_sha256"):
        raise Phase95SpecError(
            "The experiment spec changed after plan materialization; create and review a new plan."
        )
    planned_git = plan.get("git", {})
    current_git = _git_state()
    planned_commit = planned_git.get("commit")
    if not planned_commit or current_git.get("commit") != planned_commit:
        raise Phase95SpecError(
            "The checked-out commit does not match the materialized experiment plan."
        )
    if current_git.get("worktree_dirty") is not False:
        raise Phase95SpecError(
            "Refusing to run a materialized experiment plan from a dirty worktree."
        )
    return plan


def _task_from_plan(plan: dict[str, Any], task_id: int) -> Phase95Task:
    tasks = plan.get("tasks", [])
    if task_id < 0 or task_id >= len(tasks):
        raise Phase95SpecError(f"task_id {task_id} is outside 0..{len(tasks) - 1}.")
    return Phase95Task(**tasks[task_id])


def _plan_execution_contract(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "timesteps_override": plan.get("timesteps_override"),
        "skip_eval": bool(plan.get("skip_eval", False)),
        "diagnostics": bool(plan.get("rich_training_diagnostics", False)),
        "plan_id": plan.get("plan_id"),
    }


def run_pack(
    *,
    plan_path: str | Path,
    pack_id: int,
    retry_failed: bool = False,
) -> dict[str, Any]:
    """Run one packed Slurm allocation as independent one-thread processes."""

    plan_path = Path(plan_path).resolve()
    plan = _load_plan(plan_path)
    contract = _plan_execution_contract(plan)
    effective_timesteps = contract["timesteps_override"]
    effective_skip_eval = contract["skip_eval"]
    effective_diagnostics = contract["diagnostics"]
    pack_size = int(plan["packed_tasks_per_node"])
    start = int(pack_id) * pack_size
    stop = min(start + pack_size, len(plan["tasks"]))
    if start >= stop:
        raise Phase95SpecError(f"pack_id {pack_id} has no tasks.")

    logdir = Path(plan["control_dir"]) / "packed_task_logs"
    logdir.mkdir(parents=True, exist_ok=True)
    attempt_time = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    job_token = _safe_name(os.environ.get("SLURM_JOB_ID", "local"))
    attempt_stamp = f"{attempt_time}_{job_token}_pid{os.getpid()}"
    env = os.environ.copy()
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TORCH_NUM_THREADS",
    ):
        env[key] = "1"

    processes: list[tuple[int, subprocess.Popen[str], Any, Any]] = []
    for task_id in range(start, stop):
        stdout_handle = (logdir / f"pack_{int(pack_id):04d}_task_{task_id:04d}_{attempt_stamp}.out").open(
            "w", encoding="utf-8"
        )
        stderr_handle = (logdir / f"pack_{int(pack_id):04d}_task_{task_id:04d}_{attempt_stamp}.err").open(
            "w", encoding="utf-8"
        )
        command = [
            sys.executable,
            "-B",
            "-m",
            "deepreservoir.drl.phase95_recovery",
            "run-task",
            "--plan",
            str(plan_path),
            "--task-id",
            str(task_id),
        ]
        if retry_failed:
            command.append("--retry-failed")
        process = subprocess.Popen(
            command,
            cwd=plan["repo_root"],
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
        )
        processes.append((task_id, process, stdout_handle, stderr_handle))
        print(f"[phase95] pack {pack_id} launched task {task_id} pid={process.pid}")

    failures: list[dict[str, int]] = []
    for task_id, process, stdout_handle, stderr_handle in processes:
        returncode = int(process.wait())
        stdout_handle.close()
        stderr_handle.close()
        if returncode != 0:
            failures.append({"task_id": task_id, "returncode": returncode})
    result = {
        "pack_id": int(pack_id),
        "task_ids": list(range(start, stop)),
        "timesteps_override": effective_timesteps,
        "skip_eval": effective_skip_eval,
        "rich_training_diagnostics": effective_diagnostics,
        "retry_failed": bool(retry_failed),
        "plan_id": plan.get("plan_id"),
        "attempt_stamp": attempt_stamp,
        "failures": failures,
        "completed_at": _utc_now(),
    }
    _write_json(
        Path(plan["control_dir"])
        / f"pack_{int(pack_id):04d}_{attempt_stamp}.json",
        result,
    )
    if failures:
        raise RuntimeError(f"Phase 95 pack {pack_id} had failed tasks: {failures}")
    return result


def collect_results(plan_path: str | Path) -> dict[str, Any]:
    """Collect task metrics and recovery diagnostics into CSV tables."""

    plan = _load_plan(plan_path)
    spec = load_and_validate_spec(plan["spec_path"])
    windows = spec["base_eval"]["windows"]
    contract = _plan_execution_contract(plan)
    rows: list[dict[str, object]] = []
    screen_rows: list[dict[str, object]] = []
    benchmarks: dict[str, dict[str, Any]] = {}
    status_rows: list[dict[str, object]] = []
    missing: list[int] = []
    for task_value in plan["tasks"]:
        task = Phase95Task(**task_value)
        result_path = Path(task.outdir) / "phase95-task.json"
        if not result_path.exists():
            status_rows.append(
                {
                    "task_id": task.task_id,
                    "experiment": task.experiment,
                    "seed": task.seed,
                    "status": "missing",
                }
            )
            missing.append(task.task_id)
            continue
        try:
            result = _read_json(result_path)
        except (OSError, json.JSONDecodeError, Phase95SpecError) as exc:
            status_rows.append(
                {
                    "task_id": task.task_id,
                    "experiment": task.experiment,
                    "seed": task.seed,
                    "status": "invalid_record",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            missing.append(task.task_id)
            continue
        settings = resolve_train_settings(
            spec, experiment_by_name(spec, task.experiment)
        )
        requested, resolved = _resolved_timesteps(
            settings, contract["timesteps_override"]
        )
        mismatches = _completed_record_mismatches(
            result,
            task=task,
            spec_sha256=plan["spec_sha256"],
            git_commit=plan["git"]["commit"],
            requested_timesteps=requested,
            resolved_timesteps=resolved,
            diagnostics=contract["diagnostics"],
            expected_evaluations=(
                None
                if contract["skip_eval"] or result.get("status") != "completed"
                else len(windows)
            ),
            plan_id=contract["plan_id"],
        )
        actual_status = str(result.get("status", "unknown"))
        reported_status = (
            "incompatible"
            if actual_status == "completed" and mismatches
            else actual_status
        )
        base: dict[str, object] = {
            "task_id": task.task_id,
            "experiment": task.experiment,
            "seed": task.seed,
            "status": reported_status,
            "git_commit": result.get("git", {}).get("commit"),
            "plan_id": result.get("plan_id"),
        }
        status_entry: dict[str, object] = {
            **base,
            "requested_timesteps": result.get("requested_timesteps"),
            "resolved_timesteps": result.get("resolved_timesteps"),
            "started_at": result.get("started_at"),
            "ended_at": result.get("ended_at"),
            "error": (
                f"Incompatible task record: {', '.join(mismatches)}"
                if actual_status == "completed" and mismatches
                else result.get("error")
            ),
            "compatibility_mismatches": "; ".join(mismatches),
        }
        if actual_status != "completed" or mismatches:
            status_rows.append(status_entry)
            missing.append(task.task_id)
            continue
        if contract["skip_eval"]:
            status_rows.append(status_entry)
            continue
        task_rows: list[dict[str, object]] = []
        task_screen_rows: list[dict[str, object]] = []
        task_benchmarks: dict[str, dict[str, Any]] = {}
        missing_evaluations: list[str] = []
        for window in windows:
            window_name = _safe_name(window.get("name") or f"{window['start']}_{window['end']}")
            eval_dir = Path(task.outdir) / f"eval__{window_name}"
            metrics_path = eval_dir / "eval_metrics.csv"
            diagnostics_path = eval_dir / "recovery_diagnostics.json"
            screen_path = eval_dir / "corrected_objective_screen.json"
            benchmark_path = eval_dir / "corrected_historic_benchmark.json"
            required_paths = (
                metrics_path,
                diagnostics_path,
                screen_path,
                benchmark_path,
            )
            if not all(path.exists() for path in required_paths):
                missing_evaluations.append(window_name)
                continue
            try:
                metrics_frame = pd.read_csv(metrics_path)
                if metrics_frame.empty:
                    raise ValueError(f"Empty metrics file: {metrics_path}")
                metrics = metrics_frame.iloc[0].to_dict()
                diagnostics_row = _read_json(diagnostics_path)
                screen = _read_json(screen_path)
                benchmark_file = _read_json(benchmark_path)
                benchmark = screen.get("benchmark")
                benchmark_hash = str(screen.get("benchmark_sha256", ""))
                if not isinstance(benchmark, dict) or not benchmark_hash:
                    raise Phase95SpecError(
                        f"Invalid corrected benchmark payload: {screen_path}"
                    )
                if _canonical_sha256(benchmark) != benchmark_hash:
                    raise Phase95SpecError(
                        f"Corrected benchmark hash mismatch: {screen_path}"
                    )
                if benchmark_file != {
                    **benchmark,
                    "benchmark_sha256": benchmark_hash,
                }:
                    raise Phase95SpecError(
                        f"Corrected benchmark artifact mismatch: {benchmark_path}"
                    )
                if screen.get("metric_profile") != OBJECTIVE_SCREEN_PROFILE:
                    raise Phase95SpecError(
                        f"Unexpected metric profile in {screen_path}."
                    )
                if screen.get("spec_sha256") != plan["spec_sha256"]:
                    raise Phase95SpecError(
                        f"Spec hash mismatch in corrected screen: {screen_path}"
                    )
                if screen.get("git_commit") != plan["git"]["commit"]:
                    raise Phase95SpecError(
                        f"Git commit mismatch in corrected screen: {screen_path}"
                    )
                expected_window = {
                    "name": window_name,
                    "start": str(window["start"]),
                    "end": str(window["end"]),
                }
                if screen.get("window") != expected_window:
                    raise Phase95SpecError(
                        f"Window mismatch in corrected screen: {screen_path}"
                    )

                screen_row: dict[str, object] = {
                    **base,
                    "window": window_name,
                    "metric_profile": screen["metric_profile"],
                    "benchmark_sha256": benchmark_hash,
                    "objective_pass_count": int(screen["pass_count"]),
                    "objective_count": int(screen["objective_count"]),
                    "all_objectives_passed": bool(
                        screen["all_objectives_passed"]
                    ),
                }
                for metric in OBJECTIVE_METRICS:
                    screen_row[f"policy_{metric}"] = screen["policy_metrics"][
                        metric
                    ]
                    screen_row[f"historic_{metric}"] = screen[
                        "historic_benchmark"
                    ][metric]
                    screen_row[f"delta_{metric}"] = screen[
                        "policy_minus_historic"
                    ][metric]
                    screen_row[f"margin_{metric}"] = screen[
                        "criterion_margin"
                    ][metric]
                    screen_row[f"pass_{metric}"] = bool(
                        screen["pass_flags"][metric]
                    )
                for key, value in screen["flood_comparison"].items():
                    screen_row[f"flood_{key}"] = value
                metrics = dict(metrics)
                metrics["flooding_frac_days_met_unpaired"] = metrics.get(
                    "flooding_frac_days_met"
                )
                metrics["flooding_frac_days_met"] = screen["policy_metrics"][
                    "flooding_frac_days_met"
                ]
            except (
                OSError,
                KeyError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
                Phase95SpecError,
            ):
                missing_evaluations.append(window_name)
                continue
            task_rows.append(
                {
                    **base,
                    "window": window_name,
                    **metrics,
                    **diagnostics_row,
                    **{
                        key: value
                        for key, value in screen_row.items()
                        if key not in base and key != "window"
                    },
                }
            )
            task_screen_rows.append(screen_row)
            task_benchmarks[benchmark_hash] = benchmark
        if missing_evaluations:
            status_entry["status"] = "incomplete_evaluation"
            status_entry["error"] = (
                "Missing or invalid evaluation outputs: "
                + ", ".join(missing_evaluations)
            )
            missing.append(task.task_id)
        else:
            rows.extend(task_rows)
            screen_rows.extend(task_screen_rows)
            benchmarks.update(task_benchmarks)
        status_rows.append(status_entry)

    control_dir = Path(plan["control_dir"])
    _write_csv(control_dir / "phase95_task_status.csv", pd.DataFrame(status_rows))
    frame = pd.DataFrame(rows)
    benchmark_error = None
    if len(benchmarks) > 1:
        benchmark_error = (
            "Corrected evaluation outputs contain nonidentical historic "
            f"benchmarks: {sorted(benchmarks)}."
        )
    elif not missing and not contract["skip_eval"] and len(benchmarks) != 1:
        benchmark_error = (
            "A complete evaluated plan must contain exactly one corrected "
            f"historic benchmark; found {len(benchmarks)}."
        )
    incomplete = bool(missing) or benchmark_error is not None
    qualifier = "_partial" if incomplete else ""
    metrics_path: Path | None = None
    summary_path: Path | None = None
    screen_output_path: Path | None = None
    benchmark_csv_path: Path | None = None
    benchmark_json_path: Path | None = None
    retired_outputs: list[str] = []
    if incomplete:
        for final_name in (
            "phase95_metrics.csv",
            "phase95_family_summary.csv",
            "phase95_objective_screen.csv",
            "corrected_historic_benchmark.csv",
            "corrected_historic_benchmark.json",
        ):
            retired = _retire_stale_output(control_dir / final_name)
            if retired is not None:
                retired_outputs.append(str(retired))
    if not contract["skip_eval"]:
        metrics_path = control_dir / f"phase95_metrics{qualifier}.csv"
        _write_csv(metrics_path, frame)
        screen_output_path = (
            control_dir / f"phase95_objective_screen{qualifier}.csv"
        )
        _write_csv(screen_output_path, pd.DataFrame(screen_rows))
    if not frame.empty:
        identity = {
            "task_id",
            "experiment",
            "seed",
            "status",
            "git_commit",
            "plan_id",
            "window",
        }
        numeric = [
            column
            for column in frame.columns
            if column not in identity and pd.api.types.is_numeric_dtype(frame[column])
        ]
        summary = frame.groupby(["experiment", "window"])[numeric].agg(["mean", "std"])
        summary.columns = [f"{column}_{stat}" for column, stat in summary.columns]
        summary_path = control_dir / f"phase95_family_summary{qualifier}.csv"
        _write_csv(summary_path, summary.reset_index())
    if not incomplete and not contract["skip_eval"]:
        benchmark_hash, benchmark = next(iter(benchmarks.items()))
        benchmark_json_path = control_dir / "corrected_historic_benchmark.json"
        _write_json(
            benchmark_json_path,
            {**benchmark, "benchmark_sha256": benchmark_hash},
        )
        benchmark_csv_path = control_dir / "corrected_historic_benchmark.csv"
        benchmark_rows = [
            {
                "metric": metric,
                "historic": benchmark["historic_metrics"][metric],
                "metric_profile": benchmark["metric_profile"],
                "window": benchmark["window"]["name"],
                "benchmark_sha256": benchmark_hash,
                "flood_comparison_profile": benchmark["flood_comparison"][
                    "profile"
                ],
                "flood_comparison_days": benchmark["flood_comparison"][
                    "comparison_days"
                ],
                "historic_flood_safe_days": benchmark["flood_comparison"][
                    "historic_safe_days"
                ],
            }
            for metric in OBJECTIVE_METRICS
        ]
        _write_csv(benchmark_csv_path, pd.DataFrame(benchmark_rows))
    result = {
        "collected_at": _utc_now(),
        "complete": not incomplete and benchmark_error is None,
        "plan_id": plan.get("plan_id"),
        "n_tasks": len(plan["tasks"]),
        "n_completed_tasks": sum(
            row.get("status") == "completed" for row in status_rows
        ),
        "n_rows": int(len(frame)),
        "missing_task_ids": sorted(set(missing)),
        "metrics_path": None if metrics_path is None else str(metrics_path),
        "family_summary_path": None if summary_path is None else str(summary_path),
        "objective_screen_path": (
            None if screen_output_path is None else str(screen_output_path)
        ),
        "historic_benchmark_csv_path": (
            None if benchmark_csv_path is None else str(benchmark_csv_path)
        ),
        "historic_benchmark_json_path": (
            None if benchmark_json_path is None else str(benchmark_json_path)
        ),
        "benchmark_error": benchmark_error,
        "task_status_path": str(control_dir / "phase95_task_status.csv"),
        "retired_stale_outputs": retired_outputs,
    }
    _write_json(control_dir / "collection.json", result)
    if result["missing_task_ids"]:
        raise RuntimeError(
            f"Incomplete or incompatible results for tasks {result['missing_task_ids']}."
        )
    if benchmark_error is not None:
        raise RuntimeError(benchmark_error)
    print(f"[phase95] collected {len(frame)} result rows in {control_dir}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="Validate a Phase 95 recovery spec")
    validate.add_argument("--spec", type=Path, required=True)

    plan = commands.add_parser("plan", help="Materialize task and Slurm plans")
    plan.add_argument("--spec", type=Path, required=True)
    plan.add_argument("--outdir", type=Path, required=True)
    plan.add_argument("--out-root", type=Path)
    plan.add_argument("--seed", dest="seeds", type=int, action="append")
    plan.add_argument("--experiment", dest="experiments", action="append")
    plan.add_argument("--timesteps", type=int)
    plan.add_argument("--skip-eval", action="store_true")
    plan.add_argument("--diagnostics", action="store_true")

    run_task_parser = commands.add_parser("run-task", help="Run one planned task")
    run_task_parser.add_argument("--plan", type=Path, required=True)
    run_task_parser.add_argument("--task-id", type=int, required=True)
    run_task_parser.add_argument("--retry-failed", action="store_true")

    run_pack_parser = commands.add_parser("run-pack", help="Run one packed task group")
    run_pack_parser.add_argument("--plan", type=Path, required=True)
    run_pack_parser.add_argument("--pack-id", type=int, required=True)
    run_pack_parser.add_argument("--retry-failed", action="store_true")

    collect = commands.add_parser("collect", help="Collect completed task results")
    collect.add_argument("--plan", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        spec = load_and_validate_spec(args.spec)
        print(
            f"[phase95] valid spec {args.spec}: "
            f"{len(spec['experiments'])} experiments x {len(spec['seeds'])} seeds"
        )
    elif args.command == "plan":
        materialize_plan(
            spec_path=args.spec,
            outdir=args.outdir,
            out_root=args.out_root,
            seeds=args.seeds,
            experiment_names=args.experiments,
            timesteps_override=args.timesteps,
            skip_eval=args.skip_eval,
            diagnostics=args.diagnostics,
        )
    elif args.command == "run-task":
        plan = _load_plan(args.plan)
        task = _task_from_plan(plan, args.task_id)
        contract = _plan_execution_contract(plan)
        run_task(
            spec_path=plan["spec_path"],
            task=task,
            retry_failed=args.retry_failed,
            **contract,
        )
    elif args.command == "run-pack":
        run_pack(
            plan_path=args.plan,
            pack_id=args.pack_id,
            retry_failed=args.retry_failed,
        )
    else:
        collect_results(args.plan)


if __name__ == "__main__":
    main()
