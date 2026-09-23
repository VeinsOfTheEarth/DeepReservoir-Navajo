from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from deepreservoir.drl import phase95_recovery as recovery


SPEC_PATH = (
    recovery.REPO_ROOT
    / "config_files"
    / "reward_jon_phase95_corrected_recovery.json"
)
FAKE_GIT = {
    "commit": "0123456789abcdef",
    "branch": "phase95-recovery",
    "worktree_dirty": False,
}

EXPECTED_FAMILIES = (
    "reward_jon_p95_oishift875to90_hsoft",
    "reward_jon_p95_oishift875to90_heff",
    "reward_jon_p95_oishift875to90_hdisceff",
    "reward_jon_p95_peak875_heff",
    "reward_jon_p95_peak875_hdisceff",
    "reward_jon_p95_plateau90_96_hsoft",
    "reward_jon_p95_plateau90_96_heff",
    "reward_jon_p95_oishift875to90_heff_niip275",
)


def _materialize(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    **kwargs,
) -> tuple[dict, Path]:
    monkeypatch.setattr(recovery, "_git_state", lambda: dict(FAKE_GIT))
    control_dir = tmp_path / "plan"
    plan = recovery.materialize_plan(
        spec_path=SPEC_PATH,
        outdir=control_dir,
        out_root=tmp_path / "outputs",
        **kwargs,
    )
    return plan, control_dir / "plan.json"


def _write_completed_record(
    *,
    plan: dict,
    task: recovery.Phase95Task,
    plan_id: str | None = None,
) -> None:
    spec = recovery.load_and_validate_spec(plan["spec_path"])
    settings = recovery.resolve_train_settings(
        spec, recovery.experiment_by_name(spec, task.experiment)
    )
    requested, resolved = recovery._resolved_timesteps(
        settings, plan["timesteps_override"]
    )
    evaluations = []
    if not plan["skip_eval"]:
        evaluations = [{"name": "holdout_2014_2024_08_17"}]
    recovery._write_json(
        Path(task.outdir) / "phase95-task.json",
        {
            "status": "completed",
            "task": asdict(task),
            "spec_sha256": plan["spec_sha256"],
            "git": {"commit": plan["git"]["commit"]},
            "plan_id": plan["plan_id"] if plan_id is None else plan_id,
            "requested_timesteps": requested,
            "resolved_timesteps": resolved,
            "rich_training_diagnostics": plan["rich_training_diagnostics"],
            "evaluations": evaluations,
        },
    )


def _write_evaluation_artifacts(
    *,
    plan: dict,
    task: recovery.Phase95Task,
    historic_storage: float,
) -> None:
    window = {
        "name": "holdout_2014_2024_08_17",
        "start": "2014-01-01",
        "end": "2024-08-17",
    }
    eval_dir = Path(task.outdir) / f"eval__{window['name']}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"total_reward": 1.0, "flooding_frac_days_met": 0.25}]
    ).to_csv(
        eval_dir / "eval_metrics.csv", index=False
    )
    recovery._write_json(eval_dir / "recovery_diagnostics.json", {})

    historic = {metric: 1.0 for metric in recovery.OBJECTIVE_METRICS}
    historic["total_spill_af"] = 0.0
    historic["storage_frac_of_max_possible"] = historic_storage
    benchmark = {
        "schema_version": 1,
        "metric_profile": recovery.OBJECTIVE_SCREEN_PROFILE,
        "window": window,
        "historic_metrics": historic,
        "flood_comparison": {
            "profile": "archuleta_bluff_common_valid",
            "comparison_days": 3,
            "historic_available_days": 3,
            "historic_safe_days": 3,
        },
    }
    benchmark_hash = recovery._canonical_sha256(benchmark)
    policy = dict(historic)
    screen = {
        "schema_version": 1,
        "metric_profile": recovery.OBJECTIVE_SCREEN_PROFILE,
        "window": window,
        "spec_sha256": plan["spec_sha256"],
        "git_commit": plan["git"]["commit"],
        "policy_metrics": policy,
        "historic_benchmark": historic,
        "policy_minus_historic": {
            metric: 0.0 for metric in recovery.OBJECTIVE_METRICS
        },
        "criterion_margin": {
            metric: 0.0 for metric in recovery.OBJECTIVE_METRICS
        },
        "pass_flags": {
            metric: True for metric in recovery.OBJECTIVE_METRICS
        },
        "pass_count": 10,
        "objective_count": 10,
        "all_objectives_passed": True,
        "flood_comparison": {
            "flooding_comparison_profile": "archuleta_bluff_common_valid",
            "flooding_comparison_days": 3,
            "agent_flooding_available_days": 3,
            "historic_flooding_available_days": 3,
            "agent_flooding_safe_days": 3,
            "historic_flooding_safe_days": 3,
            "agent_flooding_frac_days_met": 1.0,
            "historic_flooding_frac_days_met": 1.0,
        },
        "benchmark": benchmark,
        "benchmark_sha256": benchmark_hash,
    }
    recovery._write_json(eval_dir / "corrected_objective_screen.json", screen)
    recovery._write_json(
        eval_dir / "corrected_historic_benchmark.json",
        {**benchmark, "benchmark_sha256": benchmark_hash},
    )


def test_spec_builds_seed_major_matched_tasks() -> None:
    spec = recovery.load_and_validate_spec(SPEC_PATH)
    tasks = recovery.build_tasks(spec)

    assert len(tasks) == 128
    assert [(task.seed, task.experiment) for task in tasks[:8]] == [
        (0, experiment["name"]) for experiment in spec["experiments"]
    ]
    assert tasks[8].seed == 1
    assert tuple(experiment["name"] for experiment in spec["experiments"]) == (
        EXPECTED_FAMILIES
    )
    assert spec["seeds"] == list(range(16))

    resolved = recovery.resolve_train_settings(spec, spec["experiments"][1])
    assert "spr_proxy_bridge_fraction" not in resolved
    assert resolved["decision_hydrology_timing"] == "previous_day"
    assert resolved["storage_datum_mode"] == "elevation_2019"
    assert resolved["storage_normalization"] == "train_window"
    assert resolved["niip_fallback_mode"] == "training_only"
    assert resolved["storage_budget_target_frac_of_max"] == pytest.approx(0.875)
    assert resolved["mask_incomplete_initial_spr"] is True
    assert "penalty_caps_archuleta_bluff" in resolved["reward_spec"]
    assert "histfreq_stop_excess_strong_calendar" in resolved["reward_spec"]


def test_missing_correction_is_rejected_by_spec_contract() -> None:
    spec = recovery.load_and_validate_spec(SPEC_PATH)
    settings = recovery.resolve_train_settings(spec, spec["experiments"][0])
    settings["decision_hydrology_timing"] = "same_day"

    with pytest.raises(recovery.Phase95SpecError, match="previous_day"):
        recovery.validate_corrected_settings(settings, context="invalid")


def test_corrected_objective_screen_counts_all_ten_criteria(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    historic = {metric: 0.5 for metric in recovery.OBJECTIVE_METRICS}
    historic["total_spill_af"] = 0.0
    policy = dict(historic)
    policy["hydropower_frac_of_max_possible"] = 0.49

    monkeypatch.setattr(
        recovery.drl_metrics,
        "compute_historic_summary_metrics",
        lambda rollout: dict(historic),
    )
    monkeypatch.setattr(
        recovery.drl_metrics,
        "compute_flooding_comparison_metrics",
        lambda rollout, profile: {
            "flooding_comparison_profile": profile,
            "flooding_comparison_days": 3,
            "agent_flooding_available_days": 3,
            "historic_flooding_available_days": 3,
            "agent_flooding_safe_days": 2,
            "historic_flooding_safe_days": 2,
            "agent_flooding_frac_days_met": 2.0 / 3.0,
            "historic_flooding_frac_days_met": 2.0 / 3.0,
        },
    )

    screen = recovery.build_corrected_objective_screen(
        pd.DataFrame(index=pd.date_range("2014-01-01", periods=3)),
        policy,
        window_name="holdout",
        window_start="2014-01-01",
        window_end="2014-01-03",
        spec_sha256="spec",
        git_commit="commit",
    )

    assert screen["objective_count"] == 10
    assert screen["pass_count"] == 9
    assert screen["all_objectives_passed"] is False
    assert screen["pass_flags"]["flooding_frac_days_met"] is True
    assert screen["pass_flags"]["hydropower_frac_of_max_possible"] is False
    assert len(screen["benchmark_sha256"]) == 64


def test_plan_freezes_filtered_smoke_contract_and_slurm_scripts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, plan_path = _materialize(
        tmp_path,
        monkeypatch,
        seeds=[0],
        timesteps_override=3_600,
        diagnostics=True,
    )

    assert len(plan["tasks"]) == 8
    assert plan["packed_tasks_per_node"] == 8
    assert plan["pack_count"] == 1
    assert plan["timesteps_override"] == 3_600
    assert plan["rich_training_diagnostics"] is True
    assert len(plan["plan_id"]) == 64
    assert recovery._load_plan(plan_path)["plan_id"] == plan["plan_id"]

    control_dir = plan_path.parent
    array = (control_dir / "batch_array.sbatch").read_text(encoding="utf-8")
    submit = (control_dir / "submit_batch.sh").read_text(encoding="utf-8")
    retry = (control_dir / "retry_pack.sbatch").read_text(encoding="utf-8")
    retry_submit = (control_dir / "retry_failed_pack.sh").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --array=0-0%1" in array
    assert "#SBATCH --cpus-per-task=8" in array
    assert "#SBATCH --no-requeue" in array
    assert "--dependency=afterany:${ARRAY_JOB_TOKEN}" in submit
    assert "submission.json" in submit and ".submission.lock" in submit
    assert "--retry-failed" in retry and "PACK_ID" in retry
    assert "--dependency=afterany:${JOB_TOKEN}" in retry_submit


def test_plan_digest_rejects_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, plan_path = _materialize(tmp_path, monkeypatch, seeds=[0])
    value = json.loads(plan_path.read_text(encoding="utf-8"))
    value["pack_count"] = 999
    plan_path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(recovery.Phase95SpecError, match="plan ID"):
        recovery._load_plan(plan_path)


def test_direct_task_execution_inherits_plan_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, plan_path = _materialize(
        tmp_path,
        monkeypatch,
        seeds=[0],
        experiment_names=["reward_jon_p95_peak875_hdisceff"],
        timesteps_override=3_600,
        skip_eval=True,
        diagnostics=True,
    )
    run_task = Mock()
    monkeypatch.setattr(recovery, "run_task", run_task)

    recovery.main(
        ["run-task", "--plan", str(plan_path), "--task-id", "0"]
    )

    options = run_task.call_args.kwargs
    assert options["timesteps_override"] == 3_600
    assert options["skip_eval"] is True
    assert options["diagnostics"] is True
    assert options["plan_id"] == plan["plan_id"]


def test_packed_children_resolve_execution_contract_from_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, plan_path = _materialize(
        tmp_path,
        monkeypatch,
        seeds=[0],
        experiment_names=["reward_jon_p95_peak875_hdisceff"],
        timesteps_override=3_600,
        skip_eval=True,
        diagnostics=True,
    )
    commands: list[list[str]] = []

    class FakeProcess:
        pid = 1234

        def wait(self) -> int:
            return 0

    def fake_popen(command, **kwargs):
        commands.append(command)
        return FakeProcess()

    monkeypatch.setattr(recovery.subprocess, "Popen", fake_popen)

    result = recovery.run_pack(plan_path=plan_path, pack_id=0)

    assert result["failures"] == []
    assert len(commands) == 1
    assert "--timesteps" not in commands[0]
    assert "--skip-eval" not in commands[0]
    assert "--diagnostics" not in commands[0]


def test_collector_rejects_stale_task_record_and_labels_partial_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, plan_path = _materialize(
        tmp_path,
        monkeypatch,
        seeds=[0],
        experiment_names=["reward_jon_p95_peak875_hdisceff"],
        timesteps_override=3_600,
    )
    task = recovery.Phase95Task(**plan["tasks"][0])
    _write_completed_record(plan=plan, task=task, plan_id="stale-plan")
    (plan_path.parent / "phase95_metrics.csv").write_text(
        "stale,final\n1,1\n", encoding="utf-8"
    )
    (plan_path.parent / "phase95_family_summary.csv").write_text(
        "stale,final\n1,1\n", encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="Incomplete or incompatible"):
        recovery.collect_results(plan_path)

    collection = json.loads(
        (plan_path.parent / "collection.json").read_text(encoding="utf-8")
    )
    status = pd.read_csv(plan_path.parent / "phase95_task_status.csv")
    assert collection["complete"] is False
    assert collection["metrics_path"].endswith("phase95_metrics_partial.csv")
    assert status.loc[0, "status"] == "incompatible"
    assert "plan ID" in status.loc[0, "error"]
    assert not (plan_path.parent / "phase95_metrics.csv").exists()
    assert not (plan_path.parent / "phase95_family_summary.csv").exists()
    assert len(collection["retired_stale_outputs"]) == 2


def test_collector_preserves_failed_status_and_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, plan_path = _materialize(
        tmp_path,
        monkeypatch,
        seeds=[0],
        experiment_names=["reward_jon_p95_peak875_hdisceff"],
        timesteps_override=3_600,
    )
    task = recovery.Phase95Task(**plan["tasks"][0])
    _write_completed_record(plan=plan, task=task)
    result_path = Path(task.outdir) / "phase95-task.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["status"] = "failed"
    result["error"] = "RuntimeError: simulated failure"
    recovery._write_json(result_path, result)

    with pytest.raises(RuntimeError, match="Incomplete or incompatible"):
        recovery.collect_results(plan_path)

    status = pd.read_csv(plan_path.parent / "phase95_task_status.csv")
    assert status.loc[0, "status"] == "failed"
    assert status.loc[0, "error"] == "RuntimeError: simulated failure"


def test_collector_reports_truncated_task_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, plan_path = _materialize(
        tmp_path,
        monkeypatch,
        seeds=[0],
        experiment_names=["reward_jon_p95_peak875_hdisceff"],
        timesteps_override=3_600,
    )
    task = recovery.Phase95Task(**plan["tasks"][0])
    result_path = Path(task.outdir) / "phase95-task.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text("{", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Incomplete or incompatible"):
        recovery.collect_results(plan_path)

    status = pd.read_csv(plan_path.parent / "phase95_task_status.csv")
    assert status.loc[0, "status"] == "invalid_record"
    assert "JSONDecodeError" in status.loc[0, "error"]


def test_collector_marks_missing_evaluation_artifacts_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, plan_path = _materialize(
        tmp_path,
        monkeypatch,
        seeds=[0],
        experiment_names=["reward_jon_p95_peak875_hdisceff"],
        timesteps_override=3_600,
    )
    task = recovery.Phase95Task(**plan["tasks"][0])
    _write_completed_record(plan=plan, task=task)

    with pytest.raises(RuntimeError, match="Incomplete or incompatible"):
        recovery.collect_results(plan_path)

    status = pd.read_csv(plan_path.parent / "phase95_task_status.csv")
    assert status.loc[0, "status"] == "incomplete_evaluation"
    assert "holdout_2014_2024_08_17" in status.loc[0, "error"]


def test_collector_rejects_nonidentical_corrected_benchmarks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, plan_path = _materialize(
        tmp_path,
        monkeypatch,
        seeds=[0, 1],
        experiment_names=["reward_jon_p95_peak875_hdisceff"],
        timesteps_override=3_600,
    )
    first = recovery.Phase95Task(**plan["tasks"][0])
    second = recovery.Phase95Task(**plan["tasks"][1])
    for task, storage in ((first, 0.68), (second, 0.69)):
        _write_completed_record(plan=plan, task=task)
        _write_evaluation_artifacts(
            plan=plan,
            task=task,
            historic_storage=storage,
        )

    with pytest.raises(RuntimeError, match="nonidentical historic benchmarks"):
        recovery.collect_results(plan_path)

    collection = json.loads(
        (plan_path.parent / "collection.json").read_text(encoding="utf-8")
    )
    assert collection["complete"] is False
    assert collection["benchmark_error"]


def test_collector_records_incomplete_screen_instead_of_crashing_on_key_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, plan_path = _materialize(
        tmp_path,
        monkeypatch,
        seeds=[0],
        experiment_names=["reward_jon_p95_peak875_hdisceff"],
        timesteps_override=3_600,
    )
    task = recovery.Phase95Task(**plan["tasks"][0])
    _write_completed_record(plan=plan, task=task)
    _write_evaluation_artifacts(
        plan=plan,
        task=task,
        historic_storage=0.68,
    )
    screen_path = (
        Path(task.outdir)
        / "eval__holdout_2014_2024_08_17"
        / "corrected_objective_screen.json"
    )
    screen = json.loads(screen_path.read_text(encoding="utf-8"))
    del screen["pass_flags"]
    recovery._write_json(screen_path, screen)

    with pytest.raises(RuntimeError, match="Incomplete or incompatible"):
        recovery.collect_results(plan_path)

    status = pd.read_csv(plan_path.parent / "phase95_task_status.csv")
    assert status.loc[0, "status"] == "incomplete_evaluation"


def test_collector_persists_one_benchmark_and_paired_flood_metric(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, plan_path = _materialize(
        tmp_path,
        monkeypatch,
        seeds=[0, 1],
        experiment_names=["reward_jon_p95_peak875_hdisceff"],
        timesteps_override=3_600,
    )
    for task_value in plan["tasks"]:
        task = recovery.Phase95Task(**task_value)
        _write_completed_record(plan=plan, task=task)
        _write_evaluation_artifacts(
            plan=plan,
            task=task,
            historic_storage=0.68,
        )

    collection = recovery.collect_results(plan_path)

    assert collection["complete"] is True
    assert collection["benchmark_error"] is None
    benchmark = pd.read_csv(collection["historic_benchmark_csv_path"])
    metrics = pd.read_csv(collection["metrics_path"])
    screen = pd.read_csv(collection["objective_screen_path"])
    assert len(benchmark) == 10
    assert len(set(benchmark["benchmark_sha256"])) == 1
    assert (metrics["flooding_frac_days_met_unpaired"] == 0.25).all()
    assert (metrics["flooding_frac_days_met"] == 1.0).all()
    assert (screen["objective_pass_count"] == 10).all()


def test_skip_eval_collection_succeeds_with_compatible_task_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, plan_path = _materialize(
        tmp_path,
        monkeypatch,
        seeds=[0],
        experiment_names=["reward_jon_p95_peak875_hdisceff"],
        timesteps_override=3_600,
        skip_eval=True,
    )
    task = recovery.Phase95Task(**plan["tasks"][0])
    _write_completed_record(plan=plan, task=task)

    collection = recovery.collect_results(plan_path)

    assert collection["complete"] is True
    assert collection["n_completed_tasks"] == 1
    assert collection["metrics_path"] is None
    assert collection["missing_task_ids"] == []


def test_retry_archives_failed_directory_and_task_lock_is_exclusive(
    tmp_path: Path,
) -> None:
    outdir = tmp_path / "experiment" / "seed_000"
    outdir.mkdir(parents=True)
    (outdir / "phase95-task.json").write_text(
        '{"status":"failed"}\n', encoding="utf-8"
    )

    archived = recovery._new_task_directory(outdir, retry_failed=True)

    assert archived is not None
    assert Path(archived, "phase95-task.json").is_file()
    assert outdir.is_dir() and not any(outdir.iterdir())

    with recovery._task_lock(outdir):
        contender = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                (
                    "import sys; from pathlib import Path; "
                    "from deepreservoir.drl.phase95_recovery import _task_lock; "
                    "\ntry:\n with _task_lock(Path(sys.argv[1])): pass"
                    "\nexcept RuntimeError:\n sys.exit(0)"
                    "\nsys.exit(3)"
                ),
                str(outdir),
            ],
            check=False,
        )
        assert contender.returncode == 0
    with recovery._task_lock(outdir):
        pass


def test_recovery_diagnostics_condition_spr_actions_on_spring_window() -> None:
    frame = pd.DataFrame(
        {
            "spr_proxy_window_active": [True, True, False, False],
            "spr_proxy_target_reachable_at_decision": [True, False, False, False],
            "spr_proxy_target_hit": [True, False, False, False],
            "spr_proxy_target_cfs": [10_000.0, 0.0, 5_000.0, 0.0],
            "spr_proxy_added_request_cfs": [1_000.0, 0.0, 0.0, 0.0],
            "spr_proxy_full_controller_need_cfs": [4_000.0, 0.0, 0.0, 0.0],
            "spr_proxy_controller_need_cfs": [3_000.0, 0.0, 0.0, 0.0],
            "action_1": [0.0, 0.0, 0.0, 0.0],
            "action_2": [1.0, -1.0, 0.0, -1.0],
            "action_3": [0.0, 0.0, 0.0, 0.0],
            "rc_esa_spring_peak_release.test": [2.0, -1.0, 0.0, 0.0],
            "rc_storage_control.test": [2.0, 1.0, 4.0, 4.0],
        }
    )

    diagnostics = recovery._recovery_diagnostics(frame)

    assert diagnostics["mean_action_spr_window"] == pytest.approx(0.0)
    assert diagnostics["spr_nonzero_target_fraction_window"] == pytest.approx(0.5)
    assert diagnostics["spr_nonzero_target_fraction_off_window"] == pytest.approx(0.5)
    assert diagnostics["spr_reward_abs_share_window"] == pytest.approx(0.5)
    assert diagnostics["spr_reward_abs_share_all_days"] == pytest.approx(3.0 / 14.0)
