from unittest.mock import patch
import pytest
from deepreservoir.drl import cli, selected_policy


def test_only_paper_commands_are_available():
    parser = cli.build_parser()
    for command in ("report-html", "report-metrics", "stress", "batch"):
        with pytest.raises(SystemExit):
            parser.parse_args([command])
    args = parser.parse_args(["eval", "--outdir", "runs/check"])
    assert args.model == selected_policy.SELECTED_MODEL_PATH
    args = parser.parse_args(["train", "--outdir", "runs/check"])
    assert args.seed == 13 and args.timesteps is None and args.torch_threads == 1


def test_config_supplies_environment_settings():
    cfg = selected_policy.load_selected_policy_config()
    options = cli._environment_options(cfg)
    assert len(options) == 17
    assert all(
        v == cfg["base_train"][k]
        for k, v in options.items()
        if k not in {
            "decision_hydrology_timing",
            "niip_fallback_mode",
            "storage_datum_mode",
            "storage_normalization",
            "storage_budget_target_frac_of_max",
            "mask_incomplete_initial_spr",
        }
    )
    assert options["spr_proxy_priority_release"] is True
    assert options["decision_hydrology_timing"] == "previous_day"
    assert options["niip_fallback_mode"] == "training_only"
    assert options["storage_datum_mode"] == "elevation_2019"
    assert options["storage_normalization"] == "train_window"
    assert options["storage_budget_target_frac_of_max"] == 0.875
    assert options["mask_incomplete_initial_spr"] is True


def test_corrected_environment_overrides_survive_cli_option_resolution():
    cfg = selected_policy.load_selected_policy_config()
    cfg["base_train"] = dict(cfg["base_train"])
    cfg["base_train"].update(
        {
            "niip_fallback_mode": "training_only",
            "storage_datum_mode": "elevation_2019",
            "storage_normalization": "train_window",
            "storage_budget_target_frac_of_max": 0.875,
            "mask_incomplete_initial_spr": True,
        }
    )

    options = cli._environment_options(cfg)

    assert options["niip_fallback_mode"] == "training_only"
    assert options["storage_datum_mode"] == "elevation_2019"
    assert options["storage_normalization"] == "train_window"
    assert options["storage_budget_target_frac_of_max"] == 0.875
    assert options["mask_incomplete_initial_spr"] is True


def test_nonempty_outputs_are_protected(tmp_path):
    sentinel = tmp_path / "existing.txt"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError):
        cli._new_output_directory(tmp_path)
    assert sentinel.read_text() == "keep"


def test_eval_uses_selected_dates_and_configuration(tmp_path):
    cfg = selected_policy.load_selected_policy_config()
    with patch("deepreservoir.drl.model.evaluate_model_window") as run:
        run.return_value = (None, type("Metrics", (), {"to_string": lambda *a, **k: ""})())
        cli.evaluate(selected_policy.SELECTED_MODEL_PATH, tmp_path / "eval", "cpu", cfg)
    options = run.call_args.kwargs
    assert options["window_start"] == "2014-01-01"
    assert options["window_end"] == "2024-08-17"
    assert options["reward_spec"] == selected_policy.SELECTED_REWARD_SPEC
    assert options["which_metrics"] == "all"
    assert options["decision_hydrology_timing"] == "previous_day"
    assert options["niip_fallback_mode"] == "training_only"
    assert options["storage_datum_mode"] == "elevation_2019"
    assert options["storage_normalization"] == "train_window"
    assert options["storage_budget_target_frac_of_max"] == 0.875
    assert options["mask_incomplete_initial_spr"] is True
    assert "save_plots" not in options and "run_stress_tests" not in options


def test_training_uses_complete_rollouts_and_selected_settings(tmp_path):
    args = cli.build_parser().parse_args([
        "train", "--outdir", str(tmp_path / "train"), "--timesteps", "3700",
        "--diagnostics", "--skip-eval",
    ])
    with patch("deepreservoir.drl.model.DRLModel") as constructor, patch("torch.set_num_threads"), patch("torch.set_num_interop_threads"):
        cli.train(args, selected_policy.load_selected_policy_config())
    assert constructor.call_args.kwargs["policy_type"] == selected_policy.SELECTED_POLICY_TYPE
    options = constructor.return_value.train.call_args.kwargs
    assert options["total_timesteps"] == options["n_steps"] == 3600
    assert options["batch_size"] == 60 and options["gamma"] == 0.999
    assert options["rich_training_diagnostics"] is True
    assert constructor.call_args.kwargs["decision_hydrology_timing"] == "previous_day"
    assert constructor.call_args.kwargs["niip_fallback_mode"] == "training_only"
    assert constructor.call_args.kwargs["storage_datum_mode"] == "elevation_2019"
    assert constructor.call_args.kwargs["storage_normalization"] == "train_window"
    assert constructor.call_args.kwargs["storage_budget_target_frac_of_max"] == 0.875
    assert constructor.call_args.kwargs["mask_incomplete_initial_spr"] is True
    assert (args.outdir / "paper-run.json").is_file()


def test_invalid_budget_does_not_create_output(tmp_path):
    args = cli.build_parser().parse_args([
        "train", "--outdir", str(tmp_path / "bad"), "--timesteps", "100",
    ])
    with pytest.raises(ValueError, match="3600-step"):
        cli.train(args, selected_policy.load_selected_policy_config())
    assert not args.outdir.exists()
