"""Run the archived Phase-95 baseline configuration and checkpoint.

Figure builders live under paper/; evaluation does not generate extra reports.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from deepreservoir.drl import selected_policy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("info", help="Show archived configuration and available dates")
    evaluate = commands.add_parser("eval", help="Evaluate the archived checkpoint with current code")
    evaluate.add_argument("--model", type=Path, default=selected_policy.SELECTED_MODEL_PATH)
    evaluate.add_argument("--outdir", type=Path, required=True)
    evaluate.add_argument("--device", default="cpu")
    train = commands.add_parser("train", help="Retrain the archived baseline policy configuration")
    train.add_argument("--outdir", type=Path, required=True)
    train.add_argument("--seed", type=int, default=selected_policy.SELECTED_LEGACY_SEED)
    train.add_argument("--timesteps", type=int, help="Short-run check; defaults to archived training budget")
    train.add_argument("--device", default="cpu")
    train.add_argument("--torch-threads", type=int, default=1)
    train.add_argument("--diagnostics", action="store_true", help="Log training-reward-signals data")
    train.add_argument("--skip-eval", action="store_true")
    return parser


def _new_output_directory(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a nonempty directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _environment_options(config: dict) -> dict:
    train = config["base_train"]
    options = {key: train[key] for key in (
        "obs_context", "action_scaling", "action_mode", "reward_balancing",
        "max_release_sj_main_cfs", "max_release_niip_cfs", "esa_min_flow_floor",
        "esa_baseflow_max_multiplier", "spr_proxy_priority_release",
        "spr_proxy_owns_sj_window", "spr_advice_mode",
    )}
    options["decision_hydrology_timing"] = train.get(
        "decision_hydrology_timing", "same_day"
    )
    options["niip_fallback_mode"] = train.get(
        "niip_fallback_mode", "legacy_full_series"
    )
    options["storage_datum_mode"] = train.get(
        "storage_datum_mode", "reported"
    )
    options["storage_normalization"] = train.get(
        "storage_normalization", "full_record"
    )
    options["storage_budget_target_frac_of_max"] = train.get(
        "storage_budget_target_frac_of_max", 0.780
    )
    options["mask_incomplete_initial_spr"] = train.get(
        "mask_incomplete_initial_spr", False
    )
    return options


def evaluate(model_path: Path, outdir: Path, device: str, config: dict) -> None:
    from deepreservoir.drl import model

    _new_output_directory(outdir)
    window = config["base_eval"]["windows"][0]
    _, metrics = model.evaluate_model_window(
        model_path=model_path,
        reward_spec=selected_policy.selected_experiment(config)["reward_spec"],
        window_start=window["start"], window_end=window["end"],
        outdir=outdir, device=device, which_metrics="all",
        **_environment_options(config),
    )
    print(metrics.to_string(index=False))
    print(f"Evaluation written to {outdir}")


def train(args: argparse.Namespace, config: dict) -> None:
    import torch
    from deepreservoir.drl import model

    if args.torch_threads < 1:
        raise ValueError("--torch-threads must be positive")
    settings = config["base_train"]
    episode_length = int(settings["episode_length"])
    requested = int(settings["train_timesteps"] if args.timesteps is None else args.timesteps)
    timesteps = requested // episode_length * episode_length
    if timesteps <= 0:
        raise ValueError(f"--timesteps must cover at least one {episode_length}-step rollout")
    if timesteps != requested:
        print(f"Using {timesteps} timesteps ({requested} requested), in full PPO rollouts")
    _new_output_directory(args.outdir)
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(args.torch_threads)
    run = model.DRLModel(
        reward_spec=selected_policy.selected_experiment(config)["reward_spec"],
        train_start=settings["train_start"], train_end=settings["train_end"],
        train_hydrology_transform=settings["train_hydrology_transform"],
        logdir=args.outdir, seed=args.seed, device=args.device,
        gamma=float(settings["gamma"]), policy_type=settings["policy_type"],
        policy_net_arch=settings["policy_net_arch"],
        episode_length_train=episode_length, launch_mode="paper_reproduction",
        **_environment_options(config),
    )
    run.train(
        n_episodes=timesteps // episode_length, total_timesteps=timesteps,
        n_steps=episode_length, batch_size=int(settings["batch_size"]), n_epochs=10,
        gamma=float(settings["gamma"]), policy_type=settings["policy_type"],
        policy_net_arch=settings["policy_net_arch"], track_reward_components=True,
        rich_training_diagnostics=args.diagnostics,
    )
    (args.outdir / "paper-run.json").write_text(json.dumps({
        "config": config, "seed": args.seed, "requested_timesteps": requested,
        "resolved_timesteps": timesteps, "torch_threads": args.torch_threads,
        "device": args.device, "rich_training_diagnostics": args.diagnostics,
    }, indent=2) + "\n", encoding="utf-8")
    if not args.skip_eval:
        evaluate(args.outdir / "last_model.zip", args.outdir / "eval", args.device, config)
    print(f"Training written to {args.outdir}")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = selected_policy.load_selected_policy_config()
    if args.command == "train":
        train(args, config)
    elif args.command == "eval":
        evaluate(args.model, args.outdir, args.device, config)
    else:
        from deepreservoir.drl.model import load_all_model_data

        raw = load_all_model_data()["raw"]
        print(f"Config: {selected_policy.SELECTED_CONFIG_PATH}")
        print(f"Daily data: {raw.index.min().date()} to {raw.index.max().date()}")
        print(f"Archived policy: {selected_policy.SELECTED_MODEL_PATH}")


if __name__ == "__main__":
    main()
