from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from deepreservoir.drl import model


class _FakePolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features_extractor = nn.Flatten()
        self.mlp_extractor = nn.Sequential(
            nn.Linear(4, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.action_net = nn.Linear(64, 2)
        self.value_net = nn.Linear(64, 1)
        self.net_arch = {"pi": [64, 64], "vf": [64, 64]}
        self.activation_fn = nn.Tanh
        self.ortho_init = True
        self.share_features_extractor = True
        self.normalize_images = False
        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-4, eps=1e-5)


class _FakePPO:
    def __init__(self) -> None:
        self.policy = _FakePolicy()
        self.policy_class = _FakePolicy
        self.policy_kwargs = {}
        self.device = "cpu"
        self.n_steps = 2048
        self.batch_size = 64
        self.n_epochs = 10
        self.gamma = 0.99
        self.learning_rate = 3e-4
        self.gae_lambda = 0.95
        self.clip_range = 0.2
        self.clip_range_vf = None
        self.normalize_advantage = True
        self.ent_coef = 0.0
        self.vf_coef = 0.5
        self.max_grad_norm = 0.5
        self.target_kl = None
        self.use_sde = False
        self.sde_sample_freq = -1
        self.stats_window_size = 100


class ResolvedConfigTests(unittest.TestCase):
    def test_collect_resolved_sb3_config_includes_policy_summary(self) -> None:
        agent = _FakePPO()

        resolved = model._collect_resolved_sb3_config(agent)

        self.assertEqual(resolved["algo_class"], "_FakePPO")
        self.assertEqual(resolved["policy_class"], "_FakePolicy")
        self.assertEqual(resolved["batch_size"], 64)
        self.assertEqual(resolved["n_steps"], 2048)
        self.assertIn("policy_summary", resolved)
        self.assertEqual(resolved["policy_summary"]["net_arch"], {"pi": [64, 64], "vf": [64, 64]})
        self.assertEqual(resolved["policy_summary"]["activation_fn"], "Tanh")
        self.assertEqual(resolved["policy_summary"]["optimizer_class"], "Adam")

    def test_collect_resolved_config_snapshot_includes_module_details(self) -> None:
        agent = _FakePPO()

        snapshot = model._collect_resolved_config_snapshot(agent)

        self.assertIn("library_versions", snapshot)
        self.assertIn("agent", snapshot)
        self.assertIn("policy", snapshot)
        self.assertEqual(snapshot["policy"]["optimizer"]["class"], "Adam")
        self.assertIn("action_net", snapshot["policy"]["module_reprs"])
        self.assertIn("value_net", snapshot["policy"]["module_parameter_counts"])
        self.assertGreater(snapshot["policy"]["module_parameter_counts"]["value_net"]["total"], 0)

    def test_apply_resolved_agent_config_to_manifest_writes_sidecar(self) -> None:
        agent = _FakePPO()

        with tempfile.TemporaryDirectory() as tmpdir:
            logdir = Path(tmpdir)
            manifest = {
                "config": {},
                "train_invocations": [{}],
            }

            updated = model._apply_resolved_agent_config_to_manifest(
                manifest,
                logdir=logdir,
                agent=agent,
            )

            sidecar = logdir / "resolved_config.json"
            self.assertTrue(sidecar.exists())
            self.assertIn("sb3", updated)
            self.assertEqual(updated["config"]["batch_size"], 64)
            self.assertEqual(updated["config"]["n_steps"], 2048)
            self.assertIn("resolved_config", updated["artifacts"])
            self.assertIn("resolved_config_path", updated["sb3"])

            data = json.loads(sidecar.read_text())
            self.assertEqual(data["policy"]["optimizer"]["class"], "Adam")
            self.assertIn("policy_repr", data["policy"])


if __name__ == "__main__":
    unittest.main()
