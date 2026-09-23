"""Custom PPO policy variants for DeepReservoir experiments."""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule


def _as_layer_list(value: object, *, default: Sequence[int] = (64, 64)) -> list[int]:
    if value is None:
        return [int(v) for v in default]
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(value)]


class _MaskedBranch(nn.Module):
    def __init__(
        self,
        *,
        feature_dim: int,
        indices: Sequence[int],
        net_arch: Sequence[int],
        activation_fn: type[nn.Module],
    ) -> None:
        super().__init__()
        if not indices:
            indices = list(range(feature_dim))
        idx = torch.as_tensor([int(i) for i in indices], dtype=torch.long)
        self.register_buffer("indices", idx, persistent=False)
        layers: list[nn.Module] = []
        last_dim = int(idx.numel())
        for layer_size in net_arch:
            size = int(layer_size)
            layers.append(nn.Linear(last_dim, size))
            layers.append(activation_fn())
            last_dim = size
        self.net = nn.Sequential(*layers) if layers else nn.Identity()
        self.output_dim = int(last_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = features.index_select(dim=1, index=self.indices)
        return self.net(x)


class _SplitActionMlpExtractor(nn.Module):
    """Actor/critic extractor with separate actor branches per action family."""

    def __init__(
        self,
        *,
        feature_dim: int,
        sj_indices: Sequence[int],
        niip_indices: Sequence[int],
        spr_indices: Sequence[int] | None = None,
        discretionary_sj_indices: Sequence[int] | None = None,
        hydropower_sj_indices: Sequence[int] | None = None,
        action_dim: int = 2,
        actor_arch: Sequence[int],
        vf_arch: Sequence[int],
        activation_fn: type[nn.Module],
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.sj_branch = _MaskedBranch(
            feature_dim=feature_dim,
            indices=sj_indices,
            net_arch=actor_arch,
            activation_fn=activation_fn,
        )
        self.niip_branch = _MaskedBranch(
            feature_dim=feature_dim,
            indices=niip_indices,
            net_arch=actor_arch,
            activation_fn=activation_fn,
        )
        self.spr_branch: _MaskedBranch | None = None
        if self.action_dim >= 3:
            self.spr_branch = _MaskedBranch(
                feature_dim=feature_dim,
                indices=list(spr_indices or range(feature_dim)),
                net_arch=actor_arch,
                activation_fn=activation_fn,
            )
        self.discretionary_sj_branch: _MaskedBranch | None = None
        if self.action_dim >= 4:
            self.discretionary_sj_branch = _MaskedBranch(
                feature_dim=feature_dim,
                indices=list(discretionary_sj_indices or sj_indices),
                net_arch=actor_arch,
                activation_fn=activation_fn,
            )
        self.hydropower_sj_branch: _MaskedBranch | None = None
        if self.action_dim >= 5:
            self.hydropower_sj_branch = _MaskedBranch(
                feature_dim=feature_dim,
                indices=list(hydropower_sj_indices or sj_indices),
                net_arch=actor_arch,
                activation_fn=activation_fn,
            )
        self.vf_branch = _MaskedBranch(
            feature_dim=feature_dim,
            indices=list(range(feature_dim)),
            net_arch=vf_arch,
            activation_fn=activation_fn,
        )
        self.latent_dim_sj = int(self.sj_branch.output_dim)
        self.latent_dim_niip = int(self.niip_branch.output_dim)
        self.latent_dim_spr = int(self.spr_branch.output_dim) if self.spr_branch is not None else 0
        self.latent_dim_discretionary_sj = (
            int(self.discretionary_sj_branch.output_dim)
            if self.discretionary_sj_branch is not None
            else 0
        )
        self.latent_dim_hydropower_sj = (
            int(self.hydropower_sj_branch.output_dim)
            if self.hydropower_sj_branch is not None
            else 0
        )
        self.latent_dim_pi = int(
            self.latent_dim_sj
            + self.latent_dim_niip
            + self.latent_dim_spr
            + self.latent_dim_discretionary_sj
            + self.latent_dim_hydropower_sj
        )
        self.latent_dim_vf = int(self.vf_branch.output_dim)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        parts = [self.sj_branch(features), self.niip_branch(features)]
        if self.spr_branch is not None:
            parts.append(self.spr_branch(features))
        if self.discretionary_sj_branch is not None:
            parts.append(self.discretionary_sj_branch(features))
        if self.hydropower_sj_branch is not None:
            parts.append(self.hydropower_sj_branch(features))
        return torch.cat(parts, dim=1)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.vf_branch(features)


class _SplitActionMeanNet(nn.Module):
    """Map split actor latents to continuous action means."""

    def __init__(
        self,
        *,
        latent_dim_sj: int,
        latent_dim_niip: int,
        latent_dim_spr: int = 0,
        latent_dim_discretionary_sj: int = 0,
        latent_dim_hydropower_sj: int = 0,
        action_dim: int = 2,
    ) -> None:
        super().__init__()
        self.latent_dim_sj = int(latent_dim_sj)
        self.latent_dim_niip = int(latent_dim_niip)
        self.latent_dim_spr = int(latent_dim_spr)
        self.latent_dim_discretionary_sj = int(latent_dim_discretionary_sj)
        self.latent_dim_hydropower_sj = int(latent_dim_hydropower_sj)
        self.action_dim = int(action_dim)
        self.sj_action = nn.Linear(self.latent_dim_sj, 1)
        self.niip_action = nn.Linear(self.latent_dim_niip, 1)
        self.spr_action = (
            nn.Linear(self.latent_dim_spr, 1)
            if self.action_dim >= 3 and self.latent_dim_spr > 0
            else None
        )
        self.discretionary_sj_action = (
            nn.Linear(self.latent_dim_discretionary_sj, 1)
            if self.action_dim >= 4 and self.latent_dim_discretionary_sj > 0
            else None
        )
        self.hydropower_sj_action = (
            nn.Linear(self.latent_dim_hydropower_sj, 1)
            if self.action_dim >= 5 and self.latent_dim_hydropower_sj > 0
            else None
        )

    def forward(self, latent_pi: torch.Tensor) -> torch.Tensor:
        sj_latent = latent_pi[:, : self.latent_dim_sj]
        niip_latent = latent_pi[:, self.latent_dim_sj : self.latent_dim_sj + self.latent_dim_niip]
        parts = [self.sj_action(sj_latent), self.niip_action(niip_latent)]
        if self.spr_action is not None:
            start = self.latent_dim_sj + self.latent_dim_niip
            spr_latent = latent_pi[:, start : start + self.latent_dim_spr]
            parts.append(self.spr_action(spr_latent))
        if self.discretionary_sj_action is not None:
            start = self.latent_dim_sj + self.latent_dim_niip + self.latent_dim_spr
            extra_latent = latent_pi[:, start : start + self.latent_dim_discretionary_sj]
            parts.append(self.discretionary_sj_action(extra_latent))
        if self.hydropower_sj_action is not None:
            start = (
                self.latent_dim_sj
                + self.latent_dim_niip
                + self.latent_dim_spr
                + self.latent_dim_discretionary_sj
            )
            hydro_latent = latent_pi[:, start : start + self.latent_dim_hydropower_sj]
            parts.append(self.hydropower_sj_action(hydro_latent))
        return torch.cat(parts, dim=1)


class SplitActionHeadsActorCriticPolicy(ActorCriticPolicy):
    """Single PPO policy with separate actor branches for SJ and NIIP actions.

    The critic still sees the full observation. The SJ actor branch sees all
    observations by default; the NIIP branch masks out SPR-specific observations
    so SPR request/context variables cannot directly drive the NIIP action mean.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        *args: Any,
        obs_column_names: Sequence[str] | None = None,
        sj_exclude_prefixes: Sequence[str] = (),
        sj_exclude_names: Sequence[str] = (),
        niip_exclude_prefixes: Sequence[str] = ("spr_",),
        niip_exclude_names: Sequence[str] = ("spill_avoidance_sj_frac", "animas_spr_frac"),
        **kwargs: Any,
    ) -> None:
        self.obs_column_names = [str(c) for c in (obs_column_names or [])]
        self.sj_exclude_prefixes = tuple(str(v) for v in sj_exclude_prefixes)
        self.sj_exclude_names = tuple(str(v) for v in sj_exclude_names)
        self.niip_exclude_prefixes = tuple(str(v) for v in niip_exclude_prefixes)
        self.niip_exclude_names = tuple(str(v) for v in niip_exclude_names)
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    def _branch_arches(self) -> tuple[list[int], list[int]]:
        net_arch = self.net_arch
        if isinstance(net_arch, dict):
            actor_arch = _as_layer_list(net_arch.get("pi"), default=(64, 64))
            vf_arch = _as_layer_list(net_arch.get("vf"), default=(64, 64))
        elif isinstance(net_arch, list):
            actor_arch = _as_layer_list(net_arch, default=(64, 64))
            vf_arch = _as_layer_list(net_arch, default=(64, 64))
        else:
            actor_arch = [64, 64]
            vf_arch = [64, 64]
        return actor_arch, vf_arch

    def _feature_indices(
        self,
        *,
        exclude_prefixes: Sequence[str],
        exclude_names: Sequence[str],
    ) -> list[int]:
        feature_dim = int(self.features_dim)
        if not self.obs_column_names or len(self.obs_column_names) != feature_dim:
            return list(range(feature_dim))
        indices: list[int] = []
        excluded_names = set(exclude_names)
        for i, name in enumerate(self.obs_column_names):
            if name in excluded_names:
                continue
            if any(name.startswith(prefix) for prefix in exclude_prefixes):
                continue
            indices.append(i)
        return indices or list(range(feature_dim))

    def _sj_feature_indices(self) -> list[int]:
        return self._feature_indices(
            exclude_prefixes=self.sj_exclude_prefixes,
            exclude_names=self.sj_exclude_names,
        )

    def _niip_feature_indices(self) -> list[int]:
        return self._feature_indices(
            exclude_prefixes=self.niip_exclude_prefixes,
            exclude_names=self.niip_exclude_names,
        )

    def _build_mlp_extractor(self) -> None:
        actor_arch, vf_arch = self._branch_arches()
        feature_dim = int(self.features_dim)
        action_dim = int(np.prod(self.action_space.shape))
        self.mlp_extractor = _SplitActionMlpExtractor(
            feature_dim=feature_dim,
            sj_indices=self._sj_feature_indices(),
            niip_indices=self._niip_feature_indices(),
            spr_indices=list(range(feature_dim)),
            discretionary_sj_indices=self._sj_feature_indices(),
            hydropower_sj_indices=self._sj_feature_indices(),
            action_dim=action_dim,
            actor_arch=actor_arch,
            vf_arch=vf_arch,
            activation_fn=self.activation_fn,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        if self.use_sde:
            raise NotImplementedError("SplitActionHeadsActorCriticPolicy does not support use_sde=True.")
        if not isinstance(self.action_dist, DiagGaussianDistribution):
            raise NotImplementedError(
                "SplitActionHeadsActorCriticPolicy currently supports continuous Box actions only."
            )
        action_dim = int(np.prod(self.action_space.shape))
        if action_dim not in {2, 3, 4, 5}:
            raise ValueError(
                "SplitActionHeadsActorCriticPolicy expects two continuous actions, "
                "three when using the SPR target proxy action mode, or four when "
                "using ESA/baseflow + SPR proxy action modes, or five when adding "
                "a hydropower SJ action branch."
            )

        self._build_mlp_extractor()
        _, self.log_std = self.action_dist.proba_distribution_net(
            latent_dim=self.mlp_extractor.latent_dim_pi,
            log_std_init=self.log_std_init,
        )
        self.action_net = _SplitActionMeanNet(
            latent_dim_sj=self.mlp_extractor.latent_dim_sj,
            latent_dim_niip=self.mlp_extractor.latent_dim_niip,
            latent_dim_spr=self.mlp_extractor.latent_dim_spr,
            latent_dim_discretionary_sj=self.mlp_extractor.latent_dim_discretionary_sj,
            latent_dim_hydropower_sj=self.mlp_extractor.latent_dim_hydropower_sj,
            action_dim=action_dim,
        )
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        self.optimizer = self.optimizer_class(  # type: ignore[call-arg]
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            {
                "obs_column_names": list(self.obs_column_names),
                "sj_exclude_prefixes": tuple(self.sj_exclude_prefixes),
                "sj_exclude_names": tuple(self.sj_exclude_names),
                "niip_exclude_prefixes": tuple(self.niip_exclude_prefixes),
                "niip_exclude_names": tuple(self.niip_exclude_names),
            }
        )
        return data
