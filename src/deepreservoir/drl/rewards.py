# deepreservoir/drl/rewards.py

"""Reward components for the public Navajo Reservoir policy and recovery run.

This public-branch version intentionally keeps only the reward registry
plumbing, the variants required by the selected policy documented in
SELECTED_POLICY.md, the corrected flood-location variant, and the small set of
historical variants needed to repeat the complete Phase 95 seed search.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd

from deepreservoir.define_env.niip.niip_demand import niip_daily_demand
from deepreservoir.drl.niip_targets import historic_niip_delivery_target_for_date
from deepreservoir.define_env.storage_elevation.thresholds import (
    get_navajo_storage_thresholds,
)
from deepreservoir.define_env.spring_peak_release_curve import SpringPeakReleaseCurve

# ---------------------------------------------------------------------
# Public selected-policy constants
# ---------------------------------------------------------------------

REWARD_BALANCING_CHOICES: tuple[str, ...] = ("none",)

OBJECTIVES = [
    "dam_safety",
    "storage_control",
    "esa_min_flow",
    "esa_spring_peak_release",
    "flooding",
    "hydropower",
    "niip",
    "niip_low_demand",
    "release_sparsity",
]

# Filled by @register_reward below.
REWARD_REGISTRY: Dict[str, Dict[str, "RewardFn"]] = {obj: {} for obj in OBJECTIVES}

DAM_SAFETY_STORAGE_BOUNDS_VARIANTS = frozenset({"spill_guard_warn98"})
DAM_SAFETY_SPILL_VOLUME_VARIANTS = frozenset({"spill_guard_warn98"})
STORAGE_CONTROL_STORAGE_BOUNDS_VARIANTS = frozenset(
    {
        "target_peak875_concave0_softupper98",
        "target_oishift875to90_concave0_softupper98",
        "plateau90to96_concave0_softupper98",
    }
)
STORAGE_CONTROL_SPR_OI_VARIANTS = frozenset(
    {"target_oishift875to90_concave0_softupper98"}
)
RELEASE_SPARSITY_SPR_OI_VARIANTS = frozenset()
STORAGE_CONTROL_RELEASE_PLANNING_VARIANTS = frozenset()

PRACTICAL_MIN_STORAGE_AF = 500_000.0
HIST_HOLDOUT_PLUS5_STORAGE_FRAC_OF_MAX = 0.745
HIST_HOLDOUT_PLUS10_STORAGE_FRAC_OF_MAX = 0.780

CFS_TO_AF_PER_DAY = 86400.0 / 43560.0
AF_PER_DAY_TO_CFS = 1.0 / CFS_TO_AF_PER_DAY

SPR_THRESHOLD_DAY_SPECS: tuple[tuple[float, int], ...] = (
    (2_500.0, 10),
    (5_000.0, 21),
    (8_000.0, 10),
    (10_000.0, 5),
)
SPR_PHASE28_SELECTED_TARGET_ORDER: tuple[tuple[float, int], ...] = (
    (10_000.0, 5),
    (8_000.0, 10),
    (5_000.0, 21),
    (2_500.0, 10),
)
SPR_PHASE28_SELECTED_ARCHITECTURES: tuple[str, ...] = ()
SPR_PHASE38_TRAINING_FREQ_THRESHOLD_MULTIPLIERS: dict[str, dict[int, float]] = {
    "ledgerfreq": {
        10_000: 7.017543859649123,
        8_000: 2.105263157894737,
        5_000: 0.6349206349206349,
        2_500: 1.0,
    }
}
SPR_PHASE44_REQUEST_MARGIN_CFS: dict[str, dict[int, float]] = {
    "high_clearance": {10_000: 500.0, 8_000: 250.0, 5_000: 100.0, 2_500: 0.0},
    "high_clearance_wide": {10_000: 750.0, 8_000: 400.0, 5_000: 150.0, 2_500: 0.0},
}
SPR_TARGET_FREQUENCIES: dict[int, float] = {
    10_000: 0.20,
    8_000: 0.33,
    5_000: 0.50,
    2_500: 0.80,
}
SPR_HISTORIC_TARGET_FREQUENCIES: dict[int, float] = {
    10_000: 0.091,
    8_000: 0.182,
    5_000: 0.273,
    2_500: 0.727,
}
SPR_PHASE68_FREQ_PRIOR_YEARS = 10.0
SPR_THRESHOLD_TRACKING_VARIANTS = frozenset(
    {
        "farmington_thresholds_actionproxy_smart_ledger_hammer_histfreq_stop_excess_strong",
        "farmington_thresholds_actionproxy_smart_ledger_hammer_histfreq_stop_excess_strong_calendar",
    }
)

_SPRING_PEAK_CURVE = SpringPeakReleaseCurve()

@dataclass
class RewardContext:
    """
    Information needed to compute reward for a single step.

    Notes on alpha
    --------------
    Each reward component can be assigned an ``alpha`` via the reward spec.
    The CompositeReward sets ``ctx.alpha`` (and ctx.objective/ctx.variant)
    before calling each component function.

    Convention (for now): reward functions should return an *unscaled* value
    and CompositeReward applies the alpha multiplier.
    """
    t: int                       # step index (global idx)
    date: pd.Timestamp           # current date
    obs: np.ndarray              # current observation (normalized)
    action: np.ndarray           # current action (raw [-1,1])
    next_obs: np.ndarray         # next observation (normalized)
    info: Dict[str, Any]         # extra per-step info (raw series, flags, etc.)

    # Set by CompositeReward before calling each component reward function.
    objective: str | None = None
    variant: str | None = None
    alpha: float = 1.0

RewardFn = Callable[[RewardContext], float]

def normalize_reward_balancing(mode: str | None) -> str:
    key = str(mode or "none").strip().lower()
    if key not in REWARD_BALANCING_CHOICES:
        raise ValueError(
            f"Unknown reward_balancing {mode!r}. "
            f"Choose from: {sorted(REWARD_BALANCING_CHOICES)}"
        )
    return key

def _default_navajo_max_storage_af() -> float:
    return get_navajo_storage_thresholds().max_storage_af

def _animas_farmington_cfs(ctx: RewardContext) -> float:
    """Return the Animas-at-Farmington flow from the lean scalar field when
    present, falling back to the legacy raw_forcings container.
    """
    if "animas_farmington_q_cfs" in ctx.info:
        return float(ctx.info.get("animas_farmington_q_cfs", 0.0))

    row = ctx.info.get("raw_forcings", None)
    if row is None:
        return 0.0

    getter = getattr(row, "get", None)
    if getter is None:
        return 0.0

    try:
        return float(getter("animas_farmington_q_cfs", 0.0))
    except Exception:
        return 0.0


def _decision_animas_farmington_cfs(ctx: RewardContext) -> float:
    """Return the Animas value that was available when the action was chosen."""
    if "decision_animas_farmington_q_cfs" in ctx.info:
        return float(ctx.info["decision_animas_farmington_q_cfs"])
    return _animas_farmington_cfs(ctx)

def _soft_min_threshold_score(
    *,
    value: float,
    threshold: float,
    zero_score_value: float,
    min_value: float,
) -> float:
    """Soft score for objectives that require staying above a threshold.

    Returns:
      - +1 at or above ``threshold``
      -  0 at ``zero_score_value``
      - -1 at or below ``min_value``
      - linear interpolation in between
    """
    if threshold <= zero_score_value:
        return 0.0
    if zero_score_value <= min_value:
        return 0.0

    if value >= threshold:
        return 1.0
    if value >= zero_score_value:
        return float((value - zero_score_value) / (threshold - zero_score_value))
    if value <= min_value:
        return -1.0
    return float(-((zero_score_value - value) / (zero_score_value - min_value)))

def _soft_max_threshold_score(
    *,
    value: float,
    full_score_value: float,
    threshold: float,
    min_score_value: float,
) -> float:
    """Soft score for objectives that require staying below a threshold.

    Returns:
      - +1 at or below ``full_score_value``
      -  0 at ``threshold``
      - -1 at or above ``min_score_value``
      - linear interpolation in between
    """
    if threshold <= full_score_value:
        return 0.0
    if min_score_value <= threshold:
        return 0.0

    if value <= full_score_value:
        return 1.0
    if value <= threshold:
        return float(1.0 - ((value - full_score_value) / (threshold - full_score_value)))
    if value >= min_score_value:
        return -1.0
    return float(-((value - threshold) / (min_score_value - threshold)))

def register_reward(objective: str, variant: str) -> Callable[[RewardFn], RewardFn]:
    """
    Decorator to register a reward function under an objective + variant name.
    """
    if objective not in REWARD_REGISTRY:
        raise KeyError(f"Unknown objective {objective!r}. Expected one of {OBJECTIVES}.")

    def decorator(fn: RewardFn) -> RewardFn:
        if variant in REWARD_REGISTRY[objective]:
            raise ValueError(f"Reward already registered for {objective}:{variant}")
        REWARD_REGISTRY[objective][variant] = fn
        return fn

    return decorator

@dataclass
class RewardComponent:
    objective: str
    variant: str
    alpha: float
    weight: float
    fn: RewardFn

    @property
    def key(self) -> str:
        return f"{self.objective}.{self.variant}"

@dataclass
class ActiveNormTracker:
    mean_abs: float | None = None
    active_updates: int = 0
    pos_mean_abs: float | None = None
    neg_mean_abs: float | None = None
    pos_updates: int = 0
    neg_updates: int = 0

@dataclass(frozen=True)
class ActiveNormConfig:
    ema_beta: float = 0.01
    active_threshold: float = 1e-8
    min_scale: float = 1e-3
    clip_normalized: float = 5.0
    spr_positive_clip: float = 20.0
    freeze_after_active_updates: int = 256

@dataclass
class RewardBalancerState:
    mode: str = "none"
    trackers: dict[str, ActiveNormTracker] = field(default_factory=dict)

class CompositeReward:
    """
    Combines multiple RewardComponents into a single scalar reward.

    Call with RewardContext; returns (total_reward, component_breakdown_dict).
    """

    def __init__(
        self,
        components: List[RewardComponent],
        *,
        reward_balancing: str = "none",
        update_balancing: bool = True,
        active_norm_cfg: ActiveNormConfig | None = None,
    ):
        self.components = components
        self.reward_balancing = normalize_reward_balancing(reward_balancing)
        self.update_balancing = bool(update_balancing)
        self.active_norm_cfg = active_norm_cfg or ActiveNormConfig()
        self._balancer_state = RewardBalancerState(mode=self.reward_balancing)

    def __call__(self, ctx: RewardContext) -> Tuple[float, Dict[str, float]]:
        total = 0.0
        breakdown: Dict[str, float] = {}

        for comp in self.components:
            # Provide component context to the reward function.
            ctx.objective = comp.objective
            ctx.variant = comp.variant
            ctx.alpha = float(comp.alpha)

            base = float(comp.fn(ctx))
            base = self._apply_reward_balancing(comp, base, ctx)
            # For now, alpha is a simple multiplier applied outside the reward
            # function (reward functions may *read* ctx.alpha in the future).
            r = float(comp.alpha) * base
            weighted = comp.weight * r
            breakdown[comp.key] = float(weighted)
            total += weighted

        return float(total), breakdown

    @staticmethod
    def _is_spr_component(component: RewardComponent) -> bool:
        return str(component.objective) == "esa_spring_peak_release"

    def _spr_opportunity_weight(self, ctx: RewardContext) -> float:
        if not bool(ctx.info.get("spr_window_active", False)):
            return 0.0

        oi_val = ctx.info.get("spring_oi", np.nan)
        try:
            oi = float(oi_val)
        except Exception:
            oi = np.nan
        if np.isfinite(oi):
            return float(np.clip(oi, 0.0, 1.0))

        if bool(ctx.info.get("spring_go", False)):
            return 1.0
        return 0.0

    def _ema_update(self, current: float | None, value: float, *, beta: float) -> float:
        if current is None:
            return float(value)
        return float((1.0 - beta) * float(current) + beta * float(value))

    def _apply_reward_balancing(
        self,
        component: RewardComponent,
        base_reward: float,
        ctx: RewardContext,
    ) -> float:
        if self.reward_balancing == "none":
            return float(base_reward)

        cfg = self.active_norm_cfg
        base = float(base_reward)
        if not np.isfinite(base):
            return 0.0

        is_spr = self._is_spr_component(component)
        if self.reward_balancing in {
            "active_norm_dense_only",
            "active_norm_dense_only_freeze_warmup",
        } and is_spr:
            return float(base)

        active = abs(base) > float(cfg.active_threshold)
        tracker = self._balancer_state.trackers.get(component.key)
        if tracker is None:
            tracker = ActiveNormTracker()
            self._balancer_state.trackers[component.key] = tracker

        update_allowed = bool(self.update_balancing)
        if (
            self.reward_balancing
            in {
                "active_norm_freeze_warmup",
                "active_norm_dense_only_freeze_warmup",
            }
            and tracker.active_updates >= int(cfg.freeze_after_active_updates)
        ):
            update_allowed = False

        if self.reward_balancing in {"contrib_norm", "contrib_norm_freeze_warmup"}:
            if (
                self.reward_balancing == "contrib_norm_freeze_warmup"
                and tracker.active_updates >= int(cfg.freeze_after_active_updates)
            ):
                update_allowed = False

            if update_allowed:
                tracker.mean_abs = self._ema_update(
                    tracker.mean_abs,
                    abs(base),
                    beta=float(np.clip(cfg.ema_beta, 0.0, 1.0)),
                )
                tracker.active_updates += 1

            scale = tracker.mean_abs
            if scale is None:
                return float(base)

            scale = max(float(scale), float(cfg.min_scale))
            normalized = float(base / scale)
            clip_val = float(cfg.clip_normalized)
            return float(np.clip(normalized, -clip_val, clip_val))

        update_weight = 1.0
        if self.reward_balancing == "active_norm_spr_opportunity" and is_spr:
            update_weight = self._spr_opportunity_weight(ctx)
            active = bool(active and update_weight > 0.0)

        if active and update_allowed:
            abs_base = max(abs(base), float(cfg.min_scale))
            beta = float(cfg.ema_beta) * float(update_weight)
            beta = float(np.clip(beta, 0.0, 1.0))
            tracker.mean_abs = self._ema_update(
                tracker.mean_abs,
                abs_base,
                beta=beta,
            )
            tracker.active_updates += 1

            if self.reward_balancing == "active_norm_spr_asymmetric" and is_spr:
                if base > 0.0:
                    tracker.pos_mean_abs = self._ema_update(
                        tracker.pos_mean_abs,
                        abs_base,
                        beta=beta,
                    )
                    tracker.pos_updates += 1
                elif base < 0.0:
                    tracker.neg_mean_abs = self._ema_update(
                        tracker.neg_mean_abs,
                        abs_base,
                        beta=beta,
                    )
                    tracker.neg_updates += 1

        scale = tracker.mean_abs
        if self.reward_balancing == "active_norm_spr_asymmetric" and is_spr:
            if base > float(cfg.active_threshold):
                scale = tracker.pos_mean_abs if tracker.pos_mean_abs is not None else scale
            elif base < -float(cfg.active_threshold):
                scale = tracker.neg_mean_abs if tracker.neg_mean_abs is not None else scale
        if scale is None:
            # Eval/report envs may not have a trained balancer state; in that case
            # preserve the raw reward rather than inventing a scale.
            return float(base)

        scale = max(float(scale), float(cfg.min_scale))
        normalized = float(base / scale)
        clip_val = float(cfg.clip_normalized)
        if self.reward_balancing == "active_norm_spr_loose_positive_clip" and is_spr:
            return float(
                np.clip(
                    normalized,
                    -clip_val,
                    float(cfg.spr_positive_clip),
                )
            )
        return float(np.clip(normalized, -clip_val, clip_val))

    def get_balancer_state(self) -> dict[str, object]:
        return {
            "mode": self.reward_balancing,
            "update_balancing": bool(self.update_balancing),
            "active_norm": {
                "ema_beta": float(self.active_norm_cfg.ema_beta),
                "active_threshold": float(self.active_norm_cfg.active_threshold),
                "min_scale": float(self.active_norm_cfg.min_scale),
                "clip_normalized": float(self.active_norm_cfg.clip_normalized),
                "spr_positive_clip": float(self.active_norm_cfg.spr_positive_clip),
                "freeze_after_active_updates": int(
                    self.active_norm_cfg.freeze_after_active_updates
                ),
            },
            "trackers": {
                key: {
                    "mean_abs": (
                        None if tracker.mean_abs is None else float(tracker.mean_abs)
                    ),
                    "active_updates": int(tracker.active_updates),
                    "pos_mean_abs": (
                        None if tracker.pos_mean_abs is None else float(tracker.pos_mean_abs)
                    ),
                    "neg_mean_abs": (
                        None if tracker.neg_mean_abs is None else float(tracker.neg_mean_abs)
                    ),
                    "pos_updates": int(tracker.pos_updates),
                    "neg_updates": int(tracker.neg_updates),
                }
                for key, tracker in sorted(self._balancer_state.trackers.items())
            },
        }

def build_composite_reward(
    spec: Mapping[str, Union[str, "ObjectiveSpec"]],
    weights: Optional[Dict[str, float]] = None,
    *,
    reward_balancing: str = "none",
    update_balancing: bool = True,
) -> CompositeReward:
    """
    spec: mapping objective -> variant
    weights: optional mapping objective -> scalar weight
    """
    components: List[RewardComponent] = []
    weights = weights or {}

    for objective, sel in spec.items():
        if isinstance(sel, ObjectiveSpec):
            variant = sel.variant
            alpha = float(sel.alpha)
        else:
            variant = str(sel)
            alpha = 1.0

        if objective not in REWARD_REGISTRY:
            raise KeyError(
                f"Unknown objective {objective!r}. Known: {list(REWARD_REGISTRY.keys())}"
            )
        variants = REWARD_REGISTRY[objective]
        if variant not in variants:
            raise KeyError(
                f"Unknown variant {variant!r} for objective {objective!r}. "
                f"Known: {list(variants.keys())}"
            )

        fn = variants[variant]
        w = float(weights.get(objective, 1.0))
        components.append(RewardComponent(objective, variant, alpha, w, fn))

    return CompositeReward(
        components,
        reward_balancing=reward_balancing,
        update_balancing=update_balancing,
    )

@dataclass(frozen=True)
class ObjectiveSpec:
    """Parsed objective selection from reward_spec.

    Attributes
    ----------
    variant
        The registered reward variant name, e.g. "baseline".
    alpha
        A per-objective parameter (currently used as a multiplier). Defaults to 1.
    """

    variant: str
    alpha: float = 1.0

def parse_objective_spec(spec_str: str) -> Dict[str, ObjectiveSpec]:
    """
    Format
    ------
    Comma-separated tokens. Each token may be:

      - "objective"                      -> variant="baseline", alpha=1
      - "objective:variant"              -> alpha=1
      - "objective:variant@alpha"        -> alpha parsed as float

    Examples
    --------
      "dam_safety:storage_band@0.5,esa_min_flow:baseline,hydropower"
    """
    spec: Dict[str, ObjectiveSpec] = {}
    if not spec_str:
        return spec

    pairs = [s.strip() for s in spec_str.split(",") if s.strip()]
    for token in pairs:
        # Allow "objective" (implies baseline)
        if ":" not in token:
            obj = token.strip()
            if not obj:
                continue
            spec[obj] = ObjectiveSpec("baseline", 1.0)
            continue

        obj, rest = [p.strip() for p in token.split(":", 1)]
        if not obj:
            continue

        alpha = 1.0
        variant = rest

        # Variant may include @alpha (preferred)
        if "@" in rest:
            variant, alpha_str = [p.strip() for p in rest.split("@", 1)]
            try:
                alpha = float(alpha_str)
            except Exception:
                alpha = 1.0
        else:
            # If '@' is not present, alpha defaults to 1.0.
            # For robustness, tolerate stray extra ':' segments by taking the first as the variant.
            variant = rest.split(":", 1)[0].strip()

        if not variant:
            variant = "baseline"

        spec[obj] = ObjectiveSpec(variant, alpha)
    return spec

def _dam_safety_spill_guard(
    ctx: RewardContext,
    *,
    spill_warn_frac_of_max: float,
    spill_ref_cfs: float = 2_500.0,
    max_spill_penalty: float = 5.0,
) -> float:
    """Spill-focused dam-safety objective.

    This intentionally strips out the old "keep storage high" behavior. A safe
    storage state receives zero reward/penalty. Storage inside the near-spill
    warning band receives an increasing negative pressure, and actual
    uncontrolled spill receives a larger volume-sensitive penalty.
    """
    storage = float(ctx.info["storage_af"])
    s_max = float(ctx.info.get("max_storage_af", _default_navajo_max_storage_af()))
    if s_max <= 0.0:
        return 0.0

    spill_cfs = max(float(ctx.info.get("spill_cfs", 0.0)), 0.0)
    if spill_cfs > 0.0:
        ref = max(float(spill_ref_cfs), 1.0)
        return float(-1.0 - min(spill_cfs / ref, float(max_spill_penalty)))

    spill_warn_frac_of_max = float(np.clip(spill_warn_frac_of_max, 0.0, 1.0))
    spill_warn = float(spill_warn_frac_of_max * s_max)
    if storage <= spill_warn:
        return 0.0

    span = max(s_max - spill_warn, 1.0)
    x = float(np.clip((storage - spill_warn) / span, 0.0, 1.0))
    return float(-(x**2))

def _storage_control_peak_target_no_low_plateau(
    ctx: RewardContext,
    *,
    target_frac_of_max: float,
    upper_warn_frac_of_max: float,
    reward_at_warn: float,
    low_recovery_shape: str,
    reward_at_full: float = -1.0,
) -> float:
    """Peak storage target without a flat low-storage penalty pit.

    The original peak-target storage rewards reuse the practical-minimum
    storage anchor, which makes every state below that storage equally bad.
    These variants retain the same 85% target and high-storage warning side,
    but give low-storage states a differentiable ordering all the way to zero.
    """
    storage = float(ctx.info["storage_af"])
    s_max = float(ctx.info.get("max_storage_af", 1.0))
    if s_max <= 0.0:
        return 0.0

    target_frac = float(np.clip(target_frac_of_max, 0.0, 1.0))
    warn_frac = float(np.clip(upper_warn_frac_of_max, target_frac, 1.0))
    target = max(float(target_frac * s_max), 1.0)
    warn = float(np.clip(warn_frac * s_max, target, s_max))

    if storage <= target:
        x = float(np.clip(storage / target, 0.0, 1.0))
        shape = str(low_recovery_shape).strip().lower()
        if shape in {"concave", "sqrt"}:
            x = float(np.sqrt(x))
        elif shape in {"linear", "lin"}:
            pass
        else:
            raise ValueError(f"Unknown low_recovery_shape {low_recovery_shape!r}.")
        return float(np.clip(-1.0 + 2.0 * x, -1.0, 1.0))

    if storage <= warn:
        span = warn - target
        if span <= 0.0:
            return float(reward_at_warn)
        frac = float(np.clip((storage - target) / span, 0.0, 1.0))
        return float(1.0 + (float(reward_at_warn) - 1.0) * frac)

    span = max(s_max - warn, 1.0)
    frac = float(np.clip((storage - warn) / span, 0.0, 1.0))
    reward = float(reward_at_warn) + (float(reward_at_full) - float(reward_at_warn)) * (frac**2)
    return float(np.clip(reward, -1.5, 1.0))

@register_reward("dam_safety", "spill_guard_warn98")
def dam_safety_spill_guard_warn98(ctx: RewardContext) -> float:
    """Spill-only safety penalty with warning pressure starting at 98% full."""
    return _dam_safety_spill_guard(ctx, spill_warn_frac_of_max=0.98)

def _storage_control_target_peak_concave0_softupper98(
    ctx: RewardContext,
    *,
    target_frac_of_max: float,
) -> float:
    return _storage_control_peak_target_no_low_plateau(
        ctx,
        target_frac_of_max=target_frac_of_max,
        upper_warn_frac_of_max=0.98,
        reward_at_warn=0.5,
        low_recovery_shape="concave",
    )

@register_reward("storage_control", "target_peak875_concave0_softupper98")
def storage_control_target_peak875_concave0_softupper98(ctx: RewardContext) -> float:
    """Peak at 87.5% full with concave no-plateau recovery from zero storage."""
    return _storage_control_target_peak_concave0_softupper98(
        ctx,
        target_frac_of_max=0.875,
    )


def _storage_control_useful_drawdown_credit(
    ctx: RewardContext,
    *,
    weight: float,
) -> float:
    """Offset storage pressure when drawdown serves a named objective."""

    weight = float(weight)
    if weight <= 0.0:
        return 0.0

    prev_storage = float(
        ctx.info.get("prev_storage_af", ctx.info.get("storage_af", 0.0))
    )
    storage = float(ctx.info.get("storage_af", prev_storage))
    storage_loss_af = max(prev_storage - storage, 0.0)
    if storage_loss_af <= 0.0:
        return 0.0

    useful_cfs = max(float(ctx.info.get("release_niip_cfs", 0.0)), 0.0)
    useful_cfs += max(
        float(ctx.info.get("spr_attributed_sj_release_cfs", 0.0)), 0.0
    )
    useful_cfs += max(
        float(ctx.info.get("spr_useful_unrequested_sj_release_cfs", 0.0)),
        0.0,
    )
    useful_cfs += max(
        float(ctx.info.get("esa_attributed_sj_release_cfs", 0.0)), 0.0
    )
    useful_af = max(useful_cfs, 0.0) * CFS_TO_AF_PER_DAY
    if useful_af <= 0.0:
        return 0.0

    max_total_release_cfs = float(
        ctx.info.get("max_release_sj_main_cfs", 5_000.0)
    ) + float(ctx.info.get("max_release_niip_cfs", 2_500.0))
    norm_af = max(max_total_release_cfs * CFS_TO_AF_PER_DAY, 1.0)
    credited_af = min(storage_loss_af, useful_af)
    return float(np.clip(weight * credited_af / norm_af, 0.0, 0.5))


def _storage_control_plateau_target(
    ctx: RewardContext,
    *,
    lower_frac_of_max: float,
    upper_frac_of_max: float,
    upper_warn_frac_of_max: float = 0.98,
    reward_at_warn: float = 0.5,
    reward_at_full: float = -1.0,
    useful_drawdown_discount: float = 0.0,
) -> float:
    """Storage-control objective with a broad high-storage plateau."""

    storage = float(ctx.info["storage_af"])
    s_max = float(ctx.info.get("max_storage_af", 1.0))
    if s_max <= 0.0:
        return 0.0

    lower_frac = float(np.clip(lower_frac_of_max, 0.0, 1.0))
    upper_frac = float(np.clip(upper_frac_of_max, lower_frac, 1.0))
    warn_frac = float(np.clip(upper_warn_frac_of_max, upper_frac, 1.0))
    lower = max(lower_frac * s_max, 1.0)
    upper = float(np.clip(upper_frac * s_max, lower, s_max))
    warn = float(np.clip(warn_frac * s_max, upper, s_max))

    if storage <= lower:
        x = float(np.clip(storage / lower, 0.0, 1.0))
        reward = -1.0 + 2.0 * float(np.sqrt(x))
    elif storage <= upper:
        reward = 1.0
    elif storage <= warn:
        span = max(warn - upper, 1.0)
        frac = float(np.clip((storage - upper) / span, 0.0, 1.0))
        reward = 1.0 + (float(reward_at_warn) - 1.0) * frac
    else:
        span = max(s_max - warn, 1.0)
        frac = float(np.clip((storage - warn) / span, 0.0, 1.0))
        reward = float(reward_at_warn) + (
            float(reward_at_full) - float(reward_at_warn)
        ) * (frac**2)

    reward += _storage_control_useful_drawdown_credit(
        ctx,
        weight=float(useful_drawdown_discount),
    )
    return float(np.clip(reward, -1.5, 1.25))


@register_reward("storage_control", "plateau90to96_concave0_softupper98")
def storage_control_plateau90to96_concave0_softupper98(
    ctx: RewardContext,
) -> float:
    """Full credit from 90--96% full, preserving a flexible high band."""

    return _storage_control_plateau_target(
        ctx,
        lower_frac_of_max=0.90,
        upper_frac_of_max=0.96,
    )


def _storage_control_oi_shifted_peak_target(
    ctx: RewardContext,
    *,
    wet_target_frac_of_max: float,
    scarce_target_frac_of_max: float,
) -> float:
    oi_raw = ctx.info.get("spring_oi", np.nan)
    try:
        oi = float(oi_raw)
    except Exception:
        oi = np.nan
    if not np.isfinite(oi):
        oi = 1.0
    scarcity = 1.0 - float(np.clip(oi, 0.0, 1.0))
    target = float(wet_target_frac_of_max) + (
        float(scarce_target_frac_of_max) - float(wet_target_frac_of_max)
    ) * scarcity
    return _storage_control_target_peak_concave0_softupper98(
        ctx,
        target_frac_of_max=float(np.clip(target, 0.0, 0.98)),
    )


@register_reward("storage_control", "target_oishift875to90_concave0_softupper98")
def storage_control_target_oishift875to90_concave0_softupper98(
    ctx: RewardContext,
) -> float:
    """Shift the peak target from 87.5% in wet years to 90% in scarce years."""

    return _storage_control_oi_shifted_peak_target(
        ctx,
        wet_target_frac_of_max=0.875,
        scarce_target_frac_of_max=0.90,
    )

def _esa_min_flow_mainstem_need_cfs(
    ctx: RewardContext,
    *,
    target_cfs: float = 500.0,
) -> float:
    max_release = max(float(ctx.info.get("max_release_sj_main_cfs", 5_000.0)), 0.0)
    animas_cfs = _animas_farmington_cfs(ctx)
    return float(np.clip(float(target_cfs) - animas_cfs, 0.0, max_release))

def _esa_min_flow_control_delta_cfs(ctx: RewardContext) -> float | None:
    """Return agent-controlled mainstem release surplus over ESA need.

    Days where the Animas already meets the 500 cfs target return ``None`` so
    these shaping rewards focus on controllable low-flow-release behavior rather
    than paying for exogenous Animas flow.
    """
    need_cfs = _esa_min_flow_mainstem_need_cfs(ctx)
    if need_cfs <= 1.0e-9:
        return None
    release_cfs = max(float(ctx.info.get("release_sj_main_cfs", 0.0)), 0.0)
    return float(release_cfs - need_cfs)

def _esa_green_logistic_reward(delta_cfs: float) -> float:
    z = (float(delta_cfs) + 290.0) / 80.0
    reward = -1.65 + 2.65 / (1.0 + np.exp(-z))
    return float(np.clip(reward, -1.65, 1.0))

@register_reward("esa_min_flow", "green_logistic_jon")
def esa_min_flow_green_logistic_jon(ctx: RewardContext) -> float:
    """Smooth broad-gradient ESA minimum-flow reward for controllable deficit days."""
    delta_cfs = _esa_min_flow_control_delta_cfs(ctx)
    if delta_cfs is None:
        return 0.0
    return _esa_green_logistic_reward(delta_cfs)

def _flooding_penalty_caps(q0: float, qlag2: object) -> float:
    if q0 <= 5_000.0:
        c1 = 0.0
    elif q0 >= 7_000.0:
        c1 = -1.0
    else:
        c1 = -((q0 - 5_000.0) / (7_000.0 - 5_000.0))

    if qlag2 is None or not np.isfinite(qlag2) or float(qlag2) <= 12_000.0:
        c2 = 0.0
    elif float(qlag2) >= 16_000.0:
        c2 = -1.0
    else:
        c2 = -((float(qlag2) - 12_000.0) / (16_000.0 - 12_000.0))

    return float(0.5 * (c1 + c2))


@register_reward("flooding", "penalty_caps_jon")
def flooding_penalty_caps_jon(ctx: RewardContext) -> float:
    """Legacy Phase-95 guardrail using Farmington for the 5,000-cfs term.

    This variant is retained because the archived selected policy was trained
    with it. New training intended to represent the operating criteria should
    use ``penalty_caps_archuleta_bluff``.
    """
    q0 = float(ctx.info.get("sj_at_farmington_cfs", 0.0))
    qlag2 = ctx.info.get("sj_at_farmington_lag2_cfs", None)
    return _flooding_penalty_caps(q0, qlag2)


@register_reward("flooding", "penalty_caps_archuleta_bluff")
def flooding_penalty_caps_archuleta_bluff(ctx: RewardContext) -> float:
    """Guardrail using the Archuleta and two-day-lagged Bluff proxies."""
    q0 = float(
        ctx.info.get(
            "sj_at_archuleta_proxy_cfs",
            ctx.info.get("sj_main_flow_cfs", 0.0),
        )
    )
    qlag2 = ctx.info.get(
        "sj_at_bluff_proxy_cfs",
        ctx.info.get("sj_at_farmington_lag2_cfs", None),
    )
    return _flooding_penalty_caps(q0, qlag2)

def _hydropower_positive_fraction(ctx: RewardContext) -> float:
    hydropower_mwh = float(ctx.info.get("hydropower_mwh", 0.0))
    return float(np.clip(hydropower_mwh / 768.0, 0.0, 1.0))


@register_reward("hydropower", "positive_soft")
def hydropower_positive_soft(ctx: RewardContext) -> float:
    """Smooth nonnegative generation reward without a low-output cliff."""

    return _hydropower_positive_fraction(ctx)


def _release_sj_main_cfs(ctx: RewardContext) -> float:
    return float(max(float(ctx.info.get("release_sj_main_cfs", 0.0)), 0.0))


@register_reward("hydropower", "positive_efficiency")
def hydropower_positive_efficiency(ctx: RewardContext) -> float:
    """Reward hydropower generation per acre-foot, softly gated by generation."""

    hydropower_mwh = float(ctx.info.get("hydropower_mwh", 0.0))
    release_cfs = _release_sj_main_cfs(ctx)
    if hydropower_mwh <= 0.0 or release_cfs <= 0.0:
        return 0.0

    release_af = max(release_cfs * CFS_TO_AF_PER_DAY, 1e-6)
    reference_efficiency = 768.0 / max(
        1_300.0 * CFS_TO_AF_PER_DAY, 1e-6
    )
    efficiency_score = np.clip(
        (hydropower_mwh / release_af) / reference_efficiency,
        0.0,
        1.0,
    )
    generation_gate = np.sqrt(_hydropower_positive_fraction(ctx))
    return float(
        np.clip(float(efficiency_score) * float(generation_gate), 0.0, 1.0)
    )

def _discretionary_sj_release_cfs(ctx: RewardContext) -> float:
    return float(
        max(
            float(
                ctx.info.get(
                    "discretionary_attributed_sj_release_cfs",
                    ctx.info.get("release_beyond_esa_spr_need_cfs", 0.0),
                )
            ),
            0.0,
        )
    )

def _turbine_band_efficiency_factor(release_cfs: float, *, efficient_cfs: float = 1_300.0) -> float:
    release_cfs = max(float(release_cfs), 0.0)
    if release_cfs <= efficient_cfs:
        return 1.0
    return float(np.sqrt(max(float(efficient_cfs), 1.0) / max(release_cfs, 1.0)))

@register_reward("hydropower", "positive_discretionary_efficiency")
def hydropower_positive_discretionary_efficiency(ctx: RewardContext) -> float:
    """Positive hydropower reward with efficiency shaping only for discretionary SJ release."""
    base = _hydropower_positive_fraction(ctx)
    discretionary = _discretionary_sj_release_cfs(ctx)
    if discretionary <= 0.0:
        return float(base)
    factor = _turbine_band_efficiency_factor(discretionary, efficient_cfs=1_300.0)
    return float(np.clip(base * factor, 0.0, 1.0))

def _niip_historic_day_demand_and_release(
    ctx: RewardContext,
    *,
    active_doy_start: int | None = None,
    active_doy_end: int | None = None,
    min_demand_cfs: float = 0.0,
) -> tuple[float, float] | None:
    """Return historic-delivery target and NIIP release for active target days."""
    if active_doy_start is not None or active_doy_end is not None:
        doy = int(pd.to_datetime(ctx.date).timetuple().tm_yday)
        if active_doy_start is not None and doy < int(active_doy_start):
            return None
        if active_doy_end is not None and doy > int(active_doy_end):
            return None

    # Environments precompute the target with their configured fallback table.
    # The scalar helper intentionally remains the legacy fallback for archived
    # direct reward calls that do not carry an environment-generated target.
    demand_from_env = ctx.info.get("niip_demand_cfs")
    if demand_from_env is None or not np.isfinite(float(demand_from_env)):
        demand_cfs = float(
            historic_niip_delivery_target_for_date(pd.Timestamp(ctx.date))
        )
    else:
        demand_cfs = float(demand_from_env)
    if demand_cfs <= 0.0 or demand_cfs < max(0.0, float(min_demand_cfs)):
        return None
    niip_release_cfs = float(ctx.info.get("release_niip_cfs", 0.0))
    return demand_cfs, niip_release_cfs

def _niip_no_target_release_penalty(
    ctx: RewardContext,
    *,
    min_reward: float = -1.0,
) -> float:
    """Penalize NIIP release on days with no requested NIIP delivery.

    Zero target should mean "do not release through NIIP", not indifference.
    The penalty is scaled so roughly 10% of the NIIP outlet cap is a -1 raw
    reward, then clips at ``min_reward`` for variants with a deeper floor.
    This makes large no-demand dumps visible to the learner without
    overreacting to tiny numerical releases.
    """
    release_cfs = max(float(ctx.info.get("release_niip_cfs", 0.0)), 0.0)
    if release_cfs <= 1e-6:
        return 0.0
    cap_cfs = max(float(ctx.info.get("max_release_niip_cfs", 2_500.0)), 1e-9)
    zero_score_release_cfs = max(50.0, 0.10 * cap_cfs)
    reward = -(release_cfs / zero_score_release_cfs)
    return float(np.clip(reward, float(min_reward), 0.0))

def _niip_no_target_release_penalty_log(
    ctx: RewardContext,
    *,
    weight: float,
    ref_frac_of_cap: float = 0.10,
) -> float:
    """Continuous no-target NIIP release penalty with no clipped flat floor."""
    release_cfs = max(float(ctx.info.get("release_niip_cfs", 0.0)), 0.0)
    if release_cfs <= 1e-6:
        return 0.0
    cap_cfs = max(float(ctx.info.get("max_release_niip_cfs", 2_500.0)), 1e-9)
    ref_cfs = max(25.0, float(ref_frac_of_cap) * cap_cfs)
    penalty = float(weight) * np.log1p(release_cfs / ref_cfs)
    return float(-penalty)

def _niip_historic_hardmeet_asym_reward(
    ctx: RewardContext,
    *,
    under_zero_frac: float,
    under_zero_cfs: float,
    over_full_frac: float,
    over_full_cfs: float,
    over_zero_frac: float,
    over_zero_cfs: float,
    min_reward: float = -1.0,
    no_target_penalty: str = "clipped",
    no_target_weight: float = 1.0,
    active_doy_start: int | None = None,
    active_doy_end: int | None = None,
    min_demand_cfs: float = 0.0,
    low_demand_threshold_cfs: float | None = None,
    low_under_zero_frac: float | None = None,
    low_under_zero_cfs: float | None = None,
    low_over_full_frac: float | None = None,
    low_over_full_cfs: float | None = None,
    low_over_zero_frac: float | None = None,
    low_over_zero_cfs: float | None = None,
) -> float:
    """Hard-meet historic NIIP target with asymmetric overdelivery tolerance.

    Below the daily target, the score is never positive. At or above target it
    becomes positive, with a forgiving plateau for moderate overdelivery.
    """
    active = _niip_historic_day_demand_and_release(
        ctx,
        active_doy_start=active_doy_start,
        active_doy_end=active_doy_end,
        min_demand_cfs=min_demand_cfs,
    )
    if active is None:
        if str(no_target_penalty).strip().lower() in {"log", "continuous"}:
            return _niip_no_target_release_penalty_log(ctx, weight=no_target_weight)
        return _niip_no_target_release_penalty(ctx, min_reward=min_reward)

    demand_cfs, niip_release_cfs = active
    demand_cfs = max(float(demand_cfs), 1e-9)
    niip_release_cfs = float(niip_release_cfs)
    error_cfs = niip_release_cfs - demand_cfs

    if error_cfs < 0.0:
        under_cfs = abs(error_cfs)
        use_low_demand = (
            low_demand_threshold_cfs is not None
            and demand_cfs <= float(low_demand_threshold_cfs)
        )
        if use_low_demand:
            zero_frac = (
                float(low_under_zero_frac)
                if low_under_zero_frac is not None
                else float(under_zero_frac)
            )
            zero_cfs = (
                float(low_under_zero_cfs)
                if low_under_zero_cfs is not None
                else float(under_zero_cfs)
            )
        else:
            zero_frac = float(under_zero_frac)
            zero_cfs = float(under_zero_cfs)
        zero_score_error_cfs = max(zero_cfs, zero_frac * demand_cfs)
        reward = -(under_cfs / max(zero_score_error_cfs, 1e-9))
    else:
        over_cfs = error_cfs
        use_low_demand = (
            low_demand_threshold_cfs is not None
            and demand_cfs <= float(low_demand_threshold_cfs)
        )
        if use_low_demand:
            full_frac = (
                float(low_over_full_frac)
                if low_over_full_frac is not None
                else float(over_full_frac)
            )
            full_cfs = (
                float(low_over_full_cfs)
                if low_over_full_cfs is not None
                else float(over_full_cfs)
            )
            zero_frac = (
                float(low_over_zero_frac)
                if low_over_zero_frac is not None
                else float(over_zero_frac)
            )
            zero_cfs = (
                float(low_over_zero_cfs)
                if low_over_zero_cfs is not None
                else float(over_zero_cfs)
            )
        else:
            full_frac = float(over_full_frac)
            full_cfs = float(over_full_cfs)
            zero_frac = float(over_zero_frac)
            zero_cfs = float(over_zero_cfs)
        full_credit_cfs = max(full_cfs, full_frac * demand_cfs)
        zero_score_error_cfs = max(zero_cfs, zero_frac * demand_cfs)
        if zero_score_error_cfs <= full_credit_cfs:
            return 1.0
        if over_cfs <= full_credit_cfs:
            reward = 1.0
        else:
            reward = 1.0 - (
                (over_cfs - full_credit_cfs)
                / max(zero_score_error_cfs - full_credit_cfs, 1e-9)
            )

    return float(np.clip(reward, float(min_reward), 1.0))

@register_reward("niip", "delivery_match_historic_hardmeet_b10oversoft_neg1_logoff025_jon")
def niip_delivery_match_historic_hardmeet_b10oversoft_neg1_logoff025_jon(ctx: RewardContext) -> float:
    """Band10-oversoft with active penalties clipped at -1 and 0.25x log no-target penalty."""
    return _niip_historic_hardmeet_asym_reward(
        ctx,
        under_zero_frac=0.45,
        under_zero_cfs=175.0,
        over_full_frac=0.10,
        over_full_cfs=50.0,
        over_zero_frac=0.45,
        over_zero_cfs=250.0,
        min_reward=-1.0,
        no_target_penalty="log",
        no_target_weight=0.25,
    )

def _spring_window_target_cfs(date: pd.Timestamp) -> float:
    return float(_SPRING_PEAK_CURVE.target_cfs_from_date(pd.to_datetime(date)))

def _spr_threshold_label(threshold_cfs: float, duration_days: int) -> str:
    return f"{int(threshold_cfs)}cfs_{int(duration_days)}d"

def _spr_days_so_far(
    ctx: RewardContext,
    *,
    threshold_cfs: float,
    duration_days: int,
) -> int:
    key = f"spr_days_so_far_{_spr_threshold_label(threshold_cfs, duration_days)}"
    try:
        return int(ctx.info.get(key, 0))
    except Exception:
        return 0

def _spr_need_more(
    ctx: RewardContext,
    *,
    threshold_cfs: float,
    duration_days: int,
) -> bool:
    key = f"spr_need_more_{_spr_threshold_label(threshold_cfs, duration_days)}"
    value = ctx.info.get(key, None)
    if value is None:
        return _spr_days_so_far(
            ctx,
            threshold_cfs=threshold_cfs,
            duration_days=duration_days,
        ) < int(duration_days)
    return bool(value)

@dataclass(frozen=True)
class SPR10KBridgeTerms:
    target_cfs: float
    duration_days: int
    animas_cfs: float
    decision_animas_cfs: float
    release_sj_main_cfs: float
    max_release_sj_main_cfs: float
    farmington_total_cfs: float
    max_reachable_farmington_cfs: float
    bridge_need_cfs: float
    spring_oi: float
    spring_go: bool
    days_so_far: int
    need_more: bool

def _spr_threshold_bridge_terms(
    ctx: RewardContext,
    *,
    threshold_cfs: float,
    duration_days: int,
) -> SPR10KBridgeTerms | None:
    target = _spring_window_target_cfs(ctx.date)
    if target <= 0.0:
        return None

    target_cfs = float(threshold_cfs)
    duration_days = int(duration_days)

    animas = _animas_farmington_cfs(ctx)
    decision_animas = _decision_animas_farmington_cfs(ctx)
    release = float(ctx.info.get("release_sj_main_cfs", 0.0))
    max_release = float(ctx.info.get("max_release_sj_main_cfs", 5_000.0))
    max_release = max(max_release, 0.0)
    farmington = float(animas + release)
    max_reachable = float(decision_animas + max_release)
    bridge_need = float(np.clip(target_cfs - decision_animas, 0.0, max_release))

    oi = ctx.info.get("spring_oi", np.nan)
    oi_val = float(oi) if oi is not None and np.isfinite(oi) else float("nan")
    go_val = bool(ctx.info.get("spring_go", False))
    days_so_far = _spr_days_so_far(
        ctx,
        threshold_cfs=target_cfs,
        duration_days=duration_days,
    )
    need_more = _spr_need_more(
        ctx,
        threshold_cfs=target_cfs,
        duration_days=duration_days,
    )

    return SPR10KBridgeTerms(
        target_cfs=target_cfs,
        duration_days=duration_days,
        animas_cfs=float(animas),
        decision_animas_cfs=float(decision_animas),
        release_sj_main_cfs=float(release),
        max_release_sj_main_cfs=float(max_release),
        farmington_total_cfs=float(farmington),
        max_reachable_farmington_cfs=float(max_reachable),
        bridge_need_cfs=float(bridge_need),
        spring_oi=oi_val,
        spring_go=go_val,
        days_so_far=int(days_so_far),
        need_more=bool(need_more),
    )

def _spr_progress_increment(
    terms: SPR10KBridgeTerms,
    *,
    threshold_cfs: float = 10_000.0,
    duration_days: int = 5,
) -> float:
    duration = max(int(duration_days), 1)
    prev_progress = min(max(float(terms.days_so_far), 0.0) / float(duration), 1.0)
    earned_today = 1.0 if terms.need_more and terms.farmington_total_cfs >= float(threshold_cfs) else 0.0
    curr_progress = min(
        max(float(terms.days_so_far) + float(earned_today), 0.0) / float(duration),
        1.0,
    )
    return float(curr_progress - prev_progress)

def _spr_calendar_days_left(ctx: RewardContext) -> int:
    """Use the same inclusive, available-row count supplied to SPR advice."""
    if "spr_calendar_days_left" in ctx.info:
        return max(int(ctx.info["spr_calendar_days_left"]), 0)

    # Standalone reward contexts may omit environment step metadata.
    date = pd.Timestamp(ctx.date).normalize()
    if _spring_window_target_cfs(date) <= 0.0:
        return 0
    end_month, end_day, _ = _SPRING_PEAK_CURVE.cfg.points_md_cfs[-1]
    end_date = pd.Timestamp(year=date.year, month=end_month, day=end_day)
    return max(int((end_date - date).days) + 1, 0)

def _spr_calendar_completion_possible(ctx: RewardContext, terms: SPR10KBridgeTerms) -> bool:
    remaining = int(terms.duration_days) - int(terms.days_so_far)
    return remaining > 0 and remaining <= _spr_calendar_days_left(ctx)

def _spr_thresholds_daily_reachable_selected_terms(
    ctx: RewardContext,
    *,
    respect_ledger: bool = True,
    frequency_gate_mode: str = "none",
    target_frequencies: Mapping[int, float] | None = None,
    gate_10k: bool = False,
    annual_gate_mode: str = "none",
    respect_calendar: bool = False,
) -> SPR10KBridgeTerms | None:
    """Select the highest threshold physically reachable today.

    Unlike ``_spr_thresholds_needmatch_selected_terms``, this deliberately does
    not consult threshold OI / spring-go fields. This is meant for Phase 47
    diagnostics where the SPR request is a daily physical opportunity label:
    if Animas plus max controlled SJ release can clear 10k, ask for 10k;
    otherwise try 8k, then 5k, then 2.5k.
    """
    for threshold_cfs, duration_days in SPR_PHASE28_SELECTED_TARGET_ORDER:
        terms = _spr_threshold_bridge_terms(
            ctx,
            threshold_cfs=float(threshold_cfs),
            duration_days=int(duration_days),
        )
        if terms is None:
            return None
        if bool(respect_ledger) and not bool(terms.need_more):
            continue
        if bool(respect_calendar) and not _spr_calendar_completion_possible(ctx, terms):
            continue
        if terms.max_reachable_farmington_cfs < float(threshold_cfs):
            continue
        if not _spr_actionproxy_annual_gate_allows(
            ctx,
            terms,
            mode=annual_gate_mode,
        ):
            continue
        if (
            str(frequency_gate_mode or "none").strip().lower() in {"hard", "hardfreq"}
            and _spr_frequency_gate_multiplier(
                ctx,
                terms,
                mode="hard",
                target_frequencies=target_frequencies,
                gate_10k=gate_10k,
            )
            <= 0.0
        ):
            continue
        return terms
    return None

def _spr_frequency_gate_multiplier(
    ctx: RewardContext,
    terms: SPR10KBridgeTerms,
    *,
    mode: str = "none",
    target_frequencies: Mapping[int, float] | None = None,
    gate_10k: bool = False,
) -> float:
    """Episode-local SPR frequency gate.

    The selected hard gate uses historic frequency parameters for all four
    thresholds (including 10k). The symmetric pseudo-prior cancels in both
    hard and soft modes. The hard rule is
    ``successes < frequency * (completed_years + 1)``.
    Other variants may leave 10k ungated.
    """
    mode_key = str(mode or "none").strip().lower()
    if mode_key in {"", "none", "off"}:
        return 1.0

    threshold_i = int(round(float(terms.target_cfs)))
    if threshold_i >= 10_000 and not bool(gate_10k):
        return 1.0

    freq_map = target_frequencies if target_frequencies is not None else SPR_TARGET_FREQUENCIES
    target_freq = float(freq_map.get(threshold_i, 1.0))
    if target_freq <= 0.0:
        return 1.0

    label = _spr_threshold_label(float(terms.target_cfs), int(terms.duration_days))
    try:
        completed_years = max(int(ctx.info.get("spr_completed_years_so_far", 0)), 0)
    except Exception:
        completed_years = 0
    try:
        successes = max(int(ctx.info.get(f"spr_success_years_so_far_{label}", 0)), 0)
    except Exception:
        successes = 0

    # This symmetric prior cancels from the deficit for both gate modes.
    prior_years = float(SPR_PHASE68_FREQ_PRIOR_YEARS)
    prior_successes = prior_years * target_freq
    desired_through_current = target_freq * (prior_years + float(completed_years) + 1.0)
    pseudo_successes = prior_successes + float(successes)
    deficit = float(desired_through_current - pseudo_successes)

    if mode_key in {"hard", "hardfreq"}:
        return 1.0 if deficit > 0.0 else 0.0

    if mode_key in {"soft", "softfreq"}:
        # If behind schedule, keep full credit. If already ahead, leave a small
        # learning signal but make extra lower-threshold success much quieter.
        return float(np.clip(0.20 + 0.80 * (deficit / target_freq), 0.20, 1.0))

    raise ValueError(f"Unknown SPR frequency gate mode {mode!r}")

def _spr_actionproxy_annual_gate_allows(
    ctx: RewardContext,
    terms: SPR10KBridgeTerms,
    *,
    mode: str = "none",
) -> bool:
    mode_key = str(mode or "none").strip().lower()
    if mode_key in {"", "none", "off"}:
        return True
    if mode_key in {"oi", "oihard", "hard_oi", "oi_hard"}:
        return bool(
            _spr_threshold_annual_go(
                ctx,
                float(terms.target_cfs),
                int(terms.duration_days),
            )
        )
    raise ValueError(f"Unknown SPR annual gate mode {mode!r}")

def _spr_actionproxy_highest_justified_terms(
    ctx: RewardContext,
    *,
    frequency_gate_mode: str = "hard",
    target_frequencies: Mapping[int, float] | None = None,
    gate_10k: bool = False,
    annual_gate_mode: str = "none",
    respect_calendar: bool = False,
) -> SPR10KBridgeTerms | None:
    return _spr_thresholds_daily_reachable_selected_terms(
        ctx,
        respect_ledger=True,
        frequency_gate_mode=frequency_gate_mode,
        target_frequencies=target_frequencies,
        gate_10k=gate_10k,
        annual_gate_mode=annual_gate_mode,
        respect_calendar=respect_calendar,
    )

def _spr_duration_for_threshold(threshold_cfs: float) -> int | None:
    threshold_i = int(round(float(threshold_cfs)))
    for spec_threshold, duration_days in SPR_THRESHOLD_DAY_SPECS:
        if int(round(float(spec_threshold))) == threshold_i:
            return int(duration_days)
    return None

def _is_spr_actionproxy_mode(ctx: RewardContext) -> bool:
    mode = str(ctx.info.get("action_mode", "") or "")
    return (
        mode.startswith("spr_target_proxy_3d")
        or mode.startswith("esa_base_spr_proxy_4d")
    )

def _spr_actionproxy_reward(
    ctx: RewardContext,
    *,
    threshold_weights: Mapping[int, float],
    frequency_gate_mode: str = "hard",
    target_frequencies: Mapping[int, float] | None = None,
    gate_10k: bool = False,
    annual_gate_mode: str = "none",
    respect_calendar: bool = False,
    success_reward_gain: float = 1.0,
    no_action_opportunity_penalty: float = 0.05,
    unreachable_penalty_weight: float = 0.45,
    unneeded_penalty_weight: float = 0.30,
    miss_penalty_weight: float = 0.85,
    water_cost_weight: float = 0.10,
    too_low_penalty_weight: float = 0.0,
    target_match_bonus_weight: float = 0.0,
    no_action_no_opportunity_bonus: float = 0.0,
) -> float:
    """Reward the SPR target chosen by the 3-action proxy controller.

    The environment converts the third continuous action into a threshold intent
    and, if the target is spring-window reachable, raises mainstem release just
    enough to meet that threshold. This reward therefore scores the *choice* of
    threshold rather than continuous peak matching.
    """
    if not _is_spr_actionproxy_mode(ctx):
        return 0.0

    max_release = max(float(ctx.info.get("max_release_sj_main_cfs", 5_000.0)), 1.0)
    target_cfs = float(ctx.info.get("spr_proxy_target_cfs", 0.0) or 0.0)
    target_i = int(round(target_cfs))
    target_frac = float(np.clip(target_cfs / 10_000.0, 0.0, 1.0))
    in_window = bool(ctx.info.get("spr_proxy_window_active", False))
    best_terms = (
        _spr_actionproxy_highest_justified_terms(
            ctx,
            frequency_gate_mode=frequency_gate_mode,
            target_frequencies=target_frequencies,
            gate_10k=gate_10k,
            annual_gate_mode=annual_gate_mode,
            respect_calendar=respect_calendar,
        )
        if in_window
        else None
    )

    # Choosing no SPR target should usually be allowed; it is lightly penalized
    # only when the current ledger/frequency state says there is a reachable
    # needed threshold today.
    if target_i <= 0:
        if not in_window or best_terms is None:
            return float(max(float(no_action_no_opportunity_bonus), 0.0)) if in_window else 0.0
        weight = float(threshold_weights.get(int(round(best_terms.target_cfs)), 1.0))
        gate = _spr_frequency_gate_multiplier(
            ctx,
            best_terms,
            mode=frequency_gate_mode,
            target_frequencies=target_frequencies,
            gate_10k=gate_10k,
        )
        return float(-no_action_opportunity_penalty * weight * gate)

    base_weight = float(threshold_weights.get(target_i, 1.0))
    if not in_window:
        return float(-0.10 * base_weight * max(target_frac, 0.1))

    duration_days = _spr_duration_for_threshold(target_cfs)
    if duration_days is None:
        return float(-0.25 * base_weight)

    terms = _spr_threshold_bridge_terms(
        ctx,
        threshold_cfs=float(target_cfs),
        duration_days=int(duration_days),
    )
    if terms is None:
        return float(-0.10 * base_weight * max(target_frac, 0.1))

    gate = _spr_frequency_gate_multiplier(
        ctx,
        terms,
        mode=frequency_gate_mode,
        target_frequencies=target_frequencies,
        gate_10k=gate_10k,
    )
    gated_weight = base_weight * gate

    controller_need = max(
        float(ctx.info.get("spr_proxy_controller_need_cfs", 0.0) or 0.0),
        0.0,
    )
    water_frac = float(np.clip(controller_need / max_release, 0.0, 1.0))

    reachable_at_decision = ctx.info.get(
        "spr_proxy_target_reachable_at_decision",
        ctx.info.get("spr_proxy_target_reachable", False),
    )
    if not bool(reachable_at_decision):
        return float(-unreachable_penalty_weight * base_weight * (0.25 + target_frac))

    if (
        (not terms.need_more)
        or gated_weight <= 0.0
        or not _spr_actionproxy_annual_gate_allows(ctx, terms, mode=annual_gate_mode)
        or (bool(respect_calendar) and not _spr_calendar_completion_possible(ctx, terms))
    ):
        return float(-unneeded_penalty_weight * base_weight * (0.25 + water_frac))

    target_gap_penalty = 0.0
    target_match_bonus = 0.0
    if best_terms is not None:
        best_i = int(round(float(best_terms.target_cfs)))
        if target_i < best_i:
            gap_frac = float(np.clip((float(best_i) - float(target_i)) / 10_000.0, 0.0, 1.0))
            best_weight = float(threshold_weights.get(best_i, 1.0))
            target_gap_penalty = (
                float(too_low_penalty_weight) * best_weight * (0.25 + gap_frac)
            )
        elif target_i == best_i:
            target_match_bonus = float(target_match_bonus_weight) * gated_weight

    hit = bool(ctx.info.get("spr_proxy_target_hit", False))
    if hit:
        progress_days = float(terms.duration_days) * _spr_progress_increment(
            terms,
            threshold_cfs=float(terms.target_cfs),
            duration_days=int(terms.duration_days),
        )
        # Deterministic controller should make this close to one earned day.
        reward = (
            gated_weight
            * float(success_reward_gain)
            * (2.0 * progress_days + 0.50)
        )
        reward += target_match_bonus
        reward -= target_gap_penalty
        reward -= float(water_cost_weight) * water_frac
        return float(reward)

    shortfall_frac = float(
        np.clip(
            (float(terms.target_cfs) - float(terms.farmington_total_cfs))
            / max_release,
            0.0,
            1.0,
        )
    )
    return float(
        -miss_penalty_weight * gated_weight * (0.25 + shortfall_frac)
        - target_gap_penalty
    )

def _spr_proxy_actual_excess_release_penalty(
    ctx: RewardContext,
    *,
    weight: float,
    slack_cfs: float = 100.0,
) -> float:
    """Penalize actual spring-window SJ release not explained by the proxy target.

    This is deliberately separate from ``water_cost_weight`` in the proxy reward:
    that term prices the controller's computed bridge need, while this term prices
    the realized SJ release above that need, including baseline-actor leakage.
    """
    if not _is_spr_actionproxy_mode(ctx):
        return 0.0
    if not bool(ctx.info.get("spr_proxy_window_active", False)):
        return 0.0
    max_release = max(float(ctx.info.get("max_release_sj_main_cfs", 5_000.0)), 1.0)
    release_cfs = max(float(ctx.info.get("release_sj_main_cfs", 0.0) or 0.0), 0.0)
    controller_need_cfs = max(
        float(
            ctx.info.get(
                "spr_proxy_actual_bridge_need_cfs",
                ctx.info.get("spr_proxy_controller_need_cfs", 0.0),
            )
            or 0.0
        ),
        0.0,
    )
    esa_floor_need_cfs = (
        max(float(ctx.info.get("esa_floor_required_cfs", 0.0) or 0.0), 0.0)
        if bool(ctx.info.get("esa_min_flow_floor", False))
        else 0.0
    )
    protected_cfs = max(controller_need_cfs, esa_floor_need_cfs)
    excess_cfs = max(release_cfs - protected_cfs - float(slack_cfs), 0.0)
    excess_frac = float(np.clip(excess_cfs / max_release, 0.0, 1.0))
    return float(-float(weight) * excess_frac)

def _spr_actionproxy_ledger_hammer_smart(
    ctx: RewardContext,
    *,
    target_frequencies: Mapping[int, float] | None = None,
    gate_10k: bool = False,
    annual_gate_mode: str = "none",
    respect_calendar: bool = False,
    unneeded_penalty_weight: float = 0.65,
    no_action_no_opportunity_bonus: float = 0.0,
) -> float:
    return _spr_actionproxy_reward(
        ctx,
        threshold_weights=SPR_PHASE38_TRAINING_FREQ_THRESHOLD_MULTIPLIERS["ledgerfreq"],
        frequency_gate_mode="hard",
        target_frequencies=target_frequencies,
        gate_10k=gate_10k,
        annual_gate_mode=annual_gate_mode,
        respect_calendar=respect_calendar,
        success_reward_gain=3.5,
        no_action_opportunity_penalty=0.75,
        unreachable_penalty_weight=0.85,
        unneeded_penalty_weight=float(unneeded_penalty_weight),
        too_low_penalty_weight=1.50,
        target_match_bonus_weight=0.35,
        no_action_no_opportunity_bonus=float(no_action_no_opportunity_bonus),
    )

def _spr_threshold_key(threshold_cfs: float, duration_days: int) -> str:
    return f"{int(threshold_cfs)}cfs_{int(duration_days)}d"

def _spr_threshold_annual_oi(ctx: RewardContext, threshold_cfs: float, duration_days: int) -> float:
    key = _spr_threshold_key(threshold_cfs, duration_days)
    val = ctx.info.get(f"spring_oi_{key}", np.nan)
    try:
        oi = float(val)
    except Exception:
        oi = float("nan")
    if np.isfinite(oi):
        return float(np.clip(oi, 0.0, 1.0))

    # Compatibility fallback for old 10k-only OI rollouts.
    if int(round(float(threshold_cfs))) >= 10_000:
        val = ctx.info.get("spring_oi", np.nan)
        try:
            oi = float(val)
        except Exception:
            oi = float("nan")
        if np.isfinite(oi):
            return float(np.clip(oi, 0.0, 1.0))
    return float("nan")

def _spr_threshold_annual_go(
    ctx: RewardContext,
    threshold_cfs: float,
    duration_days: int,
    *,
    min_oi: float = 0.75,
) -> bool:
    key = _spr_threshold_key(threshold_cfs, duration_days)
    go_key = f"spring_go_{key}"
    if go_key in ctx.info:
        return bool(ctx.info.get(go_key, False))
    oi = _spr_threshold_annual_oi(ctx, threshold_cfs, duration_days)
    return bool(np.isfinite(oi) and oi >= float(min_oi))

@register_reward(
    "esa_spring_peak_release",
    "farmington_thresholds_actionproxy_smart_ledger_hammer_histfreq_stop_excess_strong",
)
def spr_farmington_thresholds_actionproxy_smart_ledger_hammer_histfreq_stop_excess_strong(
    ctx: RewardContext,
) -> float:
    """Selected operational SPR proxy reward with strong actual-release excess penalty."""
    return (
        _spr_actionproxy_ledger_hammer_smart(
            ctx,
            target_frequencies=SPR_HISTORIC_TARGET_FREQUENCIES,
            gate_10k=True,
            unneeded_penalty_weight=2.0,
            no_action_no_opportunity_bonus=0.15,
        )
        + _spr_proxy_actual_excess_release_penalty(ctx, weight=1.50, slack_cfs=50.0)
    )

@register_reward(
    "esa_spring_peak_release",
    "farmington_thresholds_actionproxy_smart_ledger_hammer_histfreq_stop_excess_strong_calendar",
)
def spr_farmington_thresholds_actionproxy_smart_ledger_hammer_histfreq_stop_excess_strong_calendar(
    ctx: RewardContext,
) -> float:
    """Retraining variant: credit only targets completable in the spring window."""
    return (
        _spr_actionproxy_ledger_hammer_smart(
            ctx,
            target_frequencies=SPR_HISTORIC_TARGET_FREQUENCIES,
            gate_10k=True,
            respect_calendar=True,
            unneeded_penalty_weight=2.0,
            no_action_no_opportunity_bonus=0.15,
        )
        + _spr_proxy_actual_excess_release_penalty(ctx, weight=1.50, slack_cfs=50.0)
    )
