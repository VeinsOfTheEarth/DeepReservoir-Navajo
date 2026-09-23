"""Daily transitions and release control for the paper's Navajo Reservoir policy."""
import pickle
import numpy as np
import pandas as pd
from gymnasium import Env
from collections import deque, defaultdict
from gymnasium.spaces import Box

from deepreservoir.drl.rewards import (
    HIST_HOLDOUT_PLUS10_STORAGE_FRAC_OF_MAX,
    PRACTICAL_MIN_STORAGE_AF,
    RewardContext,
    SPR_PHASE28_SELECTED_TARGET_ORDER,
    SPR_THRESHOLD_DAY_SPECS,
)
from deepreservoir.data.metadata import project_metadata
from deepreservoir.define_env.hydropower_model import navajo_power_generation_scalar
from deepreservoir.define_env.niip.niip_demand import niip_daily_demand
from deepreservoir.drl.niip_targets import (
    historic_niip_delivery_target_for_dates,
    normalize_niip_fallback_mode,
)
from deepreservoir.define_env.storage_elevation.thresholds import (
    NAVAJO_DEADPOOL_ELEV_FT,
    NAVAJO_SPILL_ELEV_FT,
)
from deepreservoir.define_env.spring_peak_release.opportunity_index import (
    OIParams,
    ThresholdOIParams,
    precompute_oi_by_wy,
    precompute_threshold_oi_by_wy,
)
from deepreservoir.define_env.spring_peak_release_curve import SpringPeakReleaseCurve

m = project_metadata()

# Unit conversions (daily timestep)
# 1 cfs sustained over 1 day -> acre-feet
#   1 acre-foot = 43,560 ft^3
#   1 day       = 86,400 s
#   1 cfs-day   = 86,400 ft^3 = 86,400 / 43,560 acre-feet
CFS_TO_AF_PER_DAY = 86400.0 / 43560.0
AF_PER_DAY_TO_CFS = 1.0 / CFS_TO_AF_PER_DAY

# The frozen checkpoint expects these exact feature and action orders.
from deepreservoir.drl.observations import (
    SELECTED_OBS_CONTEXT, SELECTED_OBS_COLUMNS, SPR_ADVICE_OBS_COLUMNS,
    DERIVED_OBS_COLUMNS, OBS_CONTEXT_SPECS, OBS_CONTEXT_CHOICES,
)

ACTION_MODE_CHOICES = ("esa_base_spr_proxy_4d_max",)
ACTION_SCALING_CHOICES = ("linear",)
SPR_ADVICE_MODE_CHOICES = ("days_remaining",)
DECISION_HYDROLOGY_TIMING_CHOICES = ("same_day", "previous_day")
SPR_TARGET_PROXY_ACTION_MODES = frozenset(ACTION_MODE_CHOICES)
ESA_BASEFLOW_ACTION_MODES = frozenset(ACTION_MODE_CHOICES)
SPR_PROXY_ACTION_TARGETS_CFS = (0.0, 2500.0, 5000.0, 8000.0, 10000.0)
ESA_MIN_FLOW_TARGET_CFS = 500.0

_SPRING_PEAK_CURVE = SpringPeakReleaseCurve()


def _spr_threshold_label(threshold_cfs: float, duration_days: int) -> str:
    return f"{int(threshold_cfs)}cfs_{int(duration_days)}d"


def _spr_need_request_target_cfs(
    *,
    threshold_cfs: float,
    animas_cfs: float,
    cap_cfs: float,
    request_multiplier: float = 1.0,
) -> float:
    multiplier = float(request_multiplier)

    bridge_need = float(threshold_cfs) - float(animas_cfs)
    target = max(bridge_need, 0.0) * max(multiplier, 0.0)

    return float(np.clip(target, 0.0, float(cap_cfs)))


def obs_columns_for_context(obs_context: str | None) -> tuple[str, ...]:
    key = str(obs_context or SELECTED_OBS_CONTEXT).strip().lower()
    if key not in OBS_CONTEXT_SPECS:
        raise ValueError(
            f"Unknown obs_context {obs_context!r}. "
            f"Choose from: {sorted(OBS_CONTEXT_SPECS)}"
        )
    return OBS_CONTEXT_SPECS[key]


def normalize_action_scaling(action_scaling: str | None) -> str:
    key = str(action_scaling or "linear").strip().lower()
    if key not in ACTION_SCALING_CHOICES:
        raise ValueError(
            f"Unknown action_scaling {action_scaling!r}. "
            f"Choose from: {sorted(ACTION_SCALING_CHOICES)}"
        )
    return key


def normalize_action_mode(value: str | None) -> str:
    value = str(value or "esa_base_spr_proxy_4d_max").strip().lower()
    if value not in ACTION_MODE_CHOICES:
        raise ValueError(f"Unsupported paper configuration: {value!r}; expected {ACTION_MODE_CHOICES}")
    return value


def normalize_spr_advice_mode(value: str | None) -> str:
    value = str(value or "days_remaining").strip().lower()
    if value not in SPR_ADVICE_MODE_CHOICES:
        raise ValueError(f"Unsupported paper configuration: {value!r}; expected {SPR_ADVICE_MODE_CHOICES}")
    return value


def normalize_decision_hydrology_timing(value: str | None) -> str:
    """Normalize when daily hydrology becomes available to the controller.

    ``same_day`` retains the archived Phase-95 convention. ``previous_day``
    exposes only the previous daily Animas, inflow, and evaporation values when
    the policy and deterministic request logic act on the current date.
    """
    value = str(value or "same_day").strip().lower()
    if value not in DECISION_HYDROLOGY_TIMING_CHOICES:
        raise ValueError(
            f"Unsupported decision_hydrology_timing {value!r}; "
            f"expected {DECISION_HYDROLOGY_TIMING_CHOICES}"
        )
    return value


def spr_proxy_action_to_target(action_value: float) -> tuple[int, float]:
    """Map one normalized continuous action in [-1, 1] to an SPR target class."""
    frac = float((np.clip(float(action_value), -1.0, 1.0) + 1.0) / 2.0)
    n = len(SPR_PROXY_ACTION_TARGETS_CFS)
    idx = min(n - 1, int(np.floor(frac * float(n))))
    return int(idx), float(SPR_PROXY_ACTION_TARGETS_CFS[idx])


def _spr_proxy_target_index_for_cfs(target_cfs: float) -> int:
    target = float(target_cfs)
    for idx, val in enumerate(SPR_PROXY_ACTION_TARGETS_CFS):
        if int(round(float(val))) == int(round(target)):
            return int(idx)
    return 0


def compute_esa_required_release_frac_signal(
    *,
    animas_cfs: np.ndarray,
    max_release_sj_main_cfs: float,
    target_cfs: float = ESA_MIN_FLOW_TARGET_CFS,
) -> np.ndarray:
    """Fraction of the San Juan outlet cap needed to reach the ESA minimum."""
    cap = float(max_release_sj_main_cfs)
    if cap <= 0.0:
        return np.zeros_like(np.asarray(animas_cfs, dtype=np.float64), dtype=np.float32)
    animas = np.asarray(animas_cfs, dtype=np.float64)
    required = np.maximum(float(target_cfs) - animas, 0.0)
    return np.clip(required / cap, 0.0, 1.0).astype(np.float32, copy=False)


def compute_animas_spr_frac_signal(
    *,
    animas_cfs: np.ndarray,
    target_cfs: float = 10_000.0,
    clip_max: float = 1.5,
) -> np.ndarray:
    """Current Animas flow scaled to the hard SPR threshold, not ESA minimum flow."""
    denom = max(float(target_cfs), 1e-9)
    animas = np.asarray(animas_cfs, dtype=np.float64)
    return np.clip(animas / denom, 0.0, float(clip_max)).astype(np.float32, copy=False)


def compute_spr_threshold_needed_frac_signal(
    *,
    animas_cfs: np.ndarray,
    spring_window_active: np.ndarray,
    max_release_sj_main_cfs: float,
    threshold_cfs: float,
) -> np.ndarray:
    """Fraction of the San Juan outlet cap needed to reach an SPR threshold today.

    Outside the SPR window the signal is zero, so it acts as an Animas-relative
    opportunity cue rather than a fixed day-of-year instruction.
    """
    cap = float(max_release_sj_main_cfs)
    if cap <= 0.0:
        return np.zeros_like(np.asarray(animas_cfs, dtype=np.float64), dtype=np.float32)
    animas = np.asarray(animas_cfs, dtype=np.float64)
    active = np.asarray(spring_window_active, dtype=bool)
    needed = np.clip((float(threshold_cfs) - animas) / cap, 0.0, 1.0)
    return np.where(active, needed, 0.0).astype(np.float32, copy=False)


def scale_sj_action_fraction(frac: float, *, action_scaling: str | None = "linear",
           min_release_cfs: float = 0.0, max_release_sj_main_cfs: float = 5000.0) -> float:
    normalize_action_scaling(action_scaling)
    return float(np.clip(float(frac), 0.0, 1.0))


def scale_niip_action_fraction(frac: float, *, action_scaling: str | None = "linear",
           min_release_cfs: float = 0.0, max_release_niip_cfs: float = 2500.0) -> float:
    normalize_action_scaling(action_scaling)
    return float(np.clip(float(frac), 0.0, 1.0))


class NavajoReservoirEnv(Env):
    """Navajo Reservoir environment (daily timestep).

    One agent, four continuous actions (Box[-1, 1], shape=(4,)):
      - action[0] -> ESA/baseflow multiplier
      - action[1] -> NIIP release
      - action[2] -> quantized SPR target
      - action[3] -> discretionary San Juan release

    Important physical constraints implemented here:
      - Per-outlet capacity caps:
          release_sj_main_cfs <= max_release_sj_main_cfs (default 5000)
          release_niip_cfs    <= max_release_niip_cfs    (default 2500)
        If the agent requests more than the cap, we hard-cap (no proportional rescaling).

      - Deadpool constraint (elevation-based):
          If starting reservoir elevation is at/below NAVAJO_DEADPOOL_ELEV_FT (5775 ft),
          *no releases are physically possible*, so both outlet releases are forced to 0.

      - Spill constraint (elevation-based):
          If ending reservoir elevation would exceed NAVAJO_SPILL_ELEV_FT (6085 ft),
          the excess storage is automatically spilled (uncontrolled) to the San Juan mainstem.

      - Water-available constraint:
          Even above deadpool, releases cannot exceed the water available that day.
          The selected controller reserves the SPR bridge release first, then
          scales the remaining San Juan and NIIP requests to the water available.

    The ordered 20-feature observation schema is defined in observations.py.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        data_raw: pd.DataFrame,
        data_norm: pd.DataFrame,
        norm_stats: pd.DataFrame,
        reward_fn,
        episode_length: int | None = None,
        min_release_cfs: float = 0.0,
        max_release_sj_main_cfs: float = 5000.0,
        max_release_niip_cfs: float = 2500.0,
        is_eval: bool = False,
        obs_context: str = SELECTED_OBS_CONTEXT,
        action_scaling: str = "linear",
        action_mode: str = "esa_base_spr_proxy_4d_max",
        esa_min_flow_floor: bool = False,
        esa_baseflow_max_multiplier: float = 1.5,
        spr_proxy_priority_release: bool = True,
        spr_proxy_owns_sj_window: bool = False,
        spr_advice_mode: str = "days_remaining",
        decision_hydrology_timing: str = "same_day",
        niip_fallback_mode: str = "legacy_full_series",
        storage_budget_target_frac_of_max: float = HIST_HOLDOUT_PLUS10_STORAGE_FRAC_OF_MAX,
        mask_incomplete_initial_spr: bool = False,
        prior_day_hydrology: pd.Series | None = None,
    ):
        super().__init__()
        assert data_raw.index.equals(data_norm.index)

        self.data_raw = data_raw
        self.data_norm = data_norm
        self.norm_stats = norm_stats
        self.reward_fn = reward_fn
        self.is_eval = is_eval
        self.obs_context = str(obs_context or SELECTED_OBS_CONTEXT).strip().lower()
        self.action_scaling = normalize_action_scaling(action_scaling)
        self.action_mode = normalize_action_mode(action_mode)
        self.esa_min_flow_floor = bool(esa_min_flow_floor)
        self.esa_baseflow_max_multiplier = max(float(esa_baseflow_max_multiplier), 0.0)
        self.spr_proxy_priority_release = bool(spr_proxy_priority_release)
        self.spr_proxy_owns_sj_window = bool(spr_proxy_owns_sj_window)
        self.spr_advice_mode = normalize_spr_advice_mode(spr_advice_mode)
        self.decision_hydrology_timing = normalize_decision_hydrology_timing(
            decision_hydrology_timing
        )
        self.niip_fallback_mode = normalize_niip_fallback_mode(niip_fallback_mode)
        self.storage_budget_target_frac_of_max = float(
            storage_budget_target_frac_of_max
        )
        self.mask_incomplete_initial_spr = bool(mask_incomplete_initial_spr)
        self.prior_day_hydrology = prior_day_hydrology

        self.date_index = self.data_raw.index
        self.dates = self.date_index.to_list()
        self.n_steps = len(self.dates)

        # Episode length (in steps); default full series
        self.episode_length = (
            int(episode_length) if episode_length is not None else self.n_steps
        )

        # Release limits (per-outlet) in cfs
        self.min_release_cfs = float(min_release_cfs)
        self.max_release_sj_main_cfs = float(max_release_sj_main_cfs)
        self.max_release_niip_cfs = float(max_release_niip_cfs)

        # Deadpool/spill are defined by *elevation*; storage thresholds are derived
        # from the elevation-storage relationship for convenience/clamping.
        self.deadpool_elev_ft = float(NAVAJO_DEADPOOL_ELEV_FT)
        self.spill_elev_ft = float(NAVAJO_SPILL_ELEV_FT)

        # ESA/baseflow, NIIP, SPR target, discretionary San Juan release.
        self.action_dim = 4
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_dim,),
            dtype=np.float32,
        )

        # Two-day lag used by the downstream discharge diagnostic.
        self.sj_at_farmington_history = deque(maxlen=3)

        # ---- Observation schema checks ----
        # We REQUIRE raw columns for mass balance:
        required_raw = ["storage_af", "inflow_cfs", "evap_af"]
        for c in required_raw:
            if c not in self.data_raw.columns:
                raise KeyError(f"data_raw is missing required column '{c}'")

        # We REQUIRE normalized columns for observations. Storage is handled
        # dynamically because the agent changes it through releases.
        self.obs_cols = list(obs_columns_for_context(self.obs_context))
        required_norm = [
            c for c in self.obs_cols if c != "storage_af" and c not in DERIVED_OBS_COLUMNS
        ]
        for c in required_norm:
            if c not in self.data_norm.columns:
                raise KeyError(
                    f"data_norm is missing required column '{c}'. "
                    "Ensure preprocessing outputs the selected obs_context "
                    "columns into data_norm."
                )

        self.obs_dim = len(self.obs_cols)

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        # Frequently accessed scalars/arrays for the step hot path.
        storage_mean = float(self.norm_stats.loc["storage_af", "mean"])
        storage_std = float(self.norm_stats.loc["storage_af", "std"])
        self._storage_mean = storage_mean
        self._storage_std = storage_std if storage_std != 0.0 else 1.0

        self._raw_storage_af = self.data_raw["storage_af"].to_numpy(dtype=np.float64, copy=True)
        self._raw_inflow_cfs = self.data_raw["inflow_cfs"].to_numpy(dtype=np.float64, copy=True)
        self._raw_evap_af = self.data_raw["evap_af"].to_numpy(dtype=np.float64, copy=True)
        if "animas_farmington_q_cfs" in self.data_raw.columns:
            self._raw_animas_farmington_q_cfs = self.data_raw[
                "animas_farmington_q_cfs"
            ].to_numpy(dtype=np.float64, copy=True)
        else:
            self._raw_animas_farmington_q_cfs = np.zeros(self.n_steps, dtype=np.float64)

        self._decision_inflow_cfs = self._decision_hydrology_array(
            "inflow_cfs", self._raw_inflow_cfs
        )
        self._decision_evap_af = self._decision_hydrology_array(
            "evap_af", self._raw_evap_af
        )
        self._decision_animas_farmington_q_cfs = self._decision_hydrology_array(
            "animas_farmington_q_cfs", self._raw_animas_farmington_q_cfs
        )

        self._obs_norm_arrays: dict[str, np.ndarray] = {}
        for col in self.obs_cols:
            if col == "storage_af" or col in DERIVED_OBS_COLUMNS:
                continue
            arr = self.data_norm[col].to_numpy(dtype=np.float32, copy=True)
            finite = np.isfinite(arr)
            if not bool(finite.all()):
                if col == "spr_oi" or col.startswith("spring_oi_"):
                    fill = self._normalized_raw_zero(col)
                    arr = np.nan_to_num(arr, nan=fill, posinf=fill, neginf=fill)
                else:
                    n_bad = int((~finite).sum())
                    raise ValueError(
                        f"Observation column {col!r} contains {n_bad} non-finite values."
                    )
            self._obs_norm_arrays[col] = arr

        # Load elevation model
        with open(m.path("elev_area_storage_pickle"), "rb") as f:
            elev_models = pickle.load(f)
        # Elevation-area-storage interpolators (2019 table).
        # We only *require* capacity→elevation, but we also keep elevation→capacity when available.
        self.capacity_to_elev = elev_models["capacity_to_elevation"]
        self.elev_to_capacity = elev_models.get("elevation_to_capacity", None)

        # Ensure we have elevation -> capacity for spill clamping.
        if self.elev_to_capacity is None:
            # Build a numerical inverse from capacity->elevation as a last resort.
            # This is robust and keeps the environment runnable even if the pickle
            # only contains capacity->elevation.
            caps = np.linspace(0.0, 2_000_000.0, 20001)
            elevs = np.asarray(self.capacity_to_elev(caps), dtype=float)
            order = np.argsort(elevs)
            elevs_sorted = elevs[order]
            caps_sorted = caps[order]

            def _elev_to_capacity(elev_ft):
                elev_arr = np.asarray(elev_ft, dtype=float)
                return np.interp(elev_arr, elevs_sorted, caps_sorted)

            self.elev_to_capacity = _elev_to_capacity

        # Fast scalar lookup tables for the env hot path. When the loaded models are
        # scipy interp1d objects, we can reuse their native breakpoint arrays exactly
        # and evaluate them with np.interp, which is much cheaper for scalar calls.
        self._cap_to_elev_x: np.ndarray | None = None
        self._cap_to_elev_y: np.ndarray | None = None
        self._elev_to_cap_x: np.ndarray | None = None
        self._elev_to_cap_y: np.ndarray | None = None

        if hasattr(self.capacity_to_elev, "x") and hasattr(self.capacity_to_elev, "y"):
            self._cap_to_elev_x = np.asarray(self.capacity_to_elev.x, dtype=np.float64)
            self._cap_to_elev_y = np.asarray(self.capacity_to_elev.y, dtype=np.float64)
        else:
            _caps = np.linspace(0.0, 2_000_000.0, 20001, dtype=np.float64)
            _elevs = np.asarray(self.capacity_to_elev(_caps), dtype=np.float64)
            self._cap_to_elev_x = _caps
            self._cap_to_elev_y = _elevs

        if hasattr(self.elev_to_capacity, "x") and hasattr(self.elev_to_capacity, "y"):
            self._elev_to_cap_x = np.asarray(self.elev_to_capacity.x, dtype=np.float64)
            self._elev_to_cap_y = np.asarray(self.elev_to_capacity.y, dtype=np.float64)

        # Storage thresholds derived from the E–S curve (useful for clamping/debug).
        self.deadpool_storage_af = float(self._elev_to_capacity_scalar(self.deadpool_elev_ft))
        self.max_storage_af = float(self._elev_to_capacity_scalar(self.spill_elev_ft))
        storage_budget_floor_af = max(
            float(PRACTICAL_MIN_STORAGE_AF), float(self.deadpool_storage_af)
        )
        self.storage_budget_target_af = float(
            self.storage_budget_target_frac_of_max * self.max_storage_af
        )
        if (
            not np.isfinite(self.storage_budget_target_frac_of_max)
            or self.storage_budget_target_frac_of_max > 1.0
            or self.storage_budget_target_af <= storage_budget_floor_af
        ):
            raise ValueError(
                "storage_budget_target_frac_of_max must be finite, no greater "
                "than 1, and correspond to storage above the practical minimum; "
                f"received {self.storage_budget_target_frac_of_max!r}."
            )

        # --- SPR Opportunity Index precompute ---
        pm = project_metadata()
        params_path = pm.path("params.spr_oi_params_json")
        self.spring_oi_params: OIParams = OIParams.load(params_path)
        threshold_params_path = pm.path("params.spr_threshold_oi_params_json")
        self.spring_threshold_oi_params: ThresholdOIParams = ThresholdOIParams.load(
            threshold_params_path
        )

        self._spring_threshold_oi_keys = tuple(
            spec.key for spec in self.spring_threshold_oi_params.thresholds
        )

        _wy_by_day = pd.Series(
            (self.data_raw.index.year + (self.data_raw.index.month >= 10)).astype(int),
            index=self.data_raw.index,
            name="wy",
        )

        if "spr_oi" in self.data_raw.columns:
            self._spring_oi_by_wy = pd.Series(dtype=np.float64)
            self._spring_go_by_wy = pd.Series(dtype=bool)
            self.spring_oi_daily = pd.to_numeric(
                self.data_raw["spr_oi"],
                errors="coerce",
            ).astype(float)
            if "spr_go" in self.data_raw.columns:
                _go_daily = (
                    pd.to_numeric(self.data_raw["spr_go"], errors="coerce")
                    .fillna(0.0)
                    >= 0.5
                )
            else:
                _go_daily = self.spring_oi_daily >= float(
                    self.spring_oi_params.omega_on_line
                )
            self.spring_go_daily = _go_daily.astype(bool)
        else:
            _df_wy = precompute_oi_by_wy(self.data_raw, self.spring_oi_params)
            self._spring_oi_by_wy = _df_wy["oi"]
            self._spring_go_by_wy = _df_wy["go"]
            _oi_map = self._spring_oi_by_wy.to_dict()
            _go_map = self._spring_go_by_wy.to_dict()
            self.spring_oi_daily = _wy_by_day.map(_oi_map).astype(float)
            _go_daily = _wy_by_day.map(_go_map).astype("boolean").fillna(False)
            self.spring_go_daily = _go_daily.astype(bool)

        self._spring_oi_daily = self.spring_oi_daily.to_numpy(dtype=np.float64, copy=True)
        self._spring_go_daily = self.spring_go_daily.to_numpy(dtype=bool, copy=True)
        self._spring_threshold_oi_daily: dict[str, np.ndarray] = {}
        self._spring_threshold_go_daily: dict[str, np.ndarray] = {}
        missing_threshold_oi_keys: list[str] = []
        for key in self._spring_threshold_oi_keys:
            oi_daily_col = f"spring_oi_{key}"
            go_daily_col = f"spring_go_{key}"
            if oi_daily_col not in self.data_raw.columns:
                missing_threshold_oi_keys.append(key)
                continue
            oi_daily = pd.to_numeric(
                self.data_raw[oi_daily_col],
                errors="coerce",
            ).astype(float)
            if go_daily_col in self.data_raw.columns:
                go_daily = (
                    pd.to_numeric(self.data_raw[go_daily_col], errors="coerce")
                    .fillna(0.0)
                    >= 0.5
                )
            else:
                go_daily = oi_daily >= float(
                    self.spring_threshold_oi_params.omega_on_line
                )
            self._spring_threshold_oi_daily[key] = oi_daily.to_numpy(
                dtype=np.float64,
                copy=True,
            )
            self._spring_threshold_go_daily[key] = go_daily.to_numpy(
                dtype=bool,
                copy=True,
            )
        if missing_threshold_oi_keys:
            _df_threshold_wy = precompute_threshold_oi_by_wy(
                self.data_raw,
                self.spring_threshold_oi_params,
            )
            for key in missing_threshold_oi_keys:
                oi_col = f"oi_{key}"
                go_col = f"go_{key}"
                oi_daily = _wy_by_day.map(_df_threshold_wy[oi_col].to_dict()).astype(
                    float
                )
                go_daily = (
                    _wy_by_day.map(_df_threshold_wy[go_col].to_dict())
                    .astype("boolean")
                    .fillna(False)
                    .astype(bool)
                )
                self._spring_threshold_oi_daily[key] = oi_daily.to_numpy(
                    dtype=np.float64,
                    copy=True,
                )
                self._spring_threshold_go_daily[key] = go_daily.to_numpy(
                    dtype=bool,
                    copy=True,
                )
        _spring_target_daily = _SPRING_PEAK_CURVE.targets_for_date_index(self.date_index).astype(float)
        self._spring_target_daily = _spring_target_daily.to_numpy(dtype=np.float64, copy=True)
        self._spring_window_active_daily = (self._spring_target_daily > 0.0).astype(bool, copy=False)
        self._water_year = (
            self.date_index.year + (self.date_index.month >= 10)
        ).to_numpy(dtype=np.int32, copy=True)
        self._calendar_year = self.date_index.year.to_numpy(dtype=np.int32, copy=True)
        self._row_index = np.arange(self.n_steps, dtype=np.int64)
        _niip_hist = historic_niip_delivery_target_for_dates(
            self.data_raw.index,
            fallback_mode=self.niip_fallback_mode,
        )
        self._niip_historic_demand_cfs = np.asarray(
            _niip_hist.to_numpy(dtype=float),
            dtype=np.float64,
        )
        self._niip_historic_demand_af = (
            np.clip(self._niip_historic_demand_cfs, 0.0, None) * CFS_TO_AF_PER_DAY
        )
        self._niip_annual_target_by_year_af = {
            int(year): float(
                self._niip_historic_demand_af[self._calendar_year == int(year)].sum()
            )
            for year in np.unique(self._calendar_year)
        }
        self._obs_norm_arrays.update(self._build_derived_obs_arrays())

        # Internal state
        self.t = 0
        self.start_idx = 0
        self._segment_start_idx = 0
        self._segment_end_idx = self.n_steps - 1
        self._episode_end_idx = self.n_steps - 1
        self.storage_af: float | None = None
        self.episode_step_count = 0
        self._last_obs: np.ndarray | None = None
        self._spr_counter_wy: int | None = None
        self._spr_days_so_far_by_spec: dict[tuple[float, int], int] = {}
        self._spr_episode_start_wy: int | None = None
        self._spr_episode_start_date: pd.Timestamp | None = None
        self._spr_initial_partial_wy: int | None = None
        self._spr_recorded_years: set[int] = set()
        self._spr_success_years_by_spec: dict[tuple[float, int], set[int]] = {}
        self._niip_progress_year: int | None = None
        self._niip_target_cum_af: float = 0.0
        self._niip_delivered_cum_af: float = 0.0

        self.last_reward_breakdown: dict[str, float] | None = None

        # Episode bookkeeping
        self._episode_reward_sums: dict[str, float] = defaultdict(float)
        self._episode_total_reward: float = 0.0

        # Pass 2: compute only the per-step features needed by the active
        # reward configuration during training. Evaluation still computes the
        # full diagnostic payload for plotting and metrics export.
        self._init_step_feature_flags()

    # ---------------- Helpers ----------------

    def _init_step_feature_flags(self) -> None:
        """Compute the fields consumed by the selected rewards and observations."""
        comps = getattr(self.reward_fn, "components", None)
        active_pairs = {
            (str(comp.objective), str(comp.variant)) for comp in (comps or [])
        }
        objectives = {objective for objective, _ in active_pairs}
        spring = "esa_spring_peak_release" in objectives
        full = bool(self.is_eval or comps is None)

        self._compute_full_step_info = full
        self._need_animas_farmington = True
        self._need_sj_at_farmington = full or "flooding" in objectives or spring
        self._need_sj_at_farmington_lag2 = full or "flooding" in objectives
        self._need_end_elev = full or "hydropower" in objectives
        self._need_hydropower = full or "hydropower" in objectives
        self._need_storage_bounds = full or bool(objectives & {"dam_safety", "storage_control"})
        self._need_spring_oi = full or spring
        self._need_spring_meta = full or spring
        self._need_threshold_oi = full or spring
        self._need_spr_threshold_tracking = True
        self._need_release_caps = full or spring
        self._need_release_planning = full or "dam_safety" in objectives
        self._need_niip_historic_demand = True

        self.step_feature_flags = {
            "compute_full_step_info": bool(self._compute_full_step_info),
            "esa_min_flow_floor": bool(self.esa_min_flow_floor),
            "esa_baseflow_max_multiplier": float(self.esa_baseflow_max_multiplier),
            "spr_proxy_priority_release": bool(self.spr_proxy_priority_release),
            "spr_proxy_owns_sj_window": bool(self.spr_proxy_owns_sj_window),
            "spr_advice_mode": str(self.spr_advice_mode),
            "decision_hydrology_timing": str(self.decision_hydrology_timing),
            "niip_fallback_mode": str(self.niip_fallback_mode),
            "storage_budget_target_frac_of_max": float(
                self.storage_budget_target_frac_of_max
            ),
            "storage_budget_target_af": float(self.storage_budget_target_af),
            "mask_incomplete_initial_spr": bool(
                self.mask_incomplete_initial_spr
            ),
            "need_animas_farmington": bool(self._need_animas_farmington),
            "need_sj_at_farmington": bool(self._need_sj_at_farmington),
            "need_sj_at_farmington_lag2": bool(self._need_sj_at_farmington_lag2),
            "need_end_elev": bool(self._need_end_elev),
            "need_hydropower": bool(self._need_hydropower),
            "need_storage_bounds": bool(self._need_storage_bounds),
            "need_spring_oi": bool(self._need_spring_oi),
            "need_spring_meta": bool(self._need_spring_meta),
            "need_threshold_oi": bool(self._need_threshold_oi),
            "need_spr_threshold_tracking": bool(self._need_spr_threshold_tracking),
            "need_release_caps": bool(self._need_release_caps),
            "need_release_planning": bool(self._need_release_planning),
            "need_niip_historic_demand": bool(self._need_niip_historic_demand),
            "active_reward_components": sorted(active_pairs),
        }

    def _current_global_idx(self) -> int:
        return self.start_idx + self.t

    def _current_date(self) -> pd.Timestamp:
        return self.dates[self._current_global_idx()]

    def _norm_storage(self, storage_af: float) -> float:
        return (float(storage_af) - self._storage_mean) / self._storage_std

    def _capacity_to_elev_scalar(self, storage_af: float) -> float:
        x = self._cap_to_elev_x
        y = self._cap_to_elev_y
        if x is not None and y is not None:
            return float(np.interp(float(storage_af), x, y))
        return float(self.capacity_to_elev(float(storage_af)))

    def _elev_to_capacity_scalar(self, elev_ft: float) -> float:
        x = self._elev_to_cap_x
        y = self._elev_to_cap_y
        if x is not None and y is not None:
            return float(np.interp(float(elev_ft), x, y))
        return float(self.elev_to_capacity(float(elev_ft)))

    def _normalized_raw_zero(self, col: str) -> float:
        try:
            row = self.norm_stats.loc[col]
            mean = float(row["mean"])
            std = float(row["std"])
        except Exception:
            return 0.0
        if not np.isfinite(mean) or not np.isfinite(std) or abs(std) <= 1e-12:
            return 0.0
        return float((0.0 - mean) / std)

    def _decision_hydrology_array(
        self,
        column: str,
        raw_values: np.ndarray,
    ) -> np.ndarray:
        """Return hydrology available when each dated action is selected.

        A sliced evaluation window can provide its preceding calendar row in
        ``prior_day_hydrology``. At the absolute beginning of the source record,
        where no preceding observation exists, the first value initializes the
        lagged series. Interior random episodes retain the true preceding row
        because these arrays span the full training window.
        """
        raw = np.asarray(raw_values, dtype=np.float64)
        if self.decision_hydrology_timing == "same_day" or raw.size == 0:
            return raw.copy()

        shifted = np.empty_like(raw)
        shifted[1:] = raw[:-1]
        first_value = float(raw[0])
        prior = self.prior_day_hydrology
        if prior is not None and column in prior:
            prior_date = getattr(prior, "name", None)
            if prior_date is not None:
                expected = pd.Timestamp(self.date_index[0]).normalize() - pd.Timedelta(days=1)
                if pd.Timestamp(prior_date).normalize() != expected:
                    raise ValueError(
                        "prior_day_hydrology must be the calendar day immediately "
                        f"before {self.date_index[0]!s}"
                    )
            candidate = float(prior[column])
            if np.isfinite(candidate):
                first_value = candidate
        shifted[0] = first_value
        return shifted

    def _build_derived_obs_arrays(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        needed = {col for col in self.obs_cols if col in DERIVED_OBS_COLUMNS}
        if not needed:
            return out


        if "niip_historic_demand_frac" in needed:
            cap_cfs = max(float(self.max_release_niip_cfs), 1e-9)
            out["niip_historic_demand_frac"] = np.clip(
                self._niip_historic_demand_cfs / cap_cfs,
                0.0,
                1.0,
            ).astype(np.float32)


        if "esa_required_release_frac" in needed:
            out["esa_required_release_frac"] = compute_esa_required_release_frac_signal(
                animas_cfs=self._decision_animas_farmington_q_cfs,
                max_release_sj_main_cfs=self.max_release_sj_main_cfs,
                target_cfs=ESA_MIN_FLOW_TARGET_CFS,
            )

        if "animas_spr_frac" in needed:
            out["animas_spr_frac"] = compute_animas_spr_frac_signal(
                animas_cfs=self._decision_animas_farmington_q_cfs,
            )
        for threshold_cfs, duration_days in SPR_THRESHOLD_DAY_SPECS:
            label = _spr_threshold_label(threshold_cfs, duration_days)
            col = f"spr_needed_frac_{label}"
            if col in needed:
                out[col] = compute_spr_threshold_needed_frac_signal(
                    animas_cfs=self._decision_animas_farmington_q_cfs,
                    spring_window_active=self._spring_window_active_daily,
                    max_release_sj_main_cfs=self.max_release_sj_main_cfs,
                    threshold_cfs=float(threshold_cfs),
                )

        return out


    def _reset_initial_partial_spr_mask(self, *, idx: int) -> None:
        """Mask a first spring whose earlier episode decisions are unavailable."""
        self._spr_initial_partial_wy = None
        if not bool(getattr(self, "mask_incomplete_initial_spr", False)):
            return
        if self.n_steps <= 0:
            return

        idx_i = int(np.clip(idx, 0, self.n_steps - 1))
        if not bool(self._spring_window_active_daily[idx_i]):
            return

        wy = int(self._water_year[idx_i])
        first_month, first_day, _ = _SPRING_PEAK_CURVE.cfg.points_md_cfs[0]
        first_spring_day = pd.Timestamp(
            year=wy, month=int(first_month), day=int(first_day)
        )
        start_date = pd.Timestamp(self.date_index[idx_i]).normalize()
        if start_date > first_spring_day:
            self._spr_initial_partial_wy = wy

    def _spr_initial_partial_season_masked(self, idx: int) -> bool:
        """Whether this row belongs to the episode's incomplete first spring."""
        if not bool(getattr(self, "mask_incomplete_initial_spr", False)):
            return False
        masked_wy = getattr(self, "_spr_initial_partial_wy", None)
        if masked_wy is None:
            return False
        idx_i = int(idx)
        return bool(
            self._spring_window_active_daily[idx_i]
            and int(self._water_year[idx_i]) == int(masked_wy)
        )

    def _spr_window_active_for_episode(self, idx: int) -> bool:
        idx_i = int(idx)
        return bool(
            self._spring_window_active_daily[idx_i]
            and not self._spr_initial_partial_season_masked(idx_i)
        )

    def _spr_calendar_days_left(self, idx: int) -> int:
        """Available spring-window rows from today through the window's end."""
        idx_i = int(idx)
        if not self._spr_window_active_for_episode(idx_i):
            return 0
        return int(np.count_nonzero(
            (self._water_year == int(self._water_year[idx_i]))
            & self._spring_window_active_daily
            & (self._row_index >= idx_i)
        ))

    def _spr_advice_candidate(
        self,
        idx: int,
    ) -> tuple[float, int, float] | None:
        """Highest threshold worth showing to the SPR head today.

        ``spr_advice_mode`` controls how much information the advice can use.
        Public modes use decision-time reachability, calendar/ledger progress,
        and optional threshold-specific OI; no future Animas trajectory is used.
        """
        if not self._spr_window_active_for_episode(int(idx)):
            return None

        cap = float(self.max_release_sj_main_cfs)
        if cap <= 0.0:
            return None
        animas = float(self._decision_animas_farmington_q_cfs[int(idx)])

        calendar_days_left = self._spr_calendar_days_left(idx)
        candidates: list[tuple[float, int, float]] = []
        for threshold_cfs, duration_days in SPR_PHASE28_SELECTED_TARGET_ORDER:
            key = (float(threshold_cfs), int(duration_days))
            count = int(self._spr_days_so_far_by_spec.get(key, 0))
            remaining = int(duration_days) - int(count)
            if remaining <= 0:
                continue
            if (animas + cap) < float(threshold_cfs):
                continue

            viability = float(
                np.clip(
                    float(calendar_days_left) / max(float(remaining), 1.0),
                    0.0,
                    1.0,
                )
            )
            candidates.append((float(threshold_cfs), int(duration_days), viability))

        if not candidates:
            return None

        for threshold_cfs, duration_days, viability in candidates:
            if viability >= 1.0:
                return float(threshold_cfs), int(duration_days), float(viability)

        # If every reachable threshold appears incompleteable, still show the
        # highest current opportunity with low viability. That gives the policy
        # an explicit "this is tempting but probably a bad chase" signal.
        threshold_cfs, duration_days, viability = candidates[0]
        return float(threshold_cfs), int(duration_days), float(viability)

    def _spr_advice_obs_value(self, col: str, idx: int) -> float:
        candidate = self._spr_advice_candidate(idx)
        if candidate is None:
            return 0.0

        threshold_cfs, duration_days, viability = candidate
        if col == "spr_advice_threshold_frac":
            return float(np.clip(float(threshold_cfs) / 10_000.0, 0.0, 1.0))

        key = (float(threshold_cfs), int(duration_days))
        count = self._spr_days_so_far_by_spec.get(key, 0)
        if col == "spr_advice_progress_frac":
            return float(
                np.clip(
                    float(count) / max(float(duration_days), 1.0),
                    0.0,
                    1.0,
                )
            )
        if col == "spr_advice_viability_frac":
            return float(np.clip(float(viability), 0.0, 1.0))
        if col == "spr_advice_active":
            return 1.0 if float(viability) >= 1.0 else 0.0

        if col == "spr_advice_target_req05_frac":
            cap = float(self.max_release_sj_main_cfs)
            if cap <= 0.0:
                return 0.0
            animas = float(self._decision_animas_farmington_q_cfs[int(idx)])
            target_cfs = _spr_need_request_target_cfs(
                threshold_cfs=float(threshold_cfs),
                animas_cfs=animas,
                cap_cfs=cap,
                request_multiplier=1.05,
            )
            return float(np.clip(target_cfs / cap, 0.0, 1.0))
        return 0.0

    def _storage_budget_obs_value(self) -> float:
        if self.storage_af is None:
            return 0.0

        s_min = max(float(PRACTICAL_MIN_STORAGE_AF), float(self.deadpool_storage_af))
        s_max = float(self.max_storage_af)
        if s_max <= s_min:
            return 0.0

        target = float(
            np.clip(
                self.storage_budget_target_frac_of_max * s_max,
                s_min,
                s_max,
            )
        )
        if target <= s_min:
            return 1.0

        return float(np.clip((float(self.storage_af) - s_min) / (target - s_min), 0.0, 1.0))

    def _spill_pressure_obs_value(self, col: str, idx: int) -> float:
        if self.storage_af is None:
            return 0.0

        inflow_af = float(self._decision_inflow_cfs[int(idx)]) * CFS_TO_AF_PER_DAY
        evap_af = float(self._decision_evap_af[int(idx)])
        projected_no_control_af = float(self.storage_af) + inflow_af - evap_af
        excess_af = max(projected_no_control_af - float(self.max_storage_af), 0.0)
        excess_cfs = excess_af * AF_PER_DAY_TO_CFS

        if col == "spill_pressure_frac":
            total_cap = float(self.max_release_sj_main_cfs + self.max_release_niip_cfs)
            if total_cap <= 0.0:
                return 0.0
            return float(np.clip(excess_cfs / total_cap, 0.0, 1.0))

        if col == "spill_avoidance_sj_frac":
            sj_cap = float(self.max_release_sj_main_cfs)
            if sj_cap <= 0.0:
                return 0.0
            return float(np.clip(excess_cfs / sj_cap, 0.0, 1.0))

        return 0.0

    def _reset_niip_annual_progress(self, *, idx: int) -> None:
        if self.n_steps <= 0:
            self._niip_progress_year = None
            self._niip_target_cum_af = 0.0
            self._niip_delivered_cum_af = 0.0
            return

        idx_i = int(np.clip(idx, 0, self.n_steps - 1))
        year = int(self._calendar_year[idx_i])
        prior = (self._calendar_year == year) & (self._row_index < idx_i)
        prior_target_af = float(self._niip_historic_demand_af[prior].sum())
        self._niip_progress_year = year
        self._niip_target_cum_af = prior_target_af
        # Random mid-year training resets should begin on-schedule rather than
        # inheriting artificial backlog from dates before the agent acted.
        self._niip_delivered_cum_af = prior_target_af

    def _advance_niip_annual_progress(self, *, idx: int, release_niip_cfs: float) -> None:
        if self.n_steps <= 0:
            return
        idx_i = int(np.clip(idx, 0, self.n_steps - 1))
        year = int(self._calendar_year[idx_i])
        if self._niip_progress_year != year:
            self._reset_niip_annual_progress(idx=idx_i)

        self._niip_target_cum_af += float(self._niip_historic_demand_af[idx_i])
        self._niip_delivered_cum_af += max(float(release_niip_cfs), 0.0) * CFS_TO_AF_PER_DAY


    def _build_obs(self) -> np.ndarray:
        idx = self._current_global_idx()
        initial_partial_spr_masked = self._spr_initial_partial_season_masked(idx)
        obs_vals: list[float] = []
        for col in self.obs_cols:
            if initial_partial_spr_masked and col.startswith("spr_"):
                obs_vals.append(0.0)
            elif col == "storage_af":
                obs_vals.append(self._norm_storage(float(self.storage_af)))
            elif col.startswith("spr_advice_"):
                obs_vals.append(self._spr_advice_obs_value(col, idx))
            elif col == "storage_budget_frac":
                obs_vals.append(self._storage_budget_obs_value())
            elif col.startswith("spill_"):
                obs_vals.append(self._spill_pressure_obs_value(col, idx))
            elif col.startswith("spr_progress_"):
                matched = False
                for threshold_cfs, duration_days in SPR_THRESHOLD_DAY_SPECS:
                    label = _spr_threshold_label(threshold_cfs, duration_days)
                    if col == f"spr_progress_{label}":
                        count = self._spr_days_so_far_by_spec.get(
                            (float(threshold_cfs), int(duration_days)),
                            0,
                        )
                        obs_vals.append(
                            float(
                                np.clip(
                                    float(count) / max(float(duration_days), 1.0),
                                    0.0,
                                    1.0,
                                )
                            )
                        )
                        matched = True
                        break
                if not matched:
                    obs_vals.append(0.0)
            else:
                obs_vals.append(float(self._obs_norm_arrays[col][idx]))
        return np.asarray(obs_vals, dtype=np.float32)

    def _reset_spr_threshold_counters(self, *, wy: int | None) -> None:
        self._spr_counter_wy = None if wy is None else int(wy)
        self._spr_days_so_far_by_spec = {
            (float(threshold_cfs), int(duration_days)): 0
            for threshold_cfs, duration_days in SPR_THRESHOLD_DAY_SPECS
        }

    def _reset_spr_episode_frequency_tracking(
        self, *, wy: int | None, start_date: pd.Timestamp | None = None
    ) -> None:
        self._spr_episode_start_wy = None if wy is None else int(wy)
        self._spr_episode_start_date = (
            None if start_date is None else pd.Timestamp(start_date)
        )
        self._spr_recorded_years = set()
        self._spr_success_years_by_spec = {
            (float(threshold_cfs), int(duration_days)): set()
            for threshold_cfs, duration_days in SPR_THRESHOLD_DAY_SPECS
        }

    def _record_completed_spr_counter_year(self) -> None:
        if self._spr_counter_wy is None:
            return
        wy = int(self._spr_counter_wy)
        if wy in self._spr_recorded_years:
            return
        if wy == self._spr_episode_start_wy and self._spr_episode_start_date is not None:
            first_month, first_day, _ = _SPRING_PEAK_CURVE.cfg.points_md_cfs[0]
            first_spring_day = pd.Timestamp(
                year=wy, month=first_month, day=first_day
            )
            if self._spr_episode_start_date > first_spring_day:
                return
        self._spr_recorded_years.add(wy)
        for threshold_cfs, duration_days in SPR_THRESHOLD_DAY_SPECS:
            key = (float(threshold_cfs), int(duration_days))
            count = int(self._spr_days_so_far_by_spec.get(key, 0))
            if count >= int(duration_days):
                self._spr_success_years_by_spec.setdefault(key, set()).add(wy)

    def _sync_spr_threshold_counter_wy(self, wy: int) -> None:
        wy_i = int(wy)
        if self._spr_counter_wy != wy_i:
            self._record_completed_spr_counter_year()
            self._reset_spr_threshold_counters(wy=wy_i)

    def _spr_proxy_target_reachable_today(
        self,
        *,
        threshold_cfs: float,
        animas_cfs: float,
        spring_window_active: bool,
    ) -> bool:
        return bool(
            spring_window_active
            and float(threshold_cfs) > 0.0
            and (float(animas_cfs) + float(self.max_release_sj_main_cfs))
            >= float(threshold_cfs)
        )

    # ---------------- Gym API ----------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = dict(options or {})

        if self.is_eval:
            self.start_idx = 0
            self._segment_start_idx = 0
            self._segment_end_idx = self.n_steps - 1
            self._episode_end_idx = self.n_steps - 1
            self.max_steps = self.n_steps
        else:
            # Choose an episode start inside allowed segments (if provided).
            max_start = max(self.n_steps - self.episode_length, 0)
            self.start_idx = int(self.np_random.integers(0, max_start + 1))
            self._segment_start_idx = 0
            self._segment_end_idx = self.n_steps - 1
            self._episode_end_idx = int(min(self.start_idx + self.episode_length - 1, self.n_steps - 1))
            self.max_steps = int(self._episode_end_idx - self.start_idx + 1)

        self.t = 0
        self.episode_step_count = 0

        # init storage at episode start, allowing eval-side overrides for stress tests
        initial_storage_af = options.get("initial_storage_af", None)
        if initial_storage_af is None:
            self.storage_af = float(self._raw_storage_af[self.start_idx])
        else:
            self.storage_af = float(initial_storage_af)
            if self.storage_af < 0.0:
                self.storage_af = 0.0
        # Enforce physical spill level at reset as well (historical series should not exceed this,
        # but small inconsistencies/rounding can otherwise start an episode above the spill level).
        self.storage_af = float(min(self.storage_af, self.max_storage_af))


        self.last_reward_breakdown = None
        self.sj_at_farmington_history.clear()
        start_wy = int(self._water_year[self.start_idx]) if self.n_steps > 0 else None
        self._reset_initial_partial_spr_mask(idx=self.start_idx)
        self._reset_spr_episode_frequency_tracking(
            wy=start_wy,
            start_date=self.date_index[self.start_idx] if self.n_steps > 0 else None,
        )
        self._reset_spr_threshold_counters(wy=start_wy)
        self._reset_niip_annual_progress(idx=self.start_idx)

        self._episode_reward_sums = defaultdict(float)
        self._episode_total_reward = 0.0

        obs = self._build_obs()
        self._last_obs = obs
        return obs, {}

    def step(self, action):
        # ---- action shaping ----
        action = np.asarray(action, dtype=np.float32).squeeze()
        expected_action_dim = 4
        if action.ndim != 1 or action.shape[0] != expected_action_dim:
            raise ValueError(
                f"Expected action shape ({expected_action_dim},), got {action.shape}"
            )
        action = np.clip(action, -1.0, 1.0)

        obs = self._last_obs
        if obs is None:
            obs = self._build_obs()
            self._last_obs = obs
        global_idx = self._current_global_idx()
        date = self.dates[global_idx]

        # Map actions to nonnegative per-outlet releases (cfs)
        esa_baseflow_action_mode = bool(self.action_mode in ESA_BASEFLOW_ACTION_MODES)
        frac_esa_baseflow = (action[0] + 1.0) / 2.0
        frac_sj = frac_esa_baseflow
        frac_niip = (action[1] + 1.0) / 2.0
        raw_action_spr_proxy = float(action[2]) if expected_action_dim >= 3 else np.nan
        spr_proxy_action_frac = (
            float((raw_action_spr_proxy + 1.0) / 2.0)
            if expected_action_dim >= 3
            else 0.0
        )
        raw_action_discretionary_sj = float(action[3]) if expected_action_dim >= 4 else np.nan
        frac_discretionary_sj = (
            float((raw_action_discretionary_sj + 1.0) / 2.0)
            if expected_action_dim >= 4
            else 0.0
        )
        raw_spr_proxy_target_index = 0
        raw_spr_proxy_target_cfs = 0.0
        spr_proxy_target_index = 0
        spr_proxy_target_cfs = 0.0
        spr_proxy_mask_applied = False
        spr_proxy_mask_mode = "none"
        spr_calendar_window_active = False
        spr_initial_partial_season_masked = False
        spr_proxy_window_active = False
        spr_proxy_target_reachable = False
        spr_proxy_target_reachable_at_decision = False
        spr_proxy_controller_need_cfs = 0.0
        spr_proxy_actual_bridge_need_cfs = 0.0
        spr_proxy_baseline_request_cfs = 0.0
        spr_proxy_added_request_cfs = 0.0
        spr_proxy_suppressed_baseline_request_cfs = 0.0
        spr_proxy_target_hit_after_release = False
        spr_proxy_farmington_after_release_cfs = np.nan
        spr_proxy_priority_release_cfs = 0.0
        spr_proxy_priority_scaled = False
        spr_useful_unrequested_sj_cfs = 0.0
        spr_useful_unrequested_threshold_cfs = 0.0
        spr_useful_unrequested_duration_days = 0
        esa_base_required_cfs = 0.0
        esa_base_request_cfs = 0.0
        esa_base_multiplier = 0.0
        discretionary_sj_request_cfs = 0.0
        combined_sj_request_cfs = 0.0
        sj_request_combination_rule = "max"

        scaled_frac_sj = scale_sj_action_fraction(
            frac_sj,
            action_scaling=self.action_scaling,
            min_release_cfs=self.min_release_cfs,
            max_release_sj_main_cfs=self.max_release_sj_main_cfs,
        )
        scaled_frac_discretionary_sj = scale_sj_action_fraction(
            frac_discretionary_sj,
            action_scaling=self.action_scaling,
            min_release_cfs=self.min_release_cfs,
            max_release_sj_main_cfs=self.max_release_sj_main_cfs,
        )
        scaled_frac_niip = scale_niip_action_fraction(
            frac_niip,
            action_scaling=self.action_scaling,
            min_release_cfs=self.min_release_cfs,
            max_release_niip_cfs=self.max_release_niip_cfs,
        )

        decision_animas_cfs = float(
            self._decision_animas_farmington_q_cfs[global_idx]
        )
        actual_animas_cfs = float(self._raw_animas_farmington_q_cfs[global_idx])
        animas_for_esa_floor_cfs = decision_animas_cfs
        esa_floor_decision_required_cfs = float(
            max(ESA_MIN_FLOW_TARGET_CFS - animas_for_esa_floor_cfs, 0.0)
        )
        esa_floor_required_cfs = float(
            max(ESA_MIN_FLOW_TARGET_CFS - actual_animas_cfs, 0.0)
        )
        esa_base_required_cfs = float(esa_floor_decision_required_cfs)
        esa_base_multiplier = float(
            self.esa_baseflow_max_multiplier * frac_esa_baseflow
        )
        esa_base_request_cfs = float(
            min(
                self.max_release_sj_main_cfs,
                max(0.0, esa_base_required_cfs * esa_base_multiplier),
            )
        )
        discretionary_sj_request_cfs = float(
            self.min_release_cfs
            + scaled_frac_discretionary_sj
            * (self.max_release_sj_main_cfs - self.min_release_cfs)
        )
        requested_release_sj_main_cfs = float(esa_base_request_cfs)
        scaled_frac_sj = (
            float(esa_base_request_cfs / max(float(self.max_release_sj_main_cfs), 1e-9))
        )
        requested_release_niip_cfs = self.min_release_cfs + scaled_frac_niip * (
            self.max_release_niip_cfs - self.min_release_cfs
        )
        spr_proxy_baseline_request_cfs = float(requested_release_sj_main_cfs)

        raw_spr_proxy_target_index, raw_spr_proxy_target_cfs = spr_proxy_action_to_target(
            raw_action_spr_proxy
        )
        spr_proxy_target_index = int(raw_spr_proxy_target_index)
        spr_proxy_target_cfs = float(raw_spr_proxy_target_cfs)
        spr_calendar_window_active = bool(self._spring_window_active_daily[global_idx])
        spr_initial_partial_season_masked = (
            self._spr_initial_partial_season_masked(global_idx)
        )
        spr_proxy_window_active = self._spr_window_active_for_episode(global_idx)
        if spr_initial_partial_season_masked:
            spr_proxy_mask_applied = bool(spr_proxy_target_cfs > 0.0)
            spr_proxy_mask_mode = "incomplete_initial_spring"
            spr_proxy_target_index = 0
            spr_proxy_target_cfs = 0.0
        self._sync_spr_threshold_counter_wy(int(self._water_year[global_idx]))
        animas_for_spr_proxy_cfs = decision_animas_cfs

        spr_proxy_target_reachable_at_decision = bool(
            self._spr_proxy_target_reachable_today(
                threshold_cfs=float(spr_proxy_target_cfs),
                animas_cfs=animas_for_spr_proxy_cfs,
                spring_window_active=spr_proxy_window_active,
            )
        )
        if spr_proxy_target_reachable_at_decision:
            spr_proxy_controller_need_cfs = float(
                np.clip(
                    float(spr_proxy_target_cfs) - animas_for_spr_proxy_cfs,
                    0.0,
                    float(self.max_release_sj_main_cfs),
                )
            )
        sj_request_combination_rule = "max"
        requested_release_sj_main_cfs = float(
            max(
                float(esa_base_request_cfs),
                float(spr_proxy_controller_need_cfs),
                float(discretionary_sj_request_cfs),
            )
        )
        combined_sj_request_cfs = float(requested_release_sj_main_cfs)
        spr_proxy_added_request_cfs = float(
            max(
                float(requested_release_sj_main_cfs)
                - float(spr_proxy_baseline_request_cfs),
                0.0,
            )
        )
        spr_proxy_suppressed_baseline_request_cfs = float(
            max(
                float(spr_proxy_baseline_request_cfs)
                - float(requested_release_sj_main_cfs),
                0.0,
            )
        )

        # --- Deadpool constraint: no releases below deadpool ---
        # We implement this in *elevation* space (outlet intake constraint), using the
        # starting elevation of the step. This is conservative: if inflow during the day
        # would raise the reservoir above deadpool, releases become possible on the next step.
        start_elev_ft = self._capacity_to_elev_scalar(float(self.storage_af))
        deadpool_block = start_elev_ft <= float(self.deadpool_elev_ft)

        if deadpool_block:
            release_sj_main_cfs = 0.0
            release_niip_cfs = 0.0
            cap_penalty = 0.0
        else:
            # --- Per-outlet hard caps ---
            release_sj_main_cfs = float(
                min(requested_release_sj_main_cfs, self.max_release_sj_main_cfs)
            )
            release_niip_cfs = float(min(requested_release_niip_cfs, self.max_release_niip_cfs))

            # Aggregate cap-penalty = fraction of requested flow clipped by outlet caps
            clipped = (requested_release_sj_main_cfs - release_sj_main_cfs) + (
                requested_release_niip_cfs - release_niip_cfs
            )
            cap_penalty = float(max(clipped, 0.0)) / (
                (self.max_release_sj_main_cfs + self.max_release_niip_cfs) + 1e-9
            )

        animas_for_esa_floor_cfs = decision_animas_cfs
        esa_floor_release_before_cfs = float(release_sj_main_cfs)
        if self.esa_min_flow_floor and not deadpool_block:
            release_sj_main_cfs = float(
                min(
                    self.max_release_sj_main_cfs,
                    max(release_sj_main_cfs, esa_floor_decision_required_cfs),
                )
            )

        total_cfs = float(release_sj_main_cfs + release_niip_cfs)

        # --- Physical feasibility (water available) ---
        # IMPORTANT: The reservoir state is volume (acre-feet). We enforce the
        # feasibility constraint in *volume space* to avoid extra AF<->CFS
        # round-tripping and to make the mass balance more explicit.
        inflow_cfs = float(self._raw_inflow_cfs[global_idx])
        inflow_af = inflow_cfs * CFS_TO_AF_PER_DAY
        evap_af = float(self._raw_evap_af[global_idx])

        available_af = max(float(self.storage_af) + inflow_af - evap_af, 0.0)
        requested_total_af = float(total_cfs) * CFS_TO_AF_PER_DAY

        phys_penalty = 0.0
        if requested_total_af > available_af:
            pre_phys_total = float(total_cfs)
            if (
                self.spr_proxy_priority_release
                and self.action_mode in SPR_TARGET_PROXY_ACTION_MODES
                and spr_proxy_window_active
                and spr_proxy_target_cfs > 0.0
                and spr_proxy_target_reachable_at_decision
                and spr_proxy_controller_need_cfs > 0.0
            ):
                # Preserve the deterministic SPR bridge first, then scale
                # non-priority SJ and NIIP requests with whatever water remains.
                available_cfs = float(available_af) * AF_PER_DAY_TO_CFS
                priority_sj_cfs = float(
                    min(
                        float(release_sj_main_cfs),
                        float(spr_proxy_controller_need_cfs),
                        max(available_cfs, 0.0),
                    )
                )
                nonpriority_sj_cfs = float(max(release_sj_main_cfs - priority_sj_cfs, 0.0))
                remaining_available_cfs = float(max(available_cfs - priority_sj_cfs, 0.0))
                remaining_requested_cfs = float(nonpriority_sj_cfs + release_niip_cfs)
                if remaining_requested_cfs > 1e-9 and remaining_available_cfs > 0.0:
                    scale = float(min(remaining_available_cfs / remaining_requested_cfs, 1.0))
                    release_sj_main_cfs = float(priority_sj_cfs + nonpriority_sj_cfs * scale)
                    release_niip_cfs = float(release_niip_cfs * scale)
                    spr_proxy_priority_scaled = bool(scale < 0.999999)
                else:
                    release_sj_main_cfs = float(priority_sj_cfs)
                    release_niip_cfs = 0.0
                    spr_proxy_priority_scaled = bool(remaining_requested_cfs > 1e-9)
                spr_proxy_priority_release_cfs = float(priority_sj_cfs)
            else:
                # Scale controlled releases proportionally to satisfy mass balance.
                # (Ratio is identical in cfs or af/day because the timestep is fixed.)
                scale = float(available_af) / (requested_total_af + 1e-9)
                release_sj_main_cfs *= scale
                release_niip_cfs *= scale
            total_cfs = float(release_sj_main_cfs + release_niip_cfs)

            phys_penalty = (pre_phys_total - float(total_cfs)) / (
                (self.max_release_sj_main_cfs + self.max_release_niip_cfs) + 1e-9
            )

        esa_floor_added_cfs = float(max(release_sj_main_cfs - esa_floor_release_before_cfs, 0.0))
        esa_floor_active = bool(self.esa_min_flow_floor and esa_floor_added_cfs > 1e-9)
        esa_floor_met_after_release = bool(
            (actual_animas_cfs + float(release_sj_main_cfs)) >= ESA_MIN_FLOW_TARGET_CFS
        )

        spr_proxy_target_reachable = bool(
            self._spr_proxy_target_reachable_today(
                threshold_cfs=float(spr_proxy_target_cfs),
                animas_cfs=actual_animas_cfs,
                spring_window_active=spr_proxy_window_active,
            )
        )
        if (
            spr_proxy_window_active
            and spr_proxy_target_cfs > 0.0
            and spr_proxy_target_reachable_at_decision
        ):
            spr_proxy_actual_bridge_need_cfs = float(
                np.clip(
                    float(spr_proxy_target_cfs) - actual_animas_cfs,
                    0.0,
                    float(self.max_release_sj_main_cfs),
                )
            )

        # Exclusive attribution of the final controlled SJ release.  This is a
        # diagnostic accounting convention, not a physical law: SPR bridge water
        # can also satisfy ESA.  We allocate in the operational priority order we
        # want to inspect: SPR target bridge first, then residual ESA/baseflow
        # need, then discretionary SJ release.
        remaining_sj_attrib_cfs = float(max(release_sj_main_cfs, 0.0))
        spr_attributed_sj_cfs = float(
            min(remaining_sj_attrib_cfs, max(float(spr_proxy_controller_need_cfs), 0.0))
        )
        remaining_sj_attrib_cfs = float(
            max(remaining_sj_attrib_cfs - spr_attributed_sj_cfs, 0.0)
        )
        if self._need_spr_threshold_tracking:
            spr_window_for_useful_attr = self._spr_window_active_for_episode(global_idx)
            animas_for_useful_attr_cfs = float(self._raw_animas_farmington_q_cfs[global_idx])
            farmington_for_useful_attr_cfs = float(
                animas_for_useful_attr_cfs + float(release_sj_main_cfs)
            )
            for threshold_cfs, duration_days in sorted(
                SPR_THRESHOLD_DAY_SPECS,
                key=lambda spec: float(spec[0]),
                reverse=True,
            ):
                key = (float(threshold_cfs), int(duration_days))
                count_before_today = int(self._spr_days_so_far_by_spec.get(key, 0))
                if count_before_today >= int(duration_days):
                    continue
                if not spr_window_for_useful_attr:
                    continue
                if farmington_for_useful_attr_cfs < float(threshold_cfs):
                    continue

                useful_bridge_need_cfs = float(
                    max(float(threshold_cfs) - animas_for_useful_attr_cfs, 0.0)
                )
                spr_useful_unrequested_sj_cfs = float(
                    min(
                        remaining_sj_attrib_cfs,
                        max(useful_bridge_need_cfs - spr_attributed_sj_cfs, 0.0),
                    )
                )
                spr_useful_unrequested_threshold_cfs = float(threshold_cfs)
                spr_useful_unrequested_duration_days = int(duration_days)
                remaining_sj_attrib_cfs = float(
                    max(remaining_sj_attrib_cfs - spr_useful_unrequested_sj_cfs, 0.0)
                )
                break
        esa_residual_after_spr_cfs = float(
            max(
                float(esa_floor_required_cfs)
                - spr_attributed_sj_cfs
                - spr_useful_unrequested_sj_cfs,
                0.0,
            )
        )
        esa_attributed_sj_cfs = float(
            min(remaining_sj_attrib_cfs, esa_residual_after_spr_cfs)
        )
        remaining_sj_attrib_cfs = float(
            max(remaining_sj_attrib_cfs - esa_attributed_sj_cfs, 0.0)
        )
        hydropower_attributed_sj_cfs = 0.0
        discretionary_attributed_sj_cfs = float(remaining_sj_attrib_cfs)
        esa_covered_by_controlled_release_cfs = float(
            min(
                max(float(release_sj_main_cfs), 0.0),
                max(float(esa_floor_required_cfs), 0.0),
            )
        )
        spr_covered_by_controlled_release_cfs = float(
            min(max(float(release_sj_main_cfs), 0.0), max(float(spr_proxy_controller_need_cfs), 0.0))
        )
        objective_need_cfs = float(
            max(float(esa_floor_required_cfs), float(spr_proxy_actual_bridge_need_cfs))
        )
        release_beyond_esa_spr_need_cfs = float(
            max(float(release_sj_main_cfs) - objective_need_cfs, 0.0)
        )
        release_beyond_esa_spr_request_cfs = float(
            max(
                float(release_sj_main_cfs)
                - max(float(esa_base_request_cfs), float(spr_proxy_controller_need_cfs)),
                0.0,
            )
        )
        animas_for_spr_proxy_cfs = actual_animas_cfs
        spr_proxy_farmington_after_release_cfs = float(
            animas_for_spr_proxy_cfs + float(release_sj_main_cfs)
        )
        spr_proxy_target_hit_after_release = bool(
            spr_proxy_window_active
            and spr_proxy_target_cfs > 0.0
            and spr_proxy_farmington_after_release_cfs >= float(spr_proxy_target_cfs)
        )

        # --- Mass balance update ---
        # --- Mass balance update (controlled releases) ---
        controlled_total_cfs = float(total_cfs)
        controlled_total_af = controlled_total_cfs * CFS_TO_AF_PER_DAY
        new_storage_af = float(self.storage_af) + inflow_af - evap_af - controlled_total_af
        new_storage_af = max(new_storage_af, 0.0)

        # --- Automatic spill (physical): any water above the spill level is released ---
        # We model this as an uncontrolled spill that goes to the San Juan mainstem (not NIIP).
        spill_af = 0.0
        spill_cfs = 0.0
        if new_storage_af > float(self.max_storage_af):
            spill_af = float(new_storage_af - float(self.max_storage_af))
            spill_cfs = float(spill_af * AF_PER_DAY_TO_CFS)
            new_storage_af = float(self.max_storage_af)

        # Totals including spill (actual outflow at the dam)
        sj_main_flow_cfs = float(release_sj_main_cfs + spill_cfs)
        total_cfs = float(sj_main_flow_cfs + release_niip_cfs)
        total_af = float(controlled_total_af + spill_af)

        # Pass 2: during training, only compute the expensive/optional fields
        # required by the active reward configuration. Eval keeps the full
        # payload for plotting/metrics compatibility.
        animas_cfs = 0.0
        if self._need_animas_farmington:
            animas_cfs = actual_animas_cfs

        sj_at_farm_cfs = None
        sj_at_farm_lag2_cfs = None
        if self._need_sj_at_farmington:
            # IMPORTANT: mainstem outlet contributes to Farmington; NIIP does not.
            sj_at_farm_cfs = animas_cfs + float(sj_main_flow_cfs)
            if self._need_sj_at_farmington_lag2:
                self.sj_at_farmington_history.append(float(sj_at_farm_cfs))
                sj_at_farm_lag2_cfs = (
                    self.sj_at_farmington_history[-3]
                    if len(self.sj_at_farmington_history) >= 3
                    else None
                )

        new_elev_ft = None
        if self._need_end_elev or self._need_hydropower:
            new_elev_ft = self._capacity_to_elev_scalar(new_storage_af)

        hydropower_mwh = None
        if self._need_hydropower:
            # Hydropower uses *controlled* (turbine) San Juan mainstem release.
            # Spill is uncontrolled and should not contribute to generation.
            hydropower_mwh = navajo_power_generation_scalar(
                cfs_value=float(release_sj_main_cfs),
                elevation_ft=float(new_elev_ft),
            )

        wy = None
        oi_val = None
        go_val = None
        spring_window_active = False
        spr_threshold_fields: dict[str, bool | int | float] = {}
        threshold_oi_fields: dict[str, bool | float] = {}
        spr_advice_fields: dict[str, float] = {}
        if self._need_spring_meta:
            wy = int(self._water_year[global_idx])
            go_val = bool(self._spring_go_daily[global_idx])
        if self._need_spring_oi:
            oi_val = float(self._spring_oi_daily[global_idx])
        if self._need_threshold_oi:
            for key in self._spring_threshold_oi_keys:
                threshold_oi_fields[f"spring_oi_{key}"] = float(
                    self._spring_threshold_oi_daily[key][global_idx]
                )
                threshold_oi_fields[f"spring_go_{key}"] = bool(
                    self._spring_threshold_go_daily[key][global_idx]
                )
        if self._need_spr_threshold_tracking:
            if wy is None:
                wy = int(self._water_year[global_idx])
            self._sync_spr_threshold_counter_wy(wy)
            spring_window_active = self._spr_window_active_for_episode(global_idx)
            spr_threshold_fields["spr_calendar_window_active"] = bool(
                spr_calendar_window_active
            )
            spr_threshold_fields["spr_window_active"] = bool(spring_window_active)
            spr_threshold_fields["spr_initial_partial_season_masked"] = bool(
                spr_initial_partial_season_masked
            )
            spr_threshold_fields["spr_reward_eligible"] = bool(
                spring_window_active
            )
            spr_threshold_fields["spr_calendar_days_left"] = self._spr_calendar_days_left(global_idx)
            spr_threshold_fields["spr_completed_years_so_far"] = int(
                len(self._spr_recorded_years)
            )
            for threshold_cfs, duration_days in SPR_THRESHOLD_DAY_SPECS:
                label = _spr_threshold_label(threshold_cfs, duration_days)
                count = int(
                    self._spr_days_so_far_by_spec[
                        (float(threshold_cfs), int(duration_days))
                    ]
                )
                spr_threshold_fields[f"spr_days_so_far_{label}"] = count
                spr_threshold_fields[f"spr_success_years_so_far_{label}"] = int(
                    len(
                        self._spr_success_years_by_spec.get(
                            (float(threshold_cfs), int(duration_days)),
                            set(),
                        )
                    )
                )
                spr_threshold_fields[f"spr_need_more_{label}"] = bool(
                    count < int(duration_days)
                )
                spr_threshold_fields[f"spr_reachable_{label}"] = bool(
                    spring_window_active
                    and count < int(duration_days)
                    and (float(actual_animas_cfs) + float(self.max_release_sj_main_cfs))
                    >= float(threshold_cfs)
                )
                spr_threshold_fields[f"spr_reachable_at_decision_{label}"] = bool(
                    spring_window_active
                    and count < int(duration_days)
                    and (float(decision_animas_cfs) + float(self.max_release_sj_main_cfs))
                    >= float(threshold_cfs)
                )

        for col in SPR_ADVICE_OBS_COLUMNS:
            spr_advice_fields[col] = float(self._spr_advice_obs_value(col, global_idx))

        info = {
            "date": date,
            "storage_af": float(new_storage_af),
            "prev_storage_af": float(self.storage_af),
            "release_sj_main_cfs": float(release_sj_main_cfs),
            "release_niip_cfs": float(release_niip_cfs),
            # Daily outlet-flow approximation at the Archuleta gage location.
            # The Animas joins downstream and is therefore excluded; uncontrolled
            # spill remains part of the San Juan mainstem flow.
            "sj_at_archuleta_proxy_cfs": float(sj_main_flow_cfs),
        }

        if self.esa_min_flow_floor or self._compute_full_step_info:
            info.update(
                {
                    "esa_min_flow_floor": bool(self.esa_min_flow_floor),
                    "esa_min_flow_target_cfs": float(ESA_MIN_FLOW_TARGET_CFS),
                    "esa_floor_required_cfs": float(esa_floor_required_cfs),
                    "esa_floor_decision_required_cfs": float(
                        esa_floor_decision_required_cfs
                    ),
                    "esa_floor_release_before_cfs": float(esa_floor_release_before_cfs),
                    "esa_floor_added_cfs": float(esa_floor_added_cfs),
                    "esa_floor_active": bool(esa_floor_active),
                    "esa_floor_met_after_release": bool(esa_floor_met_after_release),
                }
            )

        if self._need_storage_bounds:
            info["deadpool_storage_af"] = float(self.deadpool_storage_af)
            info["max_storage_af"] = float(self.max_storage_af)

        if self._need_end_elev:
            info["elev_ft"] = float(new_elev_ft)

        if self._need_animas_farmington:
            info["animas_farmington_q_cfs"] = float(animas_cfs)

        if self._need_sj_at_farmington and sj_at_farm_cfs is not None:
            info["sj_at_farmington_cfs"] = float(sj_at_farm_cfs)

        if self._need_sj_at_farmington_lag2:
            info["sj_at_farmington_lag2_cfs"] = (
                np.nan if sj_at_farm_lag2_cfs is None else float(sj_at_farm_lag2_cfs)
            )
            # The daily model uses the Farmington proxy from two days earlier
            # as its approximation of flow near Bluff.
            info["sj_at_bluff_proxy_cfs"] = (
                np.nan if sj_at_farm_lag2_cfs is None else float(sj_at_farm_lag2_cfs)
            )

        if self._need_hydropower and hydropower_mwh is not None:
            info["hydropower_mwh"] = float(hydropower_mwh)

        if self._need_spring_oi and oi_val is not None:
            info["spring_oi"] = float(oi_val)

        if self._need_spring_meta:
            info["spring_wy"] = wy
            info["spring_go"] = go_val

        if self._need_threshold_oi:
            info.update(threshold_oi_fields)

        if self._need_release_caps:
            info["max_release_sj_main_cfs"] = float(self.max_release_sj_main_cfs)
            info["max_release_niip_cfs"] = float(self.max_release_niip_cfs)

        info.update(
            {
                "action_mode": str(self.action_mode),
                "spr_proxy_raw_target_index": int(raw_spr_proxy_target_index),
                "spr_proxy_raw_target_cfs": float(raw_spr_proxy_target_cfs),
                "spr_proxy_target_index": int(spr_proxy_target_index),
                "spr_proxy_target_cfs": float(spr_proxy_target_cfs),
                "spr_proxy_mask_applied": bool(spr_proxy_mask_applied),
                "spr_proxy_mask_mode": str(spr_proxy_mask_mode),
                "spr_proxy_action_raw": float(raw_action_spr_proxy)
                if np.isfinite(raw_action_spr_proxy)
                else np.nan,
                "spr_proxy_action_frac": float(spr_proxy_action_frac),
                "spr_proxy_window_active": bool(spr_proxy_window_active),
                "spr_proxy_target_reachable": bool(spr_proxy_target_reachable),
                "spr_proxy_target_reachable_at_decision": bool(
                    spr_proxy_target_reachable_at_decision
                ),
                "spr_proxy_controller_need_cfs": float(
                    spr_proxy_controller_need_cfs
                ),
                "spr_proxy_actual_bridge_need_cfs": float(
                    spr_proxy_actual_bridge_need_cfs
                ),
                "spr_proxy_baseline_request_sj_main_cfs": float(
                    spr_proxy_baseline_request_cfs
                ),
                "spr_proxy_added_request_cfs": float(spr_proxy_added_request_cfs),
                "spr_proxy_suppressed_baseline_request_cfs": float(
                    spr_proxy_suppressed_baseline_request_cfs
                ),
                "spr_proxy_priority_release": bool(self.spr_proxy_priority_release),
                "spr_proxy_owns_sj_window": bool(self.spr_proxy_owns_sj_window),
                "spr_advice_mode": str(self.spr_advice_mode),
                "decision_hydrology_timing": str(self.decision_hydrology_timing),
                "niip_fallback_mode": str(self.niip_fallback_mode),
                "storage_budget_target_frac_of_max": float(
                    self.storage_budget_target_frac_of_max
                ),
                "storage_budget_target_af": float(self.storage_budget_target_af),
                "mask_incomplete_initial_spr": bool(
                    self.mask_incomplete_initial_spr
                ),
                "spr_calendar_window_active": bool(spr_calendar_window_active),
                "spr_initial_partial_season_masked": bool(
                    spr_initial_partial_season_masked
                ),
                "spr_reward_eligible": bool(
                    spr_proxy_window_active
                ),
                "decision_animas_farmington_q_cfs": float(decision_animas_cfs),
                "decision_inflow_cfs": float(self._decision_inflow_cfs[global_idx]),
                "decision_evap_af": float(self._decision_evap_af[global_idx]),
                "spr_proxy_priority_release_cfs": float(
                    spr_proxy_priority_release_cfs
                ),
                "spr_proxy_priority_scaled": bool(spr_proxy_priority_scaled),
                "spr_proxy_farmington_after_release_cfs": float(
                    spr_proxy_farmington_after_release_cfs
                )
                if np.isfinite(spr_proxy_farmington_after_release_cfs)
                else np.nan,
                "spr_proxy_target_hit": bool(spr_proxy_target_hit_after_release),
                "esa_baseflow_action_mode": bool(esa_baseflow_action_mode),
                "esa_baseflow_max_multiplier": float(
                    self.esa_baseflow_max_multiplier
                ),
                "esa_base_required_sj_cfs": float(esa_base_required_cfs),
                "esa_base_request_sj_cfs": float(esa_base_request_cfs),
                "esa_base_multiplier": float(esa_base_multiplier),
                "requested_release_sj_main_cfs": float(
                    requested_release_sj_main_cfs
                ),
                "requested_release_niip_cfs": float(requested_release_niip_cfs),
                "discretionary_sj_request_cfs": float(discretionary_sj_request_cfs),
                "combined_sj_request_cfs": float(combined_sj_request_cfs),
                "sj_request_combination_rule": str(sj_request_combination_rule),
                "spr_attributed_sj_release_cfs": float(spr_attributed_sj_cfs),
                "spr_useful_unrequested_sj_release_cfs": float(
                    spr_useful_unrequested_sj_cfs
                ),
                "spr_useful_unrequested_threshold_cfs": float(
                    spr_useful_unrequested_threshold_cfs
                ),
                "spr_useful_unrequested_duration_days": int(
                    spr_useful_unrequested_duration_days
                ),
                "esa_attributed_sj_release_cfs": float(esa_attributed_sj_cfs),
                "hydropower_attributed_sj_release_cfs": float(
                    hydropower_attributed_sj_cfs
                ),
                "discretionary_attributed_sj_release_cfs": float(
                    discretionary_attributed_sj_cfs
                ),
                "esa_covered_by_controlled_release_cfs": float(
                    esa_covered_by_controlled_release_cfs
                ),
                "spr_covered_by_controlled_release_cfs": float(
                    spr_covered_by_controlled_release_cfs
                ),
                "release_beyond_esa_spr_need_cfs": float(
                    release_beyond_esa_spr_need_cfs
                ),
                "release_beyond_esa_spr_request_cfs": float(
                    release_beyond_esa_spr_request_cfs
                ),
                **spr_advice_fields,
            }
        )

        if self._need_niip_historic_demand:
            info["niip_demand_cfs"] = float(self._niip_historic_demand_cfs[global_idx])

        if self._need_release_planning:
            info.update(
                {
                    "inflow_cfs": float(inflow_cfs),
                    "inflow_af": float(inflow_af),
                    "evap_af": float(evap_af),
                    "available_af": float(available_af),
                    "total_controlled_release_cfs": float(controlled_total_cfs),
                    "total_controlled_release_af": float(controlled_total_af),
                    "spill_cfs": float(spill_cfs),
                    "spill_af": float(spill_af),
                }
            )

        if self._need_spr_threshold_tracking:
            info.update(spr_threshold_fields)

        if self._compute_full_step_info:
            raw_forcings = {"animas_farmington_q_cfs": float(animas_cfs)}
            info.update(
                {
                    "prev_elev_ft": float(start_elev_ft),
                    "inflow_cfs": float(inflow_cfs),
                    "inflow_af": float(inflow_af),
                    "evap_af": float(evap_af),
                    "available_af": float(available_af),
                    "requested_total_release_af": float(requested_total_af),
                    "total_release_cfs": float(total_cfs),
                    "total_release_af": float(total_af),
                    "spill_cfs": float(spill_cfs),
                    "spill_af": float(spill_af),
                    "sj_main_flow_cfs": float(sj_main_flow_cfs),
                    "total_controlled_release_cfs": float(controlled_total_cfs),
                    "total_controlled_release_af": float(controlled_total_af),
                    "requested_release_sj_main_cfs": float(requested_release_sj_main_cfs),
                    "requested_release_niip_cfs": float(requested_release_niip_cfs),
                    "raw_action_sj_frac": float(frac_sj),
                    "scaled_action_sj_frac": float(scaled_frac_sj),
                    "raw_action_niip_frac": float(frac_niip),
                    "scaled_action_niip_frac": float(scaled_frac_niip),
                    "raw_action_spr_proxy": float(raw_action_spr_proxy)
                    if np.isfinite(raw_action_spr_proxy)
                    else np.nan,
                    "scaled_action_spr_proxy_frac": float(spr_proxy_action_frac),
                    "raw_action_esa_base_frac": float(frac_esa_baseflow),
                    "scaled_action_esa_base_frac": float(scaled_frac_sj),
                    "raw_action_discretionary_sj_frac": float(frac_discretionary_sj),
                    "scaled_action_discretionary_sj_frac": float(
                        scaled_frac_discretionary_sj
                    ),
                    "esa_base_required_sj_cfs": float(esa_base_required_cfs),
                    "esa_base_request_sj_cfs": float(esa_base_request_cfs),
                    "esa_baseflow_max_multiplier": float(
                        self.esa_baseflow_max_multiplier
                    ),
                    "esa_base_multiplier": float(esa_base_multiplier),
                    "discretionary_sj_request_cfs": float(discretionary_sj_request_cfs),
                    "combined_sj_request_cfs": float(combined_sj_request_cfs),
                    "sj_request_combination_rule": str(sj_request_combination_rule),
                    "spr_attributed_sj_release_cfs": float(spr_attributed_sj_cfs),
                    "spr_useful_unrequested_sj_release_cfs": float(
                        spr_useful_unrequested_sj_cfs
                    ),
                    "spr_useful_unrequested_threshold_cfs": float(
                        spr_useful_unrequested_threshold_cfs
                    ),
                    "spr_useful_unrequested_duration_days": int(
                        spr_useful_unrequested_duration_days
                    ),
                    "esa_attributed_sj_release_cfs": float(esa_attributed_sj_cfs),
                    "hydropower_attributed_sj_release_cfs": float(
                        hydropower_attributed_sj_cfs
                    ),
                    "discretionary_attributed_sj_release_cfs": float(
                        discretionary_attributed_sj_cfs
                    ),
                    "esa_covered_by_controlled_release_cfs": float(
                        esa_covered_by_controlled_release_cfs
                    ),
                    "spr_covered_by_controlled_release_cfs": float(
                        spr_covered_by_controlled_release_cfs
                    ),
                    "release_beyond_esa_spr_need_cfs": float(
                        release_beyond_esa_spr_need_cfs
                    ),
                    "release_beyond_esa_spr_request_cfs": float(
                        release_beyond_esa_spr_request_cfs
                    ),
                    "max_release_sj_main_cfs": float(self.max_release_sj_main_cfs),
                    "max_release_niip_cfs": float(self.max_release_niip_cfs),
                    "deadpool_storage_af": float(self.deadpool_storage_af),
                    "deadpool_elev_ft": float(self.deadpool_elev_ft),
                    "spill_elev_ft": float(self.spill_elev_ft),
                    "deadpool_block": bool(deadpool_block),
                    "elev_ft": float(new_elev_ft),
                    "sj_at_farmington_cfs": float(sj_at_farm_cfs) if sj_at_farm_cfs is not None else float(sj_main_flow_cfs),
                    "sj_at_farmington_lag2_cfs": np.nan
                    if sj_at_farm_lag2_cfs is None
                    else float(sj_at_farm_lag2_cfs),
                    "sj_at_bluff_proxy_cfs": np.nan
                    if sj_at_farm_lag2_cfs is None
                    else float(sj_at_farm_lag2_cfs),
                    "animas_farmington_q_cfs": float(animas_cfs),
                    "max_storage_af": float(self.max_storage_af),
                    "raw_forcings": raw_forcings,
                    "hydropower_mwh": float(hydropower_mwh) if hydropower_mwh is not None else 0.0,
                    "release_cap_penalty": float(cap_penalty),
                    "release_phys_penalty": float(phys_penalty),
                    "spring_wy": wy,
                    "spring_oi": float(oi_val) if oi_val is not None else np.nan,
                    "spring_go": go_val,
                }
            )

        if self._need_spr_threshold_tracking and spring_window_active:
            farmington_for_spr = float(animas_cfs + release_sj_main_cfs)
            for threshold_cfs, duration_days in SPR_THRESHOLD_DAY_SPECS:
                if farmington_for_spr >= float(threshold_cfs):
                    key = (float(threshold_cfs), int(duration_days))
                    self._spr_days_so_far_by_spec[key] = int(
                        self._spr_days_so_far_by_spec.get(key, 0)
                    ) + 1

        self._advance_niip_annual_progress(
            idx=global_idx,
            release_niip_cfs=float(release_niip_cfs),
        )

        # Advance
        self.storage_af = float(new_storage_af)
        self.t += 1
        self.episode_step_count += 1

        global_idx_next = self._current_global_idx()
        done = bool(global_idx_next > int(self._episode_end_idx) or global_idx_next >= self.n_steps)

        # Gymnasium semantics: we treat end-of-data as "terminated"; time limits and
        # segment boundaries as "truncated".
        terminated = bool(global_idx_next >= self.n_steps)
        truncated = bool(done and not terminated)

        if not done:
            if self._need_spr_threshold_tracking:
                self._sync_spr_threshold_counter_wy(int(self._water_year[global_idx_next]))
            if (
                self._niip_progress_year is not None
                and int(self._calendar_year[global_idx_next]) != int(self._niip_progress_year)
            ):
                self._reset_niip_annual_progress(idx=global_idx_next)
            next_obs = self._build_obs()
            self._last_obs = next_obs
        else:
            next_obs = np.zeros_like(obs, dtype=np.float32)
            self._last_obs = None

        # Reward
        ctx = RewardContext(
            t=global_idx,
            date=date,
            obs=obs,
            action=action,
            next_obs=next_obs,
            info=info,
        )
        total_reward, breakdown = self.reward_fn(ctx)
        self.last_reward_breakdown = breakdown

        # Episode bookkeeping
        self._episode_total_reward += float(total_reward)
        for k, v in breakdown.items():
            self._episode_reward_sums[k] += float(v)

        info["reward_components_step"] = breakdown
        info["reward_components_episode"] = dict(self._episode_reward_sums)
        info["episode_total_reward"] = float(self._episode_total_reward)
        info["reward_components"] = breakdown

        return next_obs, float(total_reward), terminated, truncated, info
