"""Observation order for the paper's four-head policy.

Storage is standardized using the original data normalization. The remaining
features are bounded fractions computed by the environment from current state,
decision-available hydrology, historical NIIP demand, the calendar, and SPR
event counters. The archived checkpoint uses same-date daily hydrologic means;
the corrected retraining mode uses prior-day Animas flow, inflow, and
evaporation. No forecast features enter either mode. Preserve this order when
loading the bundled checkpoint: its input masks refer to feature indices.
"""

SELECTED_OBS_CONTEXT = "storage_niiphist_esa_req_sprall_advice_budget_spill"

SELECTED_OBS_COLUMNS = (
    "storage_af",
    "storage_budget_frac",
    "niip_historic_demand_frac",
    "esa_required_release_frac",
    "animas_spr_frac",
    "spr_needed_frac_10000cfs_5d",
    "spr_needed_frac_8000cfs_10d",
    "spr_needed_frac_5000cfs_21d",
    "spr_needed_frac_2500cfs_10d",
    "spr_progress_10000cfs_5d",
    "spr_progress_8000cfs_10d",
    "spr_progress_5000cfs_21d",
    "spr_progress_2500cfs_10d",
    "spr_advice_target_req05_frac",
    "spr_advice_threshold_frac",
    "spr_advice_progress_frac",
    "spr_advice_viability_frac",
    "spr_advice_active",
    "spill_pressure_frac",
    "spill_avoidance_sj_frac",
)

DERIVED_OBS_COLUMNS = frozenset(SELECTED_OBS_COLUMNS[1:])
SPR_ADVICE_OBS_COLUMNS = SELECTED_OBS_COLUMNS[13:18]
OBS_CONTEXT_SPECS = {SELECTED_OBS_CONTEXT: SELECTED_OBS_COLUMNS}
OBS_CONTEXT_CHOICES = tuple(OBS_CONTEXT_SPECS)
