"""Build a compact LaTeX comparison table for the selected policy.

Rows compare historic management, the mean across all seeds in the selected
policy family, and the frozen selected policy. The selected policy is
``reward_jon_p95_peak875_hdisceff/seed_004``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from deepreservoir.drl import selected_policy
from deepreservoir.drl.metrics import compute_historic_summary_metrics


OUTPUT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = OUTPUT_DIR / "selected_policy_comparison_with_historic.csv"
DEFAULT_TEX = OUTPUT_DIR / "selected_policy_comparison_with_historic.tex"
DEFAULT_FAMILY_DIR = REPO_ROOT / "runs" / "reward_jon_p95_peak875_hdisceff"
DEFAULT_EVAL_NAME = "eval__holdout_2014_2024_08_17"
DEFAULT_FAMILY_SEED_METRICS_CSV = (
    REPO_ROOT
    / "artifacts"
    / "legacy_phase95_policy"
    / "selected_policy_family_seed_metrics.csv"
)


@dataclass(frozen=True)
class MetricSpec:
    header: str
    key: str
    units: str = "percent"
    decimals: int = 1
    latex_header: str | None = None
    latex_basis: str = r"\%"


METRICS: tuple[MetricSpec, ...] = (
    MetricSpec(
        "Spill (kAF)",
        "total_spill_af",
        units="kaf",
        decimals=1,
        latex_header="Spill",
        latex_basis="kAF",
    ),
    MetricSpec("Storage", "storage_frac_of_max_possible", latex_basis=r"\% max."),
    MetricSpec(
        "ESA min",
        "esa_min_flow_frac_days_met",
        latex_header=r"\shortstack{ESA\\min}",
        latex_basis=r"\% days",
    ),
    MetricSpec(
        "Flood safety",
        "flooding_frac_days_met",
        latex_header=r"\shortstack{Flood\\safety}",
        latex_basis=r"\% days",
    ),
    MetricSpec(
        "SPR 10k/5d",
        "spr_freq_years_meeting_10000cfs_5d",
        latex_header=r"\shortstack{SPR\\10k/5d}",
        latex_basis=r"\% years",
    ),
    MetricSpec(
        "SPR 8k/10d",
        "spr_freq_years_meeting_8000cfs_10d",
        latex_header=r"\shortstack{SPR\\8k/10d}",
        latex_basis=r"\% years",
    ),
    MetricSpec(
        "SPR 5k/21d",
        "spr_freq_years_meeting_5000cfs_21d",
        latex_header=r"\shortstack{SPR\\5k/21d}",
        latex_basis=r"\% years",
    ),
    MetricSpec(
        "SPR 2.5k/10d",
        "spr_freq_years_meeting_2500cfs_10d",
        latex_header=r"\shortstack{SPR\\2.5k/10d}",
        latex_basis=r"\% years",
    ),
    MetricSpec(
        "Hydropower",
        "hydropower_frac_of_max_possible",
        latex_basis=r"\% max.",
    ),
    MetricSpec(
        "NIIP volume",
        "niip_annual_volume_frac_of_contract",
        latex_header=r"\shortstack{NIIP\\volume}",
        latex_basis=r"\% contract",
    ),
    MetricSpec(
        "NIIP daily",
        "niip_frac_days_demand_met_in_window",
        latex_header=r"\shortstack{NIIP\\daily}",
        latex_basis=r"\% days",
    ),
)


def _load_json(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8") as f:
        return dict(json.load(f))


def _finite(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _load_family_seed_metrics_from_runs(
    family_dir: Path, eval_name: str
) -> list[dict[str, object]]:
    paths = sorted(family_dir.glob(f"seed_*/{eval_name}/eval_metrics.json"))
    if not paths:
        raise FileNotFoundError(
            f"No seed eval metrics found under {family_dir / ('seed_*' + '/' + eval_name)}"
        )
    rows: list[dict[str, object]] = []
    for path in paths:
        metrics = _load_json(path)
        row: dict[str, object] = {
            "family": family_dir.name,
            "seed": path.parents[1].name,
            "eval_name": eval_name,
            **metrics,
        }
        rows.append(row)
    return rows


def _write_family_seed_metrics_csv(rows: list[dict[str, object]], path: Path) -> None:
    frame = pd.DataFrame(rows)
    leading = ["family", "seed", "eval_name"]
    ordered = leading + [column for column in frame.columns if column not in leading]
    path.parent.mkdir(parents=True, exist_ok=True)
    frame[ordered].to_csv(path, index=False)


def _load_family_seed_metrics_csv(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Selected-family seed metrics snapshot not found: {path}")
    return pd.read_csv(path).to_dict(orient="records")


def _mean_metrics(rows: list[dict[str, object]]) -> dict[str, float]:
    frame = pd.DataFrame(rows)
    out: dict[str, float] = {}
    for spec in METRICS:
        out[spec.key] = float(pd.to_numeric(frame[spec.key], errors="coerce").mean())
    return out


def _format_metric(value: float, spec: MetricSpec) -> str:
    if math.isnan(value):
        return ""
    if spec.units == "percent":
        return f"{100.0 * value:.{spec.decimals}f}"
    if spec.units == "kaf":
        return f"{value / 1000.0:.{spec.decimals}f}"
    return f"{value:.{spec.decimals}f}"


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def _record(label: str, metrics: dict[str, float]) -> dict[str, str]:
    row = {"Management strategy": label}
    for spec in METRICS:
        row[spec.header] = _format_metric(_finite(metrics.get(spec.key)), spec)
    return row


def _write_latex(rows: list[dict[str, str]], path: Path, *, seed_count: int) -> None:
    headers = ["Management strategy", *(spec.header for spec in METRICS)]
    column_spec = r"p{2.0cm}" + "c" * (len(headers) - 1)
    latex_headers = [
        "Strategy",
        *(spec.latex_header if spec.latex_header is not None else _latex_escape(spec.header) for spec in METRICS),
    ]
    header_line = " & ".join(latex_headers) + r" \\"
    units_line = (
        r"\textit{Units/basis} & "
        + " & ".join(rf"\multicolumn{{1}}{{c}}{{{spec.latex_basis}}}" for spec in METRICS)
        + r" \\"
    )
    body_lines = [
        " & ".join(_latex_escape(row[header]) for header in headers) + r" \\"
        for row in rows
    ]
    lines = [
        r"% Generated by paper/tables/selected_policy_comparison_with_historic/build.py",
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\caption{Historic management, selected-family mean, and selected-policy performance over the 2014--2024 evaluation period. Values are percentages except spill, which is reported in thousand acre-feet (kAF). NIIP daily timing is reported as a diagnostic metric and was not part of the selected-policy pass/fail criterion.}",
        r"\label{tab:selected-policy-comparison-with-historic}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.4pt}",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\hline",
        header_line,
        units_line,
        r"\hline",
        *body_lines,
        r"\hline",
        r"\end{tabular}",
        rf"\vspace{{0.25em}}\par\footnotesize{{Selected-family mean is the arithmetic mean across all {seed_count} seeds in the selected policy family.}}",
        r"\end{table*}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_table(
    *,
    selected_metrics_json: Path,
    selected_rollout_parquet: Path,
    family_seed_metrics_csv: Path,
    family_dir: Path,
    eval_name: str,
    refresh_family_seed_metrics_from_runs: bool,
    csv_out: Path,
    tex_out: Path,
) -> pd.DataFrame:
    selected_metrics = _load_json(selected_metrics_json)
    rollout = pd.read_parquet(selected_rollout_parquet)
    historic_metrics = compute_historic_summary_metrics(rollout)
    if refresh_family_seed_metrics_from_runs or not family_seed_metrics_csv.exists():
        seed_metrics = _load_family_seed_metrics_from_runs(family_dir, eval_name)
        _write_family_seed_metrics_csv(seed_metrics, family_seed_metrics_csv)
    else:
        seed_metrics = _load_family_seed_metrics_csv(family_seed_metrics_csv)
    family_mean = _mean_metrics(seed_metrics)

    rows = [
        _record("Historic", historic_metrics),
        _record("Selected policy", selected_metrics),
        _record(f"Family mean (n={len(seed_metrics)})", family_mean),
    ]

    csv_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_out, index=False)
    _write_latex(rows, tex_out, seed_count=len(seed_metrics))
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selected-metrics-json",
        type=Path,
        default=selected_policy.SELECTED_EVAL_METRICS_JSON_PATH,
    )
    parser.add_argument(
        "--selected-rollout-parquet",
        type=Path,
        default=selected_policy.SELECTED_EVAL_ROLLOUT_PATH,
    )
    parser.add_argument(
        "--family-seed-metrics-csv",
        type=Path,
        default=DEFAULT_FAMILY_SEED_METRICS_CSV,
        help="Tracked seed-level metrics snapshot for the selected policy family.",
    )
    parser.add_argument("--family-dir", type=Path, default=DEFAULT_FAMILY_DIR)
    parser.add_argument("--eval-name", default=DEFAULT_EVAL_NAME)
    parser.add_argument(
        "--refresh-family-seed-metrics-from-runs",
        action="store_true",
        help="Refresh the tracked selected-family seed metrics snapshot from runs/.",
    )
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--tex-out", type=Path, default=DEFAULT_TEX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = build_table(
        selected_metrics_json=args.selected_metrics_json,
        selected_rollout_parquet=args.selected_rollout_parquet,
        family_seed_metrics_csv=args.family_seed_metrics_csv,
        family_dir=args.family_dir,
        eval_name=args.eval_name,
        refresh_family_seed_metrics_from_runs=args.refresh_family_seed_metrics_from_runs,
        csv_out=args.csv_out,
        tex_out=args.tex_out,
    )
    print(f"Wrote {args.csv_out}")
    print(f"Wrote {args.tex_out}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
