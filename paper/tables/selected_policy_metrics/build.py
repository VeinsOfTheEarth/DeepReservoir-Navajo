"""Build the archived pre-correction policy metric table.

The table compares the frozen selected-policy metrics against historical
management over the same 2014-01-01 to 2024-08-17 holdout window. NIIP daily
timing is included as a diagnostic row but is not part of the hard pass/fail
screen.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from deepreservoir.drl import selected_policy
from deepreservoir.drl.metrics import compute_historic_summary_metrics


OUTPUT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = OUTPUT_DIR / "selected_policy_metrics.csv"
DEFAULT_MD = OUTPUT_DIR / "selected_policy_metrics.md"


@dataclass(frozen=True)
class MetricRowSpec:
    objective: str
    metric: str
    key: str
    units: str
    higher_is_better: bool
    hard_screen: bool = True
    benchmark_label: str = "Historical management"
    decimals: int = 3
    selected_override: float | None = None
    benchmark_override: float | None = None


METRIC_ROWS: tuple[MetricRowSpec, ...] = (
    MetricRowSpec(
        objective="Dam safety",
        metric="Spill volume",
        key="total_spill_af",
        units="acre-ft",
        higher_is_better=False,
        benchmark_label="Historical management",
        decimals=0,
    ),
    MetricRowSpec(
        objective="Storage",
        metric="Storage / maximum possible",
        key="storage_frac_of_max_possible",
        units="%",
        higher_is_better=True,
    ),
    MetricRowSpec(
        objective="ESA minimum flow",
        metric="Days meeting 500 cfs Farmington minimum",
        key="esa_min_flow_frac_days_met",
        units="%",
        higher_is_better=True,
    ),
    MetricRowSpec(
        objective="Flood control",
        metric="Flood-safe days",
        key="flooding_frac_days_met",
        units="%",
        higher_is_better=True,
    ),
    MetricRowSpec(
        objective="Spring peak release",
        metric="10,000 cfs / 5 d frequency",
        key="spr_freq_years_meeting_10000cfs_5d",
        units="%",
        higher_is_better=True,
    ),
    MetricRowSpec(
        objective="Spring peak release",
        metric="8,000 cfs / 10 d frequency",
        key="spr_freq_years_meeting_8000cfs_10d",
        units="%",
        higher_is_better=True,
    ),
    MetricRowSpec(
        objective="Spring peak release",
        metric="5,000 cfs / 21 d frequency",
        key="spr_freq_years_meeting_5000cfs_21d",
        units="%",
        higher_is_better=True,
    ),
    MetricRowSpec(
        objective="Spring peak release",
        metric="2,500 cfs / 10 d frequency",
        key="spr_freq_years_meeting_2500cfs_10d",
        units="%",
        higher_is_better=True,
    ),
    MetricRowSpec(
        objective="Hydropower",
        metric="Hydropower / maximum possible",
        key="hydropower_frac_of_max_possible",
        units="%",
        higher_is_better=True,
    ),
    MetricRowSpec(
        objective="NIIP",
        metric="Annual volume / contract",
        key="niip_annual_volume_frac_of_contract",
        units="%",
        higher_is_better=True,
        benchmark_label="Contractual target",
        benchmark_override=1.0,
    ),
    MetricRowSpec(
        objective="NIIP diagnostic",
        metric="Daily historical-demand timing",
        key="niip_frac_days_demand_met_in_window",
        units="%",
        higher_is_better=True,
        hard_screen=False,
        benchmark_label="Historical management",
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


def _format_value(value: float, units: str, decimals: int) -> str:
    if math.isnan(value):
        return ""
    if units == "%":
        return f"{100.0 * value:.{decimals}f}%"
    if units == "acre-ft":
        return f"{value:,.{decimals}f}"
    return f"{value:.{decimals}f}"


def _passes_screen(
    selected_value: float,
    benchmark_value: float,
    *,
    higher_is_better: bool,
    hard_screen: bool,
) -> str:
    if not hard_screen:
        return "Diagnostic"
    if math.isnan(selected_value) or math.isnan(benchmark_value):
        return ""
    eps = 5e-4
    ok = (
        selected_value >= benchmark_value - eps
        if higher_is_better
        else selected_value <= benchmark_value + eps
    )
    return "Pass" if ok else "Miss"


def _build_records(
    selected_metrics: dict[str, float],
    historical_metrics: dict[str, float],
    row_specs: Iterable[MetricRowSpec] = METRIC_ROWS,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spec in row_specs:
        selected_value = _finite(
            spec.selected_override
            if spec.selected_override is not None
            else selected_metrics.get(spec.key)
        )
        benchmark_value = _finite(
            spec.benchmark_override
            if spec.benchmark_override is not None
            else historical_metrics.get(spec.key)
        )
        delta = (
            selected_value - benchmark_value
            if not math.isnan(selected_value) and not math.isnan(benchmark_value)
            else float("nan")
        )
        rows.append(
            {
                "objective": spec.objective,
                "metric": spec.metric,
                "selected_policy": selected_value,
                "benchmark": benchmark_value,
                "delta": delta,
                "units": spec.units,
                "benchmark_label": spec.benchmark_label,
                "hard_screen": "yes" if spec.hard_screen else "no",
                "result": _passes_screen(
                    selected_value,
                    benchmark_value,
                    higher_is_better=spec.higher_is_better,
                    hard_screen=spec.hard_screen,
                ),
                "selected_policy_display": _format_value(
                    selected_value, spec.units, spec.decimals
                ),
                "benchmark_display": _format_value(
                    benchmark_value, spec.units, spec.decimals
                ),
                "delta_display": _format_value(delta, spec.units, spec.decimals),
            }
        )
    return rows


def _write_markdown(rows: list[dict[str, object]], path: Path) -> None:
    columns = [
        "Objective",
        "Metric",
        "Archived policy",
        "Benchmark",
        "Delta",
        "Screen",
    ]
    body = [
        [
            str(row["objective"]),
            str(row["metric"]),
            str(row["selected_policy_display"]),
            f"{row['benchmark_display']} ({row['benchmark_label']})",
            str(row["delta_display"]),
            str(row["result"]),
        ]
        for row in rows
    ]
    widths = [
        max(len(columns[i]), *(len(record[i]) for record in body))
        for i in range(len(columns))
    ]

    def fmt(record: list[str]) -> str:
        return "| " + " | ".join(
            record[i].ljust(widths[i]) for i in range(len(columns))
        ) + " |"

    lines = [
        "<!-- Generated by paper/tables/selected_policy_metrics/build.py. -->",
        "",
        "> **Prepublication artifact:** Values below use the archived pre-correction Phase-95 checkpoint and are not final corrected paper results.",
        "",
        fmt(columns),
        "| " + " | ".join("-" * widths[i] for i in range(len(columns))) + " |",
        *(fmt(record) for record in body),
        "",
        "NIIP daily timing is reported as a diagnostic and is not part of the hard pass/fail screen.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_table(
    *,
    metrics_json: Path,
    rollout_parquet: Path,
    csv_out: Path,
    md_out: Path,
) -> pd.DataFrame:
    selected_metrics = _load_json(metrics_json)
    rollout = pd.read_parquet(rollout_parquet)
    historical_metrics = compute_historic_summary_metrics(rollout)
    rows = _build_records(selected_metrics, historical_metrics)

    csv_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    _write_markdown(rows, md_out)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=selected_policy.SELECTED_EVAL_METRICS_JSON_PATH,
        help="Archived-policy metrics JSON artifact.",
    )
    parser.add_argument(
        "--rollout-parquet",
        type=Path,
        default=selected_policy.SELECTED_EVAL_ROLLOUT_PATH,
        help="Archived-policy holdout rollout parquet artifact.",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=DEFAULT_CSV,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=DEFAULT_MD,
        help="Output Markdown path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = build_table(
        metrics_json=args.metrics_json,
        rollout_parquet=args.rollout_parquet,
        csv_out=args.csv_out,
        md_out=args.md_out,
    )
    hard = table[table["hard_screen"] == "yes"]
    n_pass = int((hard["result"] == "Pass").sum())
    print(f"Wrote {args.csv_out}")
    print(f"Wrote {args.md_out}")
    print(f"Hard screen: {n_pass}/{len(hard)} pass")


if __name__ == "__main__":
    main()
