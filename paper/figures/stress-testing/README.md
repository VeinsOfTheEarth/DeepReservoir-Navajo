# Stress Testing

Run `python -B paper/figures/stress-testing/build.py` from the repository root.
It writes the four-panel `stress-testing.png` and `stress-testing.pdf`.

`data/` bundles the two required sweeps, each with a summary CSV and storage
trajectory Parquet file:

- `initial-storage-sweep-summary.csv` and `initial-storage-sweep-storage.parquet`
- `inflow-scaling-sweep-summary.csv` and `inflow-scaling-sweep-storage.parquet`

These are byte-identical copies of the corresponding underscore-named files
under `artifacts/selected_policy/initial_storage_sweep/` and
`artifacts/selected_policy/inflow_scaling_sweep/`. The selected-policy rollout
and physical-environment thresholds remain shared repository inputs. Routine
rebuilds use the local sweep snapshots, not transient run directories.
