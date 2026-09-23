# Architecture Network Nodes

`build.py` generates only `architecture-network-nodes.png` and
`architecture-network-nodes.pdf`, showing the actor/critic networks, input
masks, hidden layers, and action interpretation. The content is based on
`config_files/selected_policy_navajo_reservoir.json`,
`src/deepreservoir/drl/policies.py`, and `src/deepreservoir/drl/environs.py`.

Run `python -B paper/figures/architecture-network-nodes/build.py` from the
repository root. No additional experiment data are needed.
