# Scenario Presets

These presets keep the dataset fixed and only tune network-side experiment conditions.

- `dense_reference.json`: easy communication regime used as an upper-bound reference.
- `normal_nominal.json`: default nominal regime for the main comparison table.
- `normal_fluctuation.json`: nominal density with stronger topology fluctuation.
- `sparse_topology_stress.json`: topology-constrained regime for highlighting DAG-aware placement.
- `sparse_fluctuation_stress.json`: strongest stress regime for TCR separation and robustness studies.

The sparse presets keep all in-plane ISLs and add only a small number of low-availability cross-plane ISLs. This keeps sparse scenarios connected enough to avoid all-or-nothing failures while still exposing deployment decisions that ignore dependency-aware communication locality.

All presets currently standardize:

- `sample_size = 300`
- `max_replicas_per_service = 5`
- `deployment_mode = fixed`
- `max_windows = null` for full-trace evaluation over the entire stationary arrival sample

This keeps the budgets aligned across algorithms, avoids the earlier unfair comparison caused by mismatched replica caps, and matches the paper-level assumption of long-timescale deployment with no periodic re-deployment inside the evaluation horizon.
