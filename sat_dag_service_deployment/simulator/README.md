# MV-DSD Python Simulator

This directory contains the experiment-oriented Python simulation framework for the paper's long-timescale service deployment and short-timescale DAG execution pipeline.

## Scope

The simulator implements:

- full DAG job materialization from `arrival_traces + dag_templates`
- deployment-window statistics extraction
- Walker-style dynamic satellite environment with ISL snapshots and an aggregated communication graph
- deployment algorithms:
  - `cpmv_dsd` (MV-DSD implementation; key kept for backward compatibility)
  - `dependency_blind`
  - `sfc_path_decomp`
  - `greedy_resource`
  - `random_deploy`
  - `drl_deploy` (proxy baseline)
- schedulers:
  - `heft`
  - `etf`
  - `priority`
- end-to-end experiment orchestration and result export

The default experiment interpretation is now:

- one long-timescale deployment decision is computed from a stationary demand sample
- the deployment is frozen across the full evaluation horizon
- requests are still executed in short-timescale windows with time-varying topologies

## Main Entry Points

- Materialize requests:

```bash
python3 -m sat_dag_service_deployment.simulator.cli materialize-requests
```

- Run a sample simulation:

```bash
python3 -m sat_dag_service_deployment.simulator.cli run \
  --config sat_dag_service_deployment/configs/default_simulation.json \
  --run-name sample_run
```

- Run the calibrated scenario suite:

```bash
python3 sat_dag_service_deployment/simulator/scripts/run_scenario_suite.py
```

- Run a specific calibrated scenario:

```bash
python3 -m sat_dag_service_deployment.simulator.cli run \
  --config sat_dag_service_deployment/configs/scenarios/sparse_topology_stress.json \
  --algorithm cpmv_dsd \
  --run-name sparse_topology_cpmv
```

## Output

Each run writes to `sat_dag_service_deployment/simulator/outputs/<run_name>/`:

- `summary.json`
- `deployment_plans.json`
- `per_request_results.json`
- `manifest.json`

The scenario suite also writes:

- `scenario_suite_summary.json`
- `scenario_suite_summary.csv`

## Package Layout

- `data/`: dataset loading, request materialization, deployment-window statistics
- `env/`: constellation generation, ISL snapshots, aggregate graph, routing
- `algorithms/`: CPMV-DSD and baseline deployment methods
- `schedulers/`: HEFT, ETF, and priority-based execution
- `evaluation/`: metrics aggregation
- `orchestrator.py`: two-timescale simulation driver
- `cli.py`: command-line entry point
- `scripts/run_scenario_suite.py`: batch runner for calibrated experiment scenarios

## Recommended Scenario Use

- `dense_reference`: easy regime; use as an upper-bound sanity check.
- `normal_nominal`: main nominal comparison setting.
- `normal_fluctuation`: use when topology fluctuation should matter without collapsing connectivity.
- `sparse_topology_stress`: best setting to expose the value of topology-aware DAG deployment.
- `sparse_fluctuation_stress`: strongest stress setting for TCR separation and robustness experiments.
