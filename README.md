# MV-DSD Satellite DAG Service Deployment Simulator

This repository contains the Python simulator used for the paper on
Myerson Value-based DAG Service Deployment (MV-DSD) in LEO satellite
networks. It supports public-data-driven DAG workload generation,
long-timescale service deployment, short-timescale DAG execution, and
experiment scripts for the main evaluation figures.

The internal algorithm key is still `cpmv_dsd` for backward compatibility
with the original experiment scripts. In the paper, this implementation is
reported as MV-DSD.

## What Is Included

- `sat_dag_service_deployment/simulator/`: simulator, algorithms,
  schedulers, environment model, metrics, and experiment scripts.
- `sat_dag_service_deployment/configs/`: default and scenario-specific
  experiment configurations.
- `sat_dag_service_deployment/dag_data_set/scripts/`: scripts for
  downloading NASA EONET references, generating DAG templates, simulating
  Hawkes-type arrivals, materializing requests, and adding deadlines.
- `sat_dag_service_deployment/dag_data_set/dataset/`: input dataset used by
  the simulator, including DAG templates, public-data-derived arrival traces,
  and materialized DAG requests.

Experiment result records are not included. New runs write outputs under
`sat_dag_service_deployment/simulator/outputs/`, which is ignored by Git.

## Requirements

- Python 3.10 or later
- `numpy`
- `matplotlib` for figure plotting scripts

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Quick Start

Run a small default simulation:

```bash
python -m sat_dag_service_deployment.simulator.cli run \
  --config sat_dag_service_deployment/configs/default_simulation.json \
  --run-name sample_run
```

Run MV-DSD under the sparse topology stress scenario:

```bash
python -m sat_dag_service_deployment.simulator.cli run \
  --config sat_dag_service_deployment/configs/scenarios/sparse_topology_stress.json \
  --algorithm cpmv_dsd \
  --scheduler heft \
  --run-name sparse_topology_mv_dsd
```

Run the scenario suite:

```bash
python sat_dag_service_deployment/simulator/scripts/run_scenario_suite.py
```

## Dataset Regeneration

The included dataset can be used directly. To regenerate it from scripts,
run the following commands from the repository root:

```bash
python sat_dag_service_deployment/dag_data_set/scripts/generate_dag_dataset.py
python sat_dag_service_deployment/dag_data_set/scripts/download_eonet_reference.py --days 365
python sat_dag_service_deployment/dag_data_set/scripts/prepare_balanced_hawkes_input.py
python sat_dag_service_deployment/dag_data_set/scripts/generate_simplified_hawkes_arrivals.py
python -m sat_dag_service_deployment.simulator.cli materialize-requests
python sat_dag_service_deployment/dag_data_set/scripts/add_request_deadlines.py
```

The EONET API is continuously updated, so regenerated public-event traces
may differ from the materialized input trace included here.

## Output Files

Simulation runs write:

- `summary.json`
- `deployment_plans.json`
- `per_request_results.json`
- `manifest.json`

These files are generated under `sat_dag_service_deployment/simulator/outputs/`
and should not be committed as source code.

## Citation

If you use this simulator, please cite the corresponding MV-DSD paper.
Add the final bibliographic entry after publication.
