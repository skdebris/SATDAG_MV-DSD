from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT_DIR / "dag_data_set" / "dataset"
SIM_OUTPUT_DIR = ROOT_DIR / "simulator" / "outputs"


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 20260427,
    "data": {
        "service_catalog_path": str(DATASET_DIR / "service_library.json"),
        "template_dir": str(DATASET_DIR / "dag_templates"),
        "arrival_trace_path": str(DATASET_DIR / "arrival_traces" / "simulated_arrivals_balanced_T30days.json"),
        "job_requests_path": str(DATASET_DIR / "job_requests" / "requests_T30days.jsonl"),
    },
    "simulation": {
        "deployment_period_minutes": 95.0,
        "deployment_mode": "fixed",
        "max_windows": 6,
        "max_requests_per_window": 32,
        "planning_window_count": None,
        "planning_max_requests": None,
        "planning_reference_window_id": 0,
        "request_source_mode": "materialized",
        "persist_materialized_requests": True,
        "t_max_minutes": 30.0,
    },
    "environment": {
        "seed": None,
        "num_planes": 8,
        "satellites_per_plane": 10,
        "altitude_km": 550.0,
        "inclination_deg": 53.0,
        "planning_horizon_slots": 60,
        "planning_horizon_minutes": 95.0,
        "density_mode": "normal",
        "perturbation_mode": "low",
        "aggregate_kappa": 0.25,
        "crosslink_capacity_mbps": 1600.0,
        "inplane_capacity_mbps": 2000.0,
        "link_drop_probability": {
            "dense": 0.03,
            "normal": 0.08,
            "sparse": 0.18,
        },
        "sparse_crosslink_stride": 3,
        "sparse_crosslink_drop_bonus": 0.35,
        "link_drop_cap": 0.65,
    },
    "algorithm": {
        "name": "cpmv_dsd",
        "seed": None,
        "sample_size": 200,
        "use_structured_value": True,
        "use_structured_deployment": True,
        "use_role_matching_deployment": True,
        "use_stratified_sampling": True,
        "stratified_weight_clip_ratio": 0.0,
        "use_comm_graph_constraint": True,
        "use_topology_features": True,
        "use_cvar": False,
        "cvar_lambda": 0.3,
        "cvar_alpha": 0.95,
        "cvar_risk_amplifier": 1.0,
        "ondoc_planning_mode": "statistical",
        "ondoc_replay_request_limit": 32,
        "max_replicas_per_service": 5,
        "min_replicas_per_service": 1,
        "replica_scale_factor": 0.18,
        "comm_graph_bridge_bonus": 0.08,
    },
    "scheduler": {
        "name": "heft",
        "max_route_wait_slots": 60,
        "region_source_replica_count": 2,
    },
    "output": {
        "output_dir": str(SIM_OUTPUT_DIR),
        "run_name": "sample_run",
        "save_per_request_results": True,
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    if path is not None:
        path = Path(path)
        loaded = json.loads(path.read_text(encoding="utf-8"))
        config = _deep_merge(config, loaded)
    if overrides:
        config = _deep_merge(config, overrides)
    return config
