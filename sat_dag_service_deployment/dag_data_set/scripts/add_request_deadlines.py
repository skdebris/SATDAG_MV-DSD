from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PACKAGE_ROOT / "dag_data_set" / "dataset"
REQUESTS_PATH = DATASET_DIR / "job_requests" / "requests_T30days.jsonl"
SUMMARY_PATH = DATASET_DIR / "job_requests" / "requests_T30days_deadline_summary.json"
BACKUP_PATH = DATASET_DIR / "job_requests" / "requests_T30days.before_deadline.jsonl"
NORMAL_SCENARIO_PATH = PACKAGE_ROOT / "configs" / "scenarios" / "normal_nominal.json"

RNG_SEED = 20260427
REFERENCE_PERCENTILE = 75.0
DEADLINE_LEVEL_PROBS = (
    ("tight", 0.30),
    ("moderate", 0.50),
    ("loose", 0.20),
)
ETA_BY_LEVEL = {
    "tight": 1.6,
    "moderate": 2.2,
    "loose": 3.0,
}
PILOT_CALIBRATION_FACTOR = 3.5
DAG_FACTOR = {
    "chain_like": 0.95,
    "wide_shallow": 1.10,
    "general": 1.05,
}
MISSION_CLASS_BY_SUBARCHETYPE = {
    "1A": "ship_detection",
    "1B": "disaster_monitoring",
    "2A": "weather_monitoring",
    "2B": "resource_appraisal",
    "3A": "cooperative_monitoring",
    "3B": "disaster_monitoring",
    "3C": "weather_monitoring",
}


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * q / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def topological_order(nodes: Iterable[str], edges: Iterable[tuple[str, str]]) -> list[str]:
    nodes = list(nodes)
    adjacency = {node: [] for node in nodes}
    indegree = {node: 0 for node in nodes}
    for src, dst in edges:
        adjacency[src].append(dst)
        indegree[dst] += 1
    ready = sorted(node for node, degree in indegree.items() if degree == 0)
    order: list[str] = []
    while ready:
        node = ready.pop(0)
        order.append(node)
        for child in sorted(adjacency[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
                ready.sort()
    if len(order) != len(nodes):
        raise ValueError("DAG request contains a cycle.")
    return order


def load_reference_capacities() -> tuple[float, float, dict[str, Any]]:
    sys.path.insert(0, str(PROJECT_ROOT))
    from sat_dag_service_deployment.simulator.config import load_config
    from sat_dag_service_deployment.simulator.env import build_satellite_environment

    config = load_config(NORMAL_SCENARIO_PATH)
    env = build_satellite_environment(
        config=config["environment"],
        window_id=0,
        seed=int(config["seed"]),
    )
    cpu_ghz = [sat.cpu_capacity_ghz for sat in env.satellites.values()]
    link_mbps = [
        link.capacity_mbps
        for snapshot in env.snapshots
        for links in snapshot.adjacency.values()
        for link in links
    ]
    c_ref_gflops_per_s = percentile(cpu_ghz, REFERENCE_PERCENTILE)
    b_ref_mbps = percentile(link_mbps, REFERENCE_PERCENTILE)
    metadata = {
        "reference_scenario": str(NORMAL_SCENARIO_PATH.relative_to(PACKAGE_ROOT)),
        "reference_percentile": REFERENCE_PERCENTILE,
        "c_ref_gflops_per_s": c_ref_gflops_per_s,
        "b_ref_mbps": b_ref_mbps,
        "b_ref_mb_per_s": b_ref_mbps / 8.0,
        "num_reference_satellites": len(cpu_ghz),
        "num_reference_directed_links": len(link_mbps),
    }
    return c_ref_gflops_per_s, b_ref_mbps / 8.0, metadata


def critical_path_lower_bound(
    request: dict[str, Any],
    c_ref_gflops_per_s: float,
    b_ref_mb_per_s: float,
) -> dict[str, float]:
    nodes = {node["task_id"]: node for node in request["nodes"]}
    edges = [(edge["src"], edge["dst"]) for edge in request["edges"]]
    edge_data = {(edge["src"], edge["dst"]): float(edge["data_mb"]) for edge in request["edges"]}
    successors: dict[str, list[str]] = {task_id: [] for task_id in nodes}
    predecessors: dict[str, list[str]] = {task_id: [] for task_id in nodes}
    for src, dst in edges:
        successors[src].append(dst)
        predecessors[dst].append(src)

    order = topological_order(nodes, edges)
    cmp_seconds = {
        task_id: float(node["workload_gflops"]) / max(c_ref_gflops_per_s, 1e-9)
        for task_id, node in nodes.items()
    }
    edge_seconds = {
        edge: data_mb / max(b_ref_mb_per_s, 1e-9)
        for edge, data_mb in edge_data.items()
    }
    best_total = {}
    best_cmp = {}
    best_net = {}
    best_len = {}
    parent: dict[str, str | None] = {}

    for task_id in order:
        if not predecessors[task_id]:
            best_total[task_id] = cmp_seconds[task_id]
            best_cmp[task_id] = cmp_seconds[task_id]
            best_net[task_id] = 0.0
            best_len[task_id] = 1
            parent[task_id] = None
            continue
        candidates = []
        for pred in predecessors[task_id]:
            total = best_total[pred] + edge_seconds[(pred, task_id)] + cmp_seconds[task_id]
            candidates.append(
                (
                    total,
                    best_cmp[pred] + cmp_seconds[task_id],
                    best_net[pred] + edge_seconds[(pred, task_id)],
                    best_len[pred] + 1,
                    pred,
                )
            )
        total, cmp_part, net_part, path_len, pred = max(candidates, key=lambda item: item[0])
        best_total[task_id] = total
        best_cmp[task_id] = cmp_part
        best_net[task_id] = net_part
        best_len[task_id] = path_len
        parent[task_id] = pred

    sinks = [task_id for task_id in order if not successors[task_id]]
    sink = max(sinks or order, key=lambda task_id: best_total[task_id])
    return {
        "cp_workload_lb_seconds": best_cmp[sink],
        "cp_data_lb_seconds": best_net[sink],
        "lower_bound_seconds": best_total[sink],
        "critical_path_len": float(best_len[sink]),
    }


def choose_deadline_level(rng: random.Random) -> str:
    selector = rng.random()
    cumulative = 0.0
    for level, probability in DEADLINE_LEVEL_PROBS:
        cumulative += probability
        if selector <= cumulative:
            return level
    return DEADLINE_LEVEL_PROBS[-1][0]


def enrich_request(
    request: dict[str, Any],
    rng: random.Random,
    c_ref_gflops_per_s: float,
    b_ref_mb_per_s: float,
    reference_metadata: dict[str, Any],
) -> dict[str, Any]:
    lb = critical_path_lower_bound(request, c_ref_gflops_per_s, b_ref_mb_per_s)
    deadline_level = choose_deadline_level(rng)
    eta = ETA_BY_LEVEL[deadline_level]
    dag_factor = DAG_FACTOR.get(request["dag_type"], 1.05)
    jitter = rng.uniform(0.9, 1.1)
    relative_deadline_seconds = eta * dag_factor * lb["lower_bound_seconds"] * jitter * PILOT_CALIBRATION_FACTOR
    relative_deadline_days = relative_deadline_seconds / 86400.0
    absolute_deadline_days = float(request["arrival_time_days"]) + relative_deadline_days

    request["mission_class"] = MISSION_CLASS_BY_SUBARCHETYPE.get(
        request.get("subarchetype", ""),
        "general_remote_sensing",
    )
    request["num_nodes"] = len(request["nodes"])
    request["num_edges"] = len(request["edges"])
    request["critical_path_len"] = int(lb["critical_path_len"])
    request["cp_workload_lb"] = lb["cp_workload_lb_seconds"] / 86400.0
    request["cp_data_lb"] = lb["cp_data_lb_seconds"] / 86400.0
    request["lower_bound_time"] = lb["lower_bound_seconds"] / 86400.0
    request["deadline_level"] = deadline_level
    request["deadline_slack_eta"] = eta
    request["dag_slack_factor"] = dag_factor
    request["jitter_factor"] = jitter
    request["relative_deadline"] = relative_deadline_days
    request["absolute_deadline"] = absolute_deadline_days
    request["cp_workload_lb_minutes"] = lb["cp_workload_lb_seconds"] / 60.0
    request["cp_data_lb_minutes"] = lb["cp_data_lb_seconds"] / 60.0
    request["lower_bound_time_minutes"] = lb["lower_bound_seconds"] / 60.0
    request["relative_deadline_minutes"] = relative_deadline_seconds / 60.0
    request["absolute_deadline_days"] = absolute_deadline_days
    request["deadline_metadata"] = {
        "method": "critical_path_slack_factor",
        "time_unit_for_relative_deadline": "days",
        "time_unit_for_absolute_deadline": "days",
        "time_unit_for_lower_bound_time": "days",
        "deadline_formula": "D = eta_level * xi_dag * LB * jitter",
        "pilot_calibration_factor": PILOT_CALIBRATION_FACTOR,
        **reference_metadata,
    }
    return request


def load_requests(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_requests(path: Path, requests: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for request in requests:
            handle.write(json.dumps(request, ensure_ascii=False) + "\n")


def summarize(requests: list[dict[str, Any]], reference_metadata: dict[str, Any]) -> dict[str, Any]:
    deadlines = [float(request["relative_deadline_minutes"]) for request in requests]
    lower_bounds = [float(request["lower_bound_time_minutes"]) for request in requests]
    ratios = [
        float(request["relative_deadline_minutes"]) / max(float(request["lower_bound_time_minutes"]), 1e-9)
        for request in requests
    ]
    return {
        "request_count": len(requests),
        "deadline_generation_seed": RNG_SEED,
        "deadline_level_distribution": dict(sorted(Counter(request["deadline_level"] for request in requests).items())),
        "mission_class_distribution": dict(sorted(Counter(request["mission_class"] for request in requests).items())),
        "dag_type_distribution": dict(sorted(Counter(request["dag_type"] for request in requests).items())),
        "relative_deadline_minutes": {
            "min": min(deadlines),
            "mean": mean(deadlines),
            "p50": percentile(deadlines, 50.0),
            "p75": percentile(deadlines, 75.0),
            "p95": percentile(deadlines, 95.0),
            "max": max(deadlines),
        },
        "lower_bound_time_minutes": {
            "min": min(lower_bounds),
            "mean": mean(lower_bounds),
            "p50": percentile(lower_bounds, 50.0),
            "p75": percentile(lower_bounds, 75.0),
            "p95": percentile(lower_bounds, 95.0),
            "max": max(lower_bounds),
        },
        "deadline_to_lower_bound_ratio": {
            "min": min(ratios),
            "mean": mean(ratios),
            "p50": percentile(ratios, 50.0),
            "p75": percentile(ratios, 75.0),
            "p95": percentile(ratios, 95.0),
            "max": max(ratios),
        },
        "reference_capacity": reference_metadata,
        "pilot_calibration_factor": PILOT_CALIBRATION_FACTOR,
        "calibrated_deadline_formula": "D = eta_level * xi_dag * LB * jitter * pilot_calibration_factor",
        "field_units": {
            "cp_workload_lb": "days",
            "cp_data_lb": "days",
            "lower_bound_time": "days",
            "relative_deadline": "days",
            "absolute_deadline": "days",
            "*_minutes": "minutes",
        },
    }


def main() -> None:
    if not REQUESTS_PATH.exists():
        raise FileNotFoundError(REQUESTS_PATH)
    if not BACKUP_PATH.exists():
        BACKUP_PATH.write_text(REQUESTS_PATH.read_text(encoding="utf-8"), encoding="utf-8")

    c_ref, b_ref, reference_metadata = load_reference_capacities()
    rng = random.Random(RNG_SEED)
    requests = load_requests(REQUESTS_PATH)
    enriched = [
        enrich_request(
            request=request,
            rng=rng,
            c_ref_gflops_per_s=c_ref,
            b_ref_mb_per_s=b_ref,
            reference_metadata=reference_metadata,
        )
        for request in requests
    ]
    write_requests(REQUESTS_PATH, enriched)
    summary = summarize(enriched, reference_metadata)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Updated requests: {REQUESTS_PATH}")
    print(f"Backup: {BACKUP_PATH}")
    print(f"Summary: {SUMMARY_PATH}")
    print(
        "Deadline minutes: "
        f"mean={summary['relative_deadline_minutes']['mean']:.3f}, "
        f"p50={summary['relative_deadline_minutes']['p50']:.3f}, "
        f"p95={summary['relative_deadline_minutes']['p95']:.3f}"
    )


if __name__ == "__main__":
    main()
