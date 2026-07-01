from __future__ import annotations

import json
from pathlib import Path

from ..models import DependencyEdge, RemoteSensingDAGRequest, ServiceSpec, TaskNode
from ..utils import load_json


def load_service_catalog(path: str | Path) -> dict[str, ServiceSpec]:
    payload = load_json(path)
    services = {}
    for item in payload["services"]:
        services[item["short_name"]] = ServiceSpec(
            service_id=item["service_id"],
            short_name=item["short_name"],
            name=item["name"],
            processing_level=item["processing_level"],
            memory_mb=float(item["memory_MB"]),
            storage_mb=float(item["storage_MB"]),
            eta_gflops_per_ghz_s=float(item["eta_gflops_per_ghz_s"]),
            workload_distribution=item.get("workload_distribution", {}),
        )
    return services


def load_template_library(template_dir: str | Path) -> dict[str, dict]:
    template_dir = Path(template_dir)
    library: dict[str, dict] = {}
    for path in sorted(template_dir.glob("*/*.json")):
        if path.name == "dag_instances_summary.json":
            continue
        payload = load_json(path)
        library[payload["instance_id"]] = payload
    return library


def load_arrival_trace(path: str | Path) -> list[dict]:
    return load_json(path)


def _build_request(payload: dict) -> RemoteSensingDAGRequest:
    nodes = [
        TaskNode(
            task_id=node["task_id"],
            service_type=node["service_type"],
            service_id=node["service_id"],
            service_name=node["service_name"],
            processing_level=node["processing_level"],
            workload_gflops=float(node["workload_gflops"]),
            memory_mb=float(node["memory_mb"]),
            storage_mb=float(node["storage_mb"]),
            eta_gflops_per_ghz_s=float(node["eta_gflops_per_ghz_s"]),
            input_size_mb=float(node.get("input_size_mb", 0.0)),
            output_size_mb=float(node.get("output_size_mb", 0.0)),
            metadata=node.get("metadata", {}),
        )
        for node in payload["nodes"]
    ]
    edges = [
        DependencyEdge(
            src=edge["src"],
            dst=edge["dst"],
            data_mb=float(edge["data_mb"]),
            src_service_type=edge["src_service_type"],
            dst_service_type=edge["dst_service_type"],
        )
        for edge in payload["edges"]
    ]
    return RemoteSensingDAGRequest(
        request_id=payload["request_id"],
        event_id=payload["event_id"],
        region_id=payload["region_id"],
        arrival_time_days=float(payload["arrival_time_days"]),
        arrival_timestamp_utc=payload["arrival_timestamp_utc"],
        dag_type=payload["dag_type"],
        subarchetype=payload["subarchetype"],
        dag_instance_id=payload["dag_instance_id"],
        source_modalities=payload.get("source_modalities", []),
        nodes=nodes,
        edges=edges,
        mission_class=payload.get("mission_class"),
        num_nodes=payload.get("num_nodes"),
        num_edges=payload.get("num_edges"),
        critical_path_len=payload.get("critical_path_len"),
        cp_workload_lb=payload.get("cp_workload_lb"),
        cp_data_lb=payload.get("cp_data_lb"),
        lower_bound_time=payload.get("lower_bound_time"),
        deadline_level=payload.get("deadline_level"),
        deadline_slack_eta=payload.get("deadline_slack_eta"),
        dag_slack_factor=payload.get("dag_slack_factor"),
        jitter_factor=payload.get("jitter_factor"),
        relative_deadline=payload.get("relative_deadline"),
        absolute_deadline=payload.get("absolute_deadline"),
        cp_workload_lb_minutes=payload.get("cp_workload_lb_minutes"),
        cp_data_lb_minutes=payload.get("cp_data_lb_minutes"),
        lower_bound_time_minutes=payload.get("lower_bound_time_minutes"),
        relative_deadline_minutes=payload.get("relative_deadline_minutes"),
        absolute_deadline_days=payload.get("absolute_deadline_days"),
        metadata=payload.get("metadata", {}),
    )


def load_materialized_requests(path: str | Path) -> list[RemoteSensingDAGRequest]:
    requests: list[RemoteSensingDAGRequest] = []
    path = Path(path)
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        requests.append(_build_request(json.loads(line)))
    requests.sort(key=lambda request: request.arrival_time_days)
    return requests
