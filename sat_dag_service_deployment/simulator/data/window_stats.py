from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable

from ..models import DeploymentWindow, DeploymentWindowStats, RemoteSensingDAGRequest
from ..utils import normalize_dict, safe_mean, topological_sort


def split_requests_into_windows(
    requests: list[RemoteSensingDAGRequest],
    deployment_period_minutes: float,
    max_windows: int | None = None,
) -> list[DeploymentWindow]:
    if not requests:
        return []
    deployment_period_days = deployment_period_minutes / (24.0 * 60.0)
    first_arrival = requests[0].arrival_time_days
    grouped: dict[int, list[RemoteSensingDAGRequest]] = defaultdict(list)
    for request in requests:
        offset = max(0.0, request.arrival_time_days - first_arrival)
        window_id = int(offset // deployment_period_days)
        grouped[window_id].append(request)

    windows: list[DeploymentWindow] = []
    for window_id in sorted(grouped):
        start = first_arrival + window_id * deployment_period_days
        end = start + deployment_period_days
        windows.append(
            DeploymentWindow(
                window_id=window_id,
                start_time_days=start,
                end_time_days=end,
                requests=sorted(grouped[window_id], key=lambda item: item.arrival_time_days),
            )
        )
        if max_windows is not None and len(windows) >= max_windows:
            break
    return windows


def _request_graph(request: RemoteSensingDAGRequest) -> tuple[dict[str, list[str]], dict[str, int], dict[tuple[str, str], float]]:
    adjacency = {node.task_id: [] for node in request.nodes}
    indegree = {node.task_id: 0 for node in request.nodes}
    edge_weights: dict[tuple[str, str], float] = {}
    for edge in request.edges:
        adjacency[edge.src].append(edge.dst)
        indegree[edge.dst] += 1
        edge_weights[(edge.src, edge.dst)] = edge.data_mb
    return adjacency, indegree, edge_weights


def _critical_path(request: RemoteSensingDAGRequest) -> tuple[list[str], dict[str, float]]:
    node_map = request.node_map()
    edges = [(edge.src, edge.dst) for edge in request.edges]
    edge_weight = {(edge.src, edge.dst): edge.data_mb / 2000.0 for edge in request.edges}
    order = topological_sort(node_map.keys(), edges)
    score = {node_id: float(node_map[node_id].workload_gflops) for node_id in node_map}
    parent: dict[str, str | None] = {node_id: None for node_id in node_map}

    for node_id in order:
        for child in [edge.dst for edge in request.edges if edge.src == node_id]:
            candidate = score[node_id] + edge_weight[(node_id, child)] + node_map[child].workload_gflops
            if candidate > score[child]:
                score[child] = candidate
                parent[child] = node_id

    sink = max(order, key=lambda item: score[item])
    path = [sink]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])
    path.reverse()
    return path, score


def _sync_weight(request: RemoteSensingDAGRequest) -> dict[str, float]:
    _, indegree, _ = _request_graph(request)
    weights: dict[str, float] = defaultdict(float)
    node_map = request.node_map()
    for node_id, degree in indegree.items():
        if degree > 1:
            weights[node_map[node_id].service_type] += node_map[node_id].workload_gflops
    return weights


def _cap_requests(requests: Iterable[RemoteSensingDAGRequest], max_requests_per_window: int | None) -> list[RemoteSensingDAGRequest]:
    requests = list(requests)
    if max_requests_per_window is None or len(requests) <= max_requests_per_window:
        return requests
    return requests[:max_requests_per_window]


def build_deployment_window_stats(
    window: DeploymentWindow,
    max_requests_per_window: int | None = None,
) -> DeploymentWindowStats:
    requests = _cap_requests(window.requests, max_requests_per_window)
    dag_type_counter = Counter(request.dag_type for request in requests)
    subtype_counter = Counter(request.subarchetype for request in requests)
    service_workload: dict[str, float] = defaultdict(float)
    service_frequency: dict[str, int] = defaultdict(int)
    service_traffic: dict[tuple[str, str], float] = defaultdict(float)
    critical_path_service_count: dict[str, float] = defaultdict(float)
    critical_path_service_weight: dict[str, float] = defaultdict(float)
    service_sync_weight: dict[str, float] = defaultdict(float)
    critical_path_lengths: list[float] = []

    for request in requests:
        node_map = request.node_map()
        for node in request.nodes:
            service_workload[node.service_type] += node.workload_gflops
            service_frequency[node.service_type] += 1
        for edge in request.edges:
            service_traffic[(edge.src_service_type, edge.dst_service_type)] += edge.data_mb
        cp_nodes, cp_scores = _critical_path(request)
        critical_path_lengths.append(max(cp_scores.values(), default=0.0))
        for node_id in cp_nodes:
            node = node_map[node_id]
            critical_path_service_count[node.service_type] += 1.0
            critical_path_service_weight[node.service_type] += node.workload_gflops
        for service_type, weight in _sync_weight(request).items():
            service_sync_weight[service_type] += weight

    width_days = max(window.end_time_days - window.start_time_days, 1e-9)
    arrival_rate = len(requests) / width_days
    return DeploymentWindowStats(
        window_id=window.window_id,
        start_time_days=window.start_time_days,
        end_time_days=window.end_time_days,
        requests=requests,
        dag_type_distribution=normalize_dict(dict(dag_type_counter)),
        subarchetype_distribution=normalize_dict(dict(subtype_counter)),
        service_workload=dict(sorted(service_workload.items())),
        service_frequency=dict(sorted(service_frequency.items())),
        service_traffic=dict(sorted(service_traffic.items())),
        critical_path_service_count=dict(sorted(critical_path_service_count.items())),
        critical_path_service_weight=dict(sorted(critical_path_service_weight.items())),
        service_sync_weight=dict(sorted(service_sync_weight.items())),
        arrival_rate_per_day=arrival_rate,
        request_count=len(requests),
        metadata={
            "avg_request_workload_gflops": safe_mean(
                [sum(node.workload_gflops for node in request.nodes) for request in requests]
            ),
            "avg_critical_path_weight": safe_mean(critical_path_lengths),
        },
    )
