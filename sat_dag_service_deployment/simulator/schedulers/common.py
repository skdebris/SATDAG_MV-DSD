from __future__ import annotations

import math
from collections import defaultdict

from ..env import region_source_satellites, route_data_transfer
from ..models import DeploymentPlan, ExecutionResult, RemoteSensingDAGRequest, RouteRecord, SatelliteEnvironment, TaskExecutionRecord
from ..utils import percentile, topological_sort


def candidate_satellites_for_task(
    request: RemoteSensingDAGRequest,
    task_id: str,
    deployment_plan: DeploymentPlan,
    env: SatelliteEnvironment,
    config: dict,
) -> list[str]:
    node = request.node_map()[task_id]
    deployed = deployment_plan.service_placement.get(node.service_type, [])
    if not deployed:
        return []
    incoming = any(edge.dst == task_id for edge in request.edges)
    if incoming:
        return deployed
    source_candidates = region_source_satellites(
        env,
        request.region_id,
        replica_count=int(config.get("region_source_replica_count", 2)),
    )
    constrained = [sat_id for sat_id in deployed if sat_id in source_candidates]
    return constrained or deployed


def compute_time_days(
    request: RemoteSensingDAGRequest,
    task_id: str,
    satellite_id: str,
    deployment_plan: DeploymentPlan,
) -> float:
    node = request.node_map()[task_id]
    allocated_cpu = deployment_plan.cpu_allocation.get((satellite_id, node.service_type), 0.0)
    if allocated_cpu <= 0.0:
        return float("inf")
    speed_gflops_per_s = allocated_cpu * node.eta_gflops_per_ghz_s
    if speed_gflops_per_s <= 0.0:
        return float("inf")
    return node.workload_gflops / speed_gflops_per_s / 86400.0


def graph_views(request: RemoteSensingDAGRequest) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[tuple[str, str], float]]:
    succ = {node.task_id: [] for node in request.nodes}
    pred = {node.task_id: [] for node in request.nodes}
    edge_size: dict[tuple[str, str], float] = {}
    for edge in request.edges:
        succ[edge.src].append(edge.dst)
        pred[edge.dst].append(edge.src)
        edge_size[(edge.src, edge.dst)] = edge.data_mb
    return succ, pred, edge_size


def upward_ranks(
    request: RemoteSensingDAGRequest,
    deployment_plan: DeploymentPlan,
) -> dict[str, float]:
    succ, _, edge_size = graph_views(request)
    order = topological_sort((node.task_id for node in request.nodes), ((edge.src, edge.dst) for edge in request.edges))
    order.reverse()
    node_map = request.node_map()
    average_cpu = {}
    for node in request.nodes:
        placements = deployment_plan.service_placement.get(node.service_type, [])
        average_allocation = sum(
            deployment_plan.cpu_allocation.get((sat_id, node.service_type), 0.0)
            for sat_id in placements
        ) / max(len(placements), 1)
        speed = average_allocation * node.eta_gflops_per_ghz_s
        average_cpu[node.task_id] = node.workload_gflops / max(speed, 1e-6)
    rank = {}
    for node_id in order:
        if not succ[node_id]:
            rank[node_id] = average_cpu[node_id]
            continue
        rank[node_id] = average_cpu[node_id] + max(
            edge_size[(node_id, child)] + rank[child]
            for child in succ[node_id]
        )
    return rank


def _deadline_metadata(
    request: RemoteSensingDAGRequest,
    makespan_days: float,
    success: bool,
) -> dict:
    deadline_days = request.relative_deadline
    absolute_deadline_days = request.absolute_deadline
    if deadline_days is None and absolute_deadline_days is not None:
        deadline_days = max(0.0, absolute_deadline_days - request.arrival_time_days)
    metadata = {
        "deadline_level": request.deadline_level,
        "mission_class": request.mission_class,
        "relative_deadline_days": deadline_days,
        "relative_deadline_minutes": None if deadline_days is None else deadline_days * 24.0 * 60.0,
        "absolute_deadline_days": absolute_deadline_days,
    }
    if deadline_days is None or deadline_days <= 0.0:
        metadata.update(
            {
                "deadline_satisfied": success,
                "deadline_missed": not success,
                "normalized_tardiness": 0.0 if success else 1.0,
                "deadline_missing": True,
            }
        )
        return metadata
    deadline_satisfied = success and makespan_days <= deadline_days
    if success:
        normalized_tardiness = max(0.0, (makespan_days - deadline_days) / deadline_days)
    else:
        normalized_tardiness = 1.0
    metadata.update(
        {
            "deadline_satisfied": deadline_satisfied,
            "deadline_missed": not deadline_satisfied,
            "normalized_tardiness": normalized_tardiness,
            "deadline_missing": False,
        }
    )
    return metadata


def build_execution_result(
    request: RemoteSensingDAGRequest,
    window_id: int,
    success: bool,
    failure_reason: str | None,
    task_records: list[TaskExecutionRecord],
    route_records: list[RouteRecord],
    critical_parent: dict[str, str | None],
    critical_routes: dict[str, RouteRecord | None],
) -> ExecutionResult:
    if not task_records:
        metadata = _deadline_metadata(request=request, makespan_days=0.0, success=False)
        return ExecutionResult(
            request_id=request.request_id,
            window_id=window_id,
            success=False,
            arrival_time_days=request.arrival_time_days,
            start_time_days=request.arrival_time_days,
            finish_time_days=request.arrival_time_days,
            makespan_days=0.0,
            task_records=[],
            route_records=[],
            cp_delay_breakdown_days={"cmp": 0.0, "net": 0.0, "idle": 0.0},
            cpu_utilization={},
            energy=None,
            failure_reason=failure_reason or "no_task_records",
            metadata=metadata,
        )

    task_by_id = {record.task_id: record for record in task_records}
    start_time = min(record.start_time_days for record in task_records)
    finish_time = max(record.finish_time_days for record in task_records)
    makespan = finish_time - request.arrival_time_days
    sink = max(task_records, key=lambda record: record.finish_time_days).task_id
    cp_task_ids = [sink]
    while critical_parent.get(cp_task_ids[-1]) is not None:
        cp_task_ids.append(critical_parent[cp_task_ids[-1]])  # type: ignore[arg-type]
    cp_task_ids.reverse()
    cp_compute = sum(task_by_id[task_id].compute_time_days for task_id in cp_task_ids)
    cp_net = 0.0
    for task_id in cp_task_ids[1:]:
        route = critical_routes.get(task_id)
        if route is not None:
            cp_net += route.total_delay_days
    cp_idle = max(0.0, makespan - cp_compute - cp_net)

    per_sat_compute: dict[str, float] = defaultdict(float)
    for record in task_records:
        per_sat_compute[record.satellite_id] += record.compute_time_days
    cpu_utilization = {
        sat_id: compute_days / max(makespan, 1e-9)
        for sat_id, compute_days in per_sat_compute.items()
    }
    metadata = {
        "p95_task_finish_days": percentile([record.finish_time_days for record in task_records], 0.95),
        **_deadline_metadata(request=request, makespan_days=makespan, success=success),
    }
    return ExecutionResult(
        request_id=request.request_id,
        window_id=window_id,
        success=success,
        arrival_time_days=request.arrival_time_days,
        start_time_days=start_time,
        finish_time_days=finish_time,
        makespan_days=makespan,
        task_records=task_records,
        route_records=route_records,
        cp_delay_breakdown_days={"cmp": cp_compute, "net": cp_net, "idle": cp_idle},
        cpu_utilization=cpu_utilization,
        energy=None,
        failure_reason=failure_reason,
        metadata=metadata,
    )


def evaluate_candidate(
    request: RemoteSensingDAGRequest,
    task_id: str,
    satellite_id: str,
    deployment_plan: DeploymentPlan,
    env: SatelliteEnvironment,
    service_instance_available: dict[tuple[str, str], float],
    task_finish: dict[str, float],
    task_assignment: dict[str, str],
    config: dict,
) -> tuple[float, float, list[RouteRecord], str | None, RouteRecord | None]:
    _, predecessors, edge_size = graph_views(request)
    ready_time = request.arrival_time_days
    selected_routes: list[RouteRecord] = []
    critical_route: RouteRecord | None = None
    for pred_task in predecessors[task_id]:
        pred_sat = task_assignment[pred_task]
        route = route_data_transfer(
            environment=env,
            src_satellite_id=pred_sat,
            dst_satellite_id=satellite_id,
            data_mb=edge_size[(pred_task, task_id)],
            start_time_days=task_finish[pred_task],
            max_wait_slots=int(config.get("max_route_wait_slots", 60)),
            route_key=f"{pred_task}->{task_id}",
            src_task_id=pred_task,
            dst_task_id=task_id,
        )
        if not route.success:
            return float("inf"), float("inf"), [], route.failure_reason, None
        selected_routes.append(route)
        if route.finish_time_days > ready_time:
            ready_time = route.finish_time_days
            critical_route = route

    service_type = request.node_map()[task_id].service_type
    instance_key = (satellite_id, service_type)
    est = max(ready_time, service_instance_available.get(instance_key, request.arrival_time_days))
    compute_days = compute_time_days(request, task_id, satellite_id, deployment_plan)
    if not math.isfinite(compute_days):
        return float("inf"), float("inf"), [], "nonfinite_compute_time", None
    eft = est + compute_days
    return est, eft, selected_routes, None, critical_route
