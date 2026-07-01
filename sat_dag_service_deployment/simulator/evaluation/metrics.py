from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from ..models import DeploymentPlan, ExecutionResult
from ..utils import percentile, safe_mean


def _days_to_minutes(value: float) -> float:
    return value * 24.0 * 60.0


def summarize_results(
    execution_results: list[ExecutionResult],
    deployment_plans: list[DeploymentPlan],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = metadata or {}
    request_count = len(execution_results)
    completed = [result for result in execution_results if result.success]
    deadlines_present = any(
        not result.metadata.get("deadline_missing", True)
        for result in execution_results
    )
    deadline_satisfied = [
        result
        for result in execution_results
        if bool(result.metadata.get("deadline_satisfied", False))
    ]
    deadline_missed = [
        result
        for result in execution_results
        if not bool(result.metadata.get("deadline_satisfied", False))
    ]
    normalized_tardiness = [
        float(result.metadata.get("normalized_tardiness", 0.0))
        for result in execution_results
    ]
    makespans = [result.makespan_days for result in completed]
    on_time_makespans = [result.makespan_days for result in deadline_satisfied]
    cp_cmp = [result.cp_delay_breakdown_days["cmp"] for result in completed]
    cp_net = [result.cp_delay_breakdown_days["net"] for result in completed]
    cp_idle = [result.cp_delay_breakdown_days["idle"] for result in completed]
    cross_sat_routes = [
        route
        for result in completed
        for route in result.route_records
        if route.src_satellite_id != route.dst_satellite_id and route.success
    ]
    service_replica_samples: dict[str, list[int]] = defaultdict(list)
    for plan in deployment_plans:
        for service, replicas in plan.replica_count.items():
            service_replica_samples[service].append(replicas)

    raw_completion_rate = len(completed) / max(request_count, 1)
    deadline_satisfaction_ratio = len(deadline_satisfied) / max(request_count, 1)
    task_completion_rate = deadline_satisfaction_ratio if deadlines_present else raw_completion_rate
    completed_count = len(deadline_satisfied) if deadlines_present else len(completed)
    failed_count = request_count - completed_count

    summary = {
        "request_count": request_count,
        "completed_count": completed_count,
        "failed_count": failed_count,
        "task_completion_rate": task_completion_rate,
        "deadline_satisfaction_ratio": deadline_satisfaction_ratio if deadlines_present else raw_completion_rate,
        "deadline_satisfied_count": len(deadline_satisfied) if deadlines_present else len(completed),
        "deadline_missed_count": len(deadline_missed) if deadlines_present else request_count - len(completed),
        "deadline_miss_rate": len(deadline_missed) / max(request_count, 1) if deadlines_present else 1.0 - raw_completion_rate,
        "mean_normalized_tardiness": safe_mean(normalized_tardiness) if deadlines_present else 0.0,
        "raw_completed_count": len(completed),
        "raw_failed_count": request_count - len(completed),
        "raw_task_completion_rate": raw_completion_rate,
        "mean_makespan_minutes": _days_to_minutes(safe_mean(makespans)),
        "mean_on_time_makespan_minutes": _days_to_minutes(safe_mean(on_time_makespans)),
        "p95_makespan_minutes": _days_to_minutes(percentile(makespans, 0.95)),
        "p99_makespan_minutes": _days_to_minutes(percentile(makespans, 0.99)),
        "mean_cp_cmp_minutes": _days_to_minutes(safe_mean(cp_cmp)),
        "mean_cp_net_minutes": _days_to_minutes(safe_mean(cp_net)),
        "mean_cp_idle_minutes": _days_to_minutes(safe_mean(cp_idle)),
        "cross_sat_route_count": len(cross_sat_routes),
        "cross_sat_traffic_mb": sum(route.data_mb for route in cross_sat_routes),
        "failure_reasons": dict(
            Counter(result.failure_reason for result in execution_results if result.failure_reason)
        ),
        "replica_statistics": {
            service: {
                "mean": safe_mean(replica_counts),
                "max": max(replica_counts),
                "min": min(replica_counts),
            }
            for service, replica_counts in sorted(service_replica_samples.items())
        },
        "metadata": metadata,
    }
    return summary
