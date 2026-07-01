from __future__ import annotations

from ..models import DeploymentPlan, DeploymentWindow, RemoteSensingDAGRequest, SatelliteEnvironment, TaskExecutionRecord
from ..utils import topological_sort
from .base import BaseScheduler
from .common import (
    build_execution_result,
    candidate_satellites_for_task,
    evaluate_candidate,
    graph_views,
    upward_ranks,
)


class HEFTScheduler(BaseScheduler):
    name = "heft"

    def __init__(self) -> None:
        self.window_id = -1
        self.service_instance_available: dict[tuple[str, str], float] = {}

    def reset_window(self, window: DeploymentWindow) -> None:
        self.window_id = window.window_id
        self.service_instance_available = {}

    def schedule(
        self,
        request: RemoteSensingDAGRequest,
        deployment_plan: DeploymentPlan,
        env: SatelliteEnvironment,
        current_time: float,
        config: dict,
    ):
        succ, pred, _ = graph_views(request)
        topo = topological_sort((node.task_id for node in request.nodes), ((edge.src, edge.dst) for edge in request.edges))
        ranks = upward_ranks(request, deployment_plan)
        order = sorted(topo, key=lambda task_id: (-ranks[task_id], topo.index(task_id)))

        task_records = []
        route_records = []
        task_finish: dict[str, float] = {}
        task_assignment: dict[str, str] = {}
        critical_parent: dict[str, str | None] = {}
        critical_routes = {}

        for task_id in order:
            if any(parent not in task_finish for parent in pred[task_id]):
                return build_execution_result(
                    request=request,
                    window_id=self.window_id,
                    success=False,
                    failure_reason="predecessor_not_scheduled",
                    task_records=task_records,
                    route_records=route_records,
                    critical_parent=critical_parent,
                    critical_routes=critical_routes,
                )
            candidates = candidate_satellites_for_task(request, task_id, deployment_plan, env, config)
            if not candidates:
                return build_execution_result(
                    request=request,
                    window_id=self.window_id,
                    success=False,
                    failure_reason=f"no_deployed_service_for_{request.node_map()[task_id].service_type}",
                    task_records=task_records,
                    route_records=route_records,
                    critical_parent=critical_parent,
                    critical_routes=critical_routes,
                )

            best = None
            best_routes = []
            best_critical_route = None
            for satellite_id in candidates:
                est, eft, routes, failure_reason, critical_route = evaluate_candidate(
                    request=request,
                    task_id=task_id,
                    satellite_id=satellite_id,
                    deployment_plan=deployment_plan,
                    env=env,
                    service_instance_available=self.service_instance_available,
                    task_finish=task_finish,
                    task_assignment=task_assignment,
                    config=config,
                )
                if failure_reason is not None:
                    continue
                candidate = (eft, est, satellite_id)
                if best is None or candidate < best:
                    best = candidate
                    best_routes = routes
                    best_critical_route = critical_route
            if best is None:
                return build_execution_result(
                    request=request,
                    window_id=self.window_id,
                    success=False,
                    failure_reason="no_feasible_satellite_route_pair",
                    task_records=task_records,
                    route_records=route_records,
                    critical_parent=critical_parent,
                    critical_routes=critical_routes,
                )
            eft, est, satellite_id = best
            task_finish[task_id] = eft
            task_assignment[task_id] = satellite_id
            node = request.node_map()[task_id]
            self.service_instance_available[(satellite_id, node.service_type)] = eft
            for route in best_routes:
                route_records.append(route)
            task_records.append(
                TaskExecutionRecord(
                    request_id=request.request_id,
                    task_id=task_id,
                    service_type=node.service_type,
                    satellite_id=satellite_id,
                    start_time_days=est,
                    finish_time_days=eft,
                    compute_time_days=eft - est,
                    predecessor_ready_time_days=max([request.arrival_time_days] + [route.finish_time_days for route in best_routes]),
                    scheduler_priority=ranks[task_id],
                    selected_reason="min_eft_under_heft",
                )
            )
            if best_critical_route is None:
                critical_parent[task_id] = pred[task_id][0] if pred[task_id] else None
            else:
                critical_parent[task_id] = best_critical_route.src_task_id
            critical_routes[task_id] = best_critical_route

        return build_execution_result(
            request=request,
            window_id=self.window_id,
            success=True,
            failure_reason=None,
            task_records=task_records,
            route_records=route_records,
            critical_parent=critical_parent,
            critical_routes=critical_routes,
        )
