from __future__ import annotations

from ..models import DeploymentPlan, DeploymentWindow, RemoteSensingDAGRequest, SatelliteEnvironment, TaskExecutionRecord
from ..utils import topological_sort
from .base import BaseScheduler
from .common import build_execution_result, candidate_satellites_for_task, evaluate_candidate, graph_views


class PriorityScheduler(BaseScheduler):
    name = "priority"

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
        topo_index = {task_id: idx for idx, task_id in enumerate(topo)}
        node_map = request.node_map()
        unscheduled = set(topo)
        ready = {task_id for task_id in topo if not pred[task_id]}
        task_records = []
        route_records = []
        task_finish: dict[str, float] = {}
        task_assignment: dict[str, str] = {}
        critical_parent: dict[str, str | None] = {}
        critical_routes = {}

        while unscheduled:
            if not ready:
                return build_execution_result(
                    request=request,
                    window_id=self.window_id,
                    success=False,
                    failure_reason="priority_ready_set_empty",
                    task_records=task_records,
                    route_records=route_records,
                    critical_parent=critical_parent,
                    critical_routes=critical_routes,
                )
            task_id = min(
                ready,
                key=lambda task: (
                    -node_map[task].workload_gflops,
                    -len(succ[task]),
                    topo_index[task],
                ),
            )
            candidates = candidate_satellites_for_task(request, task_id, deployment_plan, env, config)
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
                    failure_reason="priority_scheduler_no_candidate",
                    task_records=task_records,
                    route_records=route_records,
                    critical_parent=critical_parent,
                    critical_routes=critical_routes,
                )
            eft, est, satellite_id = best
            unscheduled.remove(task_id)
            ready.remove(task_id)
            task_finish[task_id] = eft
            task_assignment[task_id] = satellite_id
            self.service_instance_available[(satellite_id, node_map[task_id].service_type)] = eft
            route_records.extend(best_routes)
            task_records.append(
                TaskExecutionRecord(
                    request_id=request.request_id,
                    task_id=task_id,
                    service_type=node_map[task_id].service_type,
                    satellite_id=satellite_id,
                    start_time_days=est,
                    finish_time_days=eft,
                    compute_time_days=eft - est,
                    predecessor_ready_time_days=max([request.arrival_time_days] + [route.finish_time_days for route in best_routes]),
                    scheduler_priority=node_map[task_id].workload_gflops + len(succ[task_id]),
                    selected_reason="priority_weighted_task_first",
                )
            )
            critical_parent[task_id] = best_critical_route.src_task_id if best_critical_route is not None else (pred[task_id][0] if pred[task_id] else None)
            critical_routes[task_id] = best_critical_route
            for child in succ[task_id]:
                if child in unscheduled and all(parent in task_finish for parent in pred[child]):
                    ready.add(child)

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
