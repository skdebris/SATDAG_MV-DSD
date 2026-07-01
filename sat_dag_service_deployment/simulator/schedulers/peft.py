from __future__ import annotations

from ..models import DeploymentPlan, DeploymentWindow, RemoteSensingDAGRequest, SatelliteEnvironment, TaskExecutionRecord
from ..utils import topological_sort
from .base import BaseScheduler
from .common import (
    build_execution_result,
    candidate_satellites_for_task,
    compute_time_days,
    evaluate_candidate,
    graph_views,
)


class PEFTScheduler(BaseScheduler):
    name = "peft"

    def __init__(self) -> None:
        self.window_id = -1
        self.service_instance_available: dict[tuple[str, str], float] = {}

    def reset_window(self, window: DeploymentWindow) -> None:
        self.window_id = window.window_id
        self.service_instance_available = {}

    def _optimistic_cost_table(
        self,
        request: RemoteSensingDAGRequest,
        deployment_plan: DeploymentPlan,
        env: SatelliteEnvironment,
        config: dict,
    ) -> dict[tuple[str, str], float]:
        succ, _, edge_size = graph_views(request)
        order = topological_sort((node.task_id for node in request.nodes), ((edge.src, edge.dst) for edge in request.edges))
        order.reverse()
        all_sats = list(env.satellites)
        service_candidates = {
            node.task_id: deployment_plan.service_placement.get(node.service_type, [])
            for node in request.nodes
        }
        oct_table: dict[tuple[str, str], float] = {}
        average_bandwidth = float(config.get("peft_average_bandwidth_mbps", 1600.0))
        for task_id in order:
            for sat_id in all_sats:
                if not succ[task_id]:
                    oct_table[(task_id, sat_id)] = 0.0
                    continue
                child_costs = []
                for child_id in succ[task_id]:
                    best_child = float("inf")
                    for child_sat in service_candidates.get(child_id, []):
                        child_compute = compute_time_days(request, child_id, child_sat, deployment_plan)
                        if not child_compute < float("inf"):
                            continue
                        data_mb = edge_size[(task_id, child_id)]
                        comm_days = 0.0 if child_sat == sat_id else (data_mb * 8.0 / average_bandwidth) / 86400.0
                        best_child = min(best_child, comm_days + child_compute + oct_table.get((child_id, child_sat), 0.0))
                    child_costs.append(best_child)
                oct_table[(task_id, sat_id)] = max(child_costs) if child_costs else 0.0
        return oct_table

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
        oct_table = self._optimistic_cost_table(request, deployment_plan, env, config)
        rank_oct = {
            task_id: sum(oct_table.get((task_id, sat_id), 0.0) for sat_id in env.satellites) / max(len(env.satellites), 1)
            for task_id in topo
        }

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
                    failure_reason="peft_ready_set_empty",
                    task_records=task_records,
                    route_records=route_records,
                    critical_parent=critical_parent,
                    critical_routes=critical_routes,
                )
            task_id = min(ready, key=lambda task: (-rank_oct[task], topo_index[task]))
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
                candidate = (eft + oct_table.get((task_id, satellite_id), 0.0), eft, est, satellite_id)
                if best is None or candidate < best:
                    best = candidate
                    best_routes = routes
                    best_critical_route = critical_route
            if best is None:
                return build_execution_result(
                    request=request,
                    window_id=self.window_id,
                    success=False,
                    failure_reason="peft_no_feasible_candidate",
                    task_records=task_records,
                    route_records=route_records,
                    critical_parent=critical_parent,
                    critical_routes=critical_routes,
                )
            _, eft, est, satellite_id = best
            unscheduled.remove(task_id)
            ready.remove(task_id)
            task_finish[task_id] = eft
            task_assignment[task_id] = satellite_id
            node = request.node_map()[task_id]
            self.service_instance_available[(satellite_id, node.service_type)] = eft
            route_records.extend(best_routes)
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
                    scheduler_priority=rank_oct[task_id],
                    selected_reason="min_oeft_under_peft",
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
