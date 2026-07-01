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


class CCPMDAGScheduler(BaseScheduler):
    name = "ccp_mdag"

    def __init__(self) -> None:
        self.window_id = -1
        self.service_instance_available: dict[tuple[str, str], float] = {}
        self.public_cache: set[tuple[str, str, str]] = set()

    def reset_window(self, window: DeploymentWindow) -> None:
        self.window_id = window.window_id
        self.service_instance_available = {}
        self.public_cache = set()

    def _task_signature(self, request: RemoteSensingDAGRequest, task_id: str) -> tuple[str, str, str]:
        node = request.node_map()[task_id]
        template_node = str(node.metadata.get("template_node_id", task_id))
        return (node.service_type, request.region_id, template_node)

    def _composite_ranks(
        self,
        request: RemoteSensingDAGRequest,
        deployment_plan: DeploymentPlan,
        env: SatelliteEnvironment,
        config: dict,
    ) -> dict[str, float]:
        succ, _, edge_size = graph_views(request)
        order = topological_sort((node.task_id for node in request.nodes), ((edge.src, edge.dst) for edge in request.edges))
        order.reverse()
        alpha = float(config.get("ccp_alpha", 0.85))
        beta = float(config.get("ccp_beta", 0.15))
        average_bandwidth = float(config.get("ccp_average_bandwidth_mbps", 1600.0))
        ranks: dict[str, float] = {}
        for task_id in order:
            placements = deployment_plan.service_placement.get(request.node_map()[task_id].service_type, [])
            comp_samples = [
                compute_time_days(request, task_id, sat_id, deployment_plan)
                for sat_id in placements
            ]
            comp_samples = [value for value in comp_samples if value < float("inf")]
            avg_exec = sum(comp_samples) / max(len(comp_samples), 1)
            avg_energy = avg_exec * max(request.node_map()[task_id].workload_gflops, 1.0)
            if not succ[task_id]:
                ranks[task_id] = alpha * avg_exec + beta * avg_energy
                continue
            successor_cost = max(
                (edge_size[(task_id, child)] * 8.0 / average_bandwidth) / 86400.0 + ranks[child]
                for child in succ[task_id]
            )
            ranks[task_id] = alpha * avg_exec + beta * avg_energy + successor_cost
        return ranks

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
        ranks = self._composite_ranks(request, deployment_plan, env, config)
        enable_cache = bool(config.get("enable_public_cache", False))
        cache_gain = float(config.get("ccp_cache_priority_bonus", 0.05))

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
                    failure_reason="ccp_ready_set_empty",
                    task_records=task_records,
                    route_records=route_records,
                    critical_parent=critical_parent,
                    critical_routes=critical_routes,
                )
            task_id = min(
                ready,
                key=lambda task: (
                    -(ranks[task] + (cache_gain if self._task_signature(request, task) in self.public_cache else 0.0)),
                    topo_index[task],
                ),
            )
            signature = self._task_signature(request, task_id)
            if enable_cache and signature in self.public_cache and pred[task_id]:
                parent_finish = max(task_finish[parent] for parent in pred[task_id])
                cache_days = float(config.get("ccp_cache_access_seconds", 0.05)) / 86400.0
                finish = parent_finish + cache_days
                satellite_id = task_assignment[pred[task_id][0]]
                task_finish[task_id] = finish
                task_assignment[task_id] = satellite_id
                unscheduled.remove(task_id)
                ready.remove(task_id)
                node = request.node_map()[task_id]
                task_records.append(
                    TaskExecutionRecord(
                        request_id=request.request_id,
                        task_id=task_id,
                        service_type=node.service_type,
                        satellite_id=satellite_id,
                        start_time_days=parent_finish,
                        finish_time_days=finish,
                        compute_time_days=cache_days,
                        predecessor_ready_time_days=parent_finish,
                        scheduler_priority=ranks[task_id] + cache_gain,
                        selected_reason="ccp_public_cache_hit",
                    )
                )
                critical_parent[task_id] = pred[task_id][0]
                critical_routes[task_id] = None
                for child in succ[task_id]:
                    if child in unscheduled and all(parent in task_finish for parent in pred[child]):
                        ready.add(child)
                continue

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
                energy_proxy = (eft - est) * max(request.node_map()[task_id].workload_gflops, 1.0)
                candidate = (eft + float(config.get("ccp_energy_weight", 0.02)) * energy_proxy, eft, est, satellite_id)
                if best is None or candidate < best:
                    best = candidate
                    best_routes = routes
                    best_critical_route = critical_route
            if best is None:
                return build_execution_result(
                    request=request,
                    window_id=self.window_id,
                    success=False,
                    failure_reason="ccp_no_feasible_candidate",
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
            if enable_cache:
                self.public_cache.add(signature)
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
                    scheduler_priority=ranks[task_id],
                    selected_reason="ccp_composite_priority",
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
