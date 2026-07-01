from __future__ import annotations

import math
import random
from collections import defaultdict

from ..models import DeploymentPlan, DeploymentWindowStats, RemoteSensingDAGRequest, SatelliteEnvironment, ServiceSpec
from ..utils import min_max_normalize, topological_sort
from .base import BaseDeploymentAlgorithm
from .common import place_services_by_score, satellite_feature_maps, service_replica_targets


def _service_demand_signal(window_stats: DeploymentWindowStats, service_catalog: dict[str, ServiceSpec]) -> dict[str, float]:
    traffic: dict[str, float] = defaultdict(float)
    for (src, dst), volume in window_stats.service_traffic.items():
        traffic[src] += volume
        traffic[dst] += volume
    return {
        service: (
            window_stats.service_workload.get(service, 0.0)
            + 0.20 * window_stats.service_frequency.get(service, 0.0)
            + 0.03 * traffic.get(service, 0.0)
        )
        for service in service_catalog
    }


def _service_demand_signal_from_requests(
    requests: list[RemoteSensingDAGRequest],
    service_catalog: dict[str, ServiceSpec],
) -> dict[str, float]:
    workload: dict[str, float] = defaultdict(float)
    frequency: dict[str, float] = defaultdict(float)
    traffic: dict[str, float] = defaultdict(float)
    for request in requests:
        for node in request.nodes:
            workload[node.service_type] += node.workload_gflops
            frequency[node.service_type] += 1.0
        for edge in request.edges:
            traffic[edge.src_service_type] += edge.data_mb
            traffic[edge.dst_service_type] += edge.data_mb
    return {
        service: (
            workload.get(service, 0.0)
            + 0.20 * frequency.get(service, 0.0)
            + 0.03 * traffic.get(service, 0.0)
        )
        for service in service_catalog
    }


def _resource_state(env: SatelliteEnvironment) -> tuple[dict[str, float], dict[str, float], dict[str, int]]:
    return (
        {sat_id: sat.memory_mb for sat_id, sat in env.satellites.items()},
        {sat_id: sat.storage_mb for sat_id, sat in env.satellites.items()},
        {sat_id: sat.container_slots for sat_id, sat in env.satellites.items()},
    )


def _can_configure(
    sat_id: str,
    service: str,
    service_catalog: dict[str, ServiceSpec],
    remaining_memory: dict[str, float],
    remaining_storage: dict[str, float],
    remaining_slots: dict[str, int],
) -> bool:
    spec = service_catalog[service]
    return (
        remaining_slots[sat_id] > 0
        and remaining_memory[sat_id] >= spec.memory_mb
        and remaining_storage[sat_id] >= spec.storage_mb
    )


def _consume_config(
    sat_id: str,
    service: str,
    service_catalog: dict[str, ServiceSpec],
    remaining_memory: dict[str, float],
    remaining_storage: dict[str, float],
    remaining_slots: dict[str, int],
) -> None:
    spec = service_catalog[service]
    remaining_memory[sat_id] -= spec.memory_mb
    remaining_storage[sat_id] -= spec.storage_mb
    remaining_slots[sat_id] -= 1


def _critical_path(request: RemoteSensingDAGRequest) -> list[str]:
    node_map = request.node_map()
    edges = [(edge.src, edge.dst) for edge in request.edges]
    order = topological_sort(node_map.keys(), edges)
    score = {node_id: node_map[node_id].workload_gflops for node_id in node_map}
    parent: dict[str, str | None] = {node_id: None for node_id in node_map}
    edge_weight = {(edge.src, edge.dst): edge.data_mb / 100.0 for edge in request.edges}
    children: dict[str, list[str]] = {node_id: [] for node_id in node_map}
    for edge in request.edges:
        children[edge.src].append(edge.dst)
    for node_id in order:
        for child_id in children[node_id]:
            candidate = score[node_id] + edge_weight[(node_id, child_id)] + node_map[child_id].workload_gflops
            if candidate > score[child_id]:
                score[child_id] = candidate
                parent[child_id] = node_id
    sink = max(order, key=lambda task_id: score[task_id])
    path = [sink]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])  # type: ignore[arg-type]
    path.reverse()
    return path


def _cpu_allocation_from_load(
    algorithm_name: str,
    env: SatelliteEnvironment,
    service_catalog: dict[str, ServiceSpec],
    placements: dict[str, list[str]],
    load: dict[tuple[str, str], float],
    metadata: dict,
) -> DeploymentPlan:
    cpu_allocation: dict[tuple[str, str], float] = {}
    for sat_id, satellite in env.satellites.items():
        hosted = [service for service, sats in placements.items() if sat_id in sats]
        if not hosted:
            continue
        weights = {service: max(0.05, load.get((sat_id, service), 0.0)) for service in hosted}
        total = sum(weights.values())
        for service, weight in weights.items():
            cpu_allocation[(sat_id, service)] = round(satellite.cpu_capacity_ghz * weight / total, 6)
    return DeploymentPlan(
        algorithm_name=algorithm_name,
        window_id=int(metadata["window_id"]),
        service_placement={service: sorted(set(sats)) for service, sats in placements.items()},
        cpu_allocation=cpu_allocation,
        replica_count={service: len(set(sats)) for service, sats in placements.items()},
        metadata=metadata,
    )


class JSDTSAOSAlgorithm(BaseDeploymentAlgorithm):
    name = "jsdts_aos_sat"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def _candidate_score(
        self,
        weights: tuple[float, float, float, float],
        feature_maps: dict[str, dict[str, float]],
        window_stats: DeploymentWindowStats,
        env: SatelliteEnvironment,
    ) -> float:
        cpu_w, mem_w, store_w, degree_w = weights
        sat_score = {
            sat_id: (
                cpu_w * feature_maps["cpu"][sat_id]
                + mem_w * feature_maps["memory"][sat_id]
                + store_w * feature_maps["storage"][sat_id]
                + degree_w * feature_maps["degree"][sat_id]
            )
            for sat_id in env.satellites
        }
        capacity_gain = sum(sorted(sat_score.values(), reverse=True)[: max(1, len(window_stats.service_workload))])
        traffic_pressure = sum(window_stats.service_traffic.values()) / 1000.0
        topology_gain = sum(feature_maps["degree"].values()) / max(len(feature_maps["degree"]), 1)
        storage_penalty = 0.08 * store_w * len(window_stats.service_workload)
        return capacity_gain + degree_w * topology_gain - 0.01 * traffic_pressure - storage_penalty

    def deploy(
        self,
        window_stats: DeploymentWindowStats,
        env: SatelliteEnvironment,
        service_catalog: dict[str, ServiceSpec],
        config: dict,
    ) -> DeploymentPlan:
        rng = random.Random(self.seed + window_stats.window_id * 4271)
        feature_maps = satellite_feature_maps(env)
        demand_signal = _service_demand_signal(window_stats, service_catalog)
        replica_targets = service_replica_targets(env, demand_signal, config)
        population_size = int(config.get("aos_population_size", 24))
        max_iter = int(config.get("aos_iterations", 20))
        population = [
            tuple(rng.uniform(0.05, 1.0) for _ in range(4))
            for _ in range(population_size)
        ]
        best = population[0]
        best_score = -float("inf")
        for iteration in range(max_iter):
            scored = [
                (self._candidate_score(candidate, feature_maps, window_stats, env), candidate)
                for candidate in population
            ]
            scored.sort(reverse=True, key=lambda item: item[0])
            if scored[0][0] > best_score:
                best_score, best = scored[0]
            worst = scored[-1][1]
            next_population = [best]
            exploration = max(0.05, 1.0 - (iteration + 1) / max(max_iter, 1))
            while len(next_population) < population_size:
                parent = rng.choice(scored[: max(2, population_size // 3)])[1]
                child = []
                for value, best_value, worst_value in zip(parent, best, worst):
                    direction = rng.random() * (best_value - worst_value)
                    perturb = rng.uniform(-exploration, exploration) * 0.25
                    child.append(max(0.01, value + 0.45 * direction + perturb))
                total = sum(child)
                next_population.append(tuple(value / total for value in child))
            population = next_population

        cpu_w, mem_w, store_w, degree_w = best
        satellite_priority = {
            sat_id: (
                cpu_w * feature_maps["cpu"][sat_id]
                + mem_w * feature_maps["memory"][sat_id]
                + store_w * feature_maps["storage"][sat_id]
                + degree_w * feature_maps["degree"][sat_id]
            )
            for sat_id in env.satellites
        }
        service_scores = defaultdict(dict)
        for service in service_catalog:
            for sat_id in env.satellites:
                service_scores[service][sat_id] = satellite_priority[sat_id]
        return place_services_by_score(
            algorithm_name=self.name,
            env=env,
            service_catalog=service_catalog,
            service_order=sorted(service_catalog, key=lambda service: demand_signal.get(service, 0.0), reverse=True),
            replica_targets=replica_targets,
            satellite_scores=service_scores,
            cpu_weight_signal=demand_signal,
            metadata={
                "window_id": window_stats.window_id,
                "aos_best_score": best_score,
                "aos_weights": {"cpu": cpu_w, "memory": mem_w, "storage": store_w, "degree": degree_w},
            },
            satellite_priority=satellite_priority,
        )


class OnDocSatAlgorithm(BaseDeploymentAlgorithm):
    name = "ondoc_sat"

    def _deploy_statistical(
        self,
        window_stats: DeploymentWindowStats,
        env: SatelliteEnvironment,
        service_catalog: dict[str, ServiceSpec],
        config: dict,
    ) -> DeploymentPlan:
        feature_maps = satellite_feature_maps(env)
        demand_signal = _service_demand_signal(window_stats, service_catalog)
        replica_targets = service_replica_targets(env, demand_signal, config)

        traffic_by_service: dict[str, float] = defaultdict(float)
        for (src_service, dst_service), volume in window_stats.service_traffic.items():
            traffic_by_service[src_service] += volume
            traffic_by_service[dst_service] += volume
        traffic_signal = min_max_normalize(traffic_by_service)
        workload_signal = min_max_normalize(window_stats.service_workload)

        service_scores: dict[str, dict[str, float]] = defaultdict(dict)
        for service in service_catalog:
            workload_bias = workload_signal.get(service, 0.0)
            traffic_bias = traffic_signal.get(service, 0.0)
            for sat_id in env.satellites:
                service_scores[service][sat_id] = (
                    (0.50 + 0.10 * workload_bias) * feature_maps["cpu"][sat_id]
                    + 0.18 * feature_maps["memory"][sat_id]
                    + 0.10 * feature_maps["storage"][sat_id]
                    + (0.18 + 0.12 * traffic_bias) * feature_maps["degree"][sat_id]
                )

        satellite_priority = {
            sat_id: (
                0.52 * feature_maps["cpu"][sat_id]
                + 0.18 * feature_maps["memory"][sat_id]
                + 0.10 * feature_maps["storage"][sat_id]
                + 0.20 * feature_maps["degree"][sat_id]
            )
            for sat_id in env.satellites
        }
        return place_services_by_score(
            algorithm_name=self.name,
            env=env,
            service_catalog=service_catalog,
            service_order=sorted(service_catalog, key=lambda service: demand_signal.get(service, 0.0), reverse=True),
            replica_targets=replica_targets,
            satellite_scores=service_scores,
            cpu_weight_signal=demand_signal,
            metadata={
                "window_id": window_stats.window_id,
                "ondoc_planning_mode": "statistical",
                "available_request_count": len(window_stats.requests),
                "implementation_note": "Statistical OnDoc-style joint server-service selection without request-level DAG replay.",
            },
            satellite_priority=satellite_priority,
        )

    def _deploy_trace_replay(
        self,
        window_stats: DeploymentWindowStats,
        env: SatelliteEnvironment,
        service_catalog: dict[str, ServiceSpec],
        config: dict,
    ) -> DeploymentPlan:
        remaining_memory, remaining_storage, remaining_slots = _resource_state(env)
        configured: dict[str, set[str]] = {sat_id: set() for sat_id in env.satellites}
        usage: dict[tuple[str, str], float] = defaultdict(float)
        load: dict[tuple[str, str], float] = defaultdict(float)
        feature_maps = satellite_feature_maps(env)
        all_requests = sorted(window_stats.requests, key=lambda item: item.arrival_time_days)
        replay_limit = int(config.get("ondoc_replay_request_limit", 32))
        replay_requests = all_requests[:replay_limit] if replay_limit > 0 else all_requests
        for request in replay_requests:
            assigned: dict[str, str] = {}
            order = topological_sort((node.task_id for node in request.nodes), ((edge.src, edge.dst) for edge in request.edges))
            for task_id in order:
                node = request.node_map()[task_id]
                service = node.service_type
                predecessor_sats = [
                    assigned[edge.src]
                    for edge in request.edges
                    if edge.dst == task_id and edge.src in assigned
                ]
                candidates = []
                for sat_id in env.satellites:
                    already = service in configured[sat_id]
                    configurable = already or _can_configure(sat_id, service, service_catalog, remaining_memory, remaining_storage, remaining_slots)
                    if not configurable:
                        continue
                    reuse_bonus = 0.25 if already else 0.0
                    locality = 0.0
                    for pred_sat in predecessor_sats:
                        locality += env.aggregate_graph.get(pred_sat, {}).get(sat_id, 0.0)
                        if pred_sat == sat_id:
                            locality += max(env.aggregate_threshold, 1.0)
                    score = (
                        0.50 * feature_maps["cpu"][sat_id]
                        + 0.20 * feature_maps["memory"][sat_id]
                        + 0.20 * feature_maps["degree"][sat_id]
                        + reuse_bonus
                        + 0.10 * locality / max(env.aggregate_threshold, 1.0)
                    )
                    candidates.append((score, sat_id, already))
                if not candidates:
                    continue
                _, sat_id, already = max(candidates, key=lambda item: (item[0], item[1]))
                if not already:
                    _consume_config(sat_id, service, service_catalog, remaining_memory, remaining_storage, remaining_slots)
                    configured[sat_id].add(service)
                assigned[task_id] = sat_id
                usage[(sat_id, service)] += 1.0
                load[(sat_id, service)] += node.workload_gflops

        demand_signal = _service_demand_signal_from_requests(replay_requests, service_catalog)
        replica_targets = service_replica_targets(env, demand_signal, config)
        placements = {service: [] for service in service_catalog}
        ranked_pairs = sorted(usage, key=lambda pair: (usage[pair], load[pair]), reverse=True)
        for sat_id, service in ranked_pairs:
            if len(placements[service]) < replica_targets.get(service, 1):
                placements[service].append(sat_id)
        fallback_scores = satellite_feature_maps(env)
        for service in service_catalog:
            if placements[service]:
                continue
            sat_id = max(env.satellites, key=lambda item: fallback_scores["cpu"][item])
            placements[service].append(sat_id)
            load[(sat_id, service)] += demand_signal.get(service, 1.0)
        return _cpu_allocation_from_load(
            algorithm_name=self.name,
            env=env,
            service_catalog=service_catalog,
            placements=placements,
            load=load,
            metadata={
                "window_id": window_stats.window_id,
                "ondoc_planning_mode": "trace_replay",
                "replay_request_count": len(replay_requests),
                "available_request_count": len(window_stats.requests),
                "implementation_note": "History-limited OnDoc-style function configuration adapted to a fixed satellite deployment.",
            },
        )

    def deploy(
        self,
        window_stats: DeploymentWindowStats,
        env: SatelliteEnvironment,
        service_catalog: dict[str, ServiceSpec],
        config: dict,
    ) -> DeploymentPlan:
        mode = str(config.get("ondoc_planning_mode", "statistical")).lower()
        if mode in {"trace_replay", "replay", "history_replay"}:
            return self._deploy_trace_replay(window_stats, env, service_catalog, config)
        return self._deploy_statistical(window_stats, env, service_catalog, config)


class FloodSFCPGreedyAlgorithm(BaseDeploymentAlgorithm):
    name = "floodsfcp_greedy"

    def deploy(
        self,
        window_stats: DeploymentWindowStats,
        env: SatelliteEnvironment,
        service_catalog: dict[str, ServiceSpec],
        config: dict,
    ) -> DeploymentPlan:
        feature_maps = satellite_feature_maps(env)
        usage: dict[tuple[str, str], float] = defaultdict(float)
        load: dict[tuple[str, str], float] = defaultdict(float)
        for request in window_stats.requests:
            previous_sat: str | None = None
            for task_id in _critical_path(request):
                node = request.node_map()[task_id]
                service = node.service_type
                candidates = []
                for sat_id in env.satellites:
                    connectivity = 0.0
                    if previous_sat is not None:
                        connectivity = env.aggregate_graph.get(previous_sat, {}).get(sat_id, 0.0)
                        if previous_sat == sat_id:
                            connectivity = max(connectivity, env.aggregate_threshold)
                    score = (
                        0.45 * feature_maps["cpu"][sat_id]
                        + 0.20 * feature_maps["storage"][sat_id]
                        + 0.25 * feature_maps["degree"][sat_id]
                        + 0.10 * connectivity / max(env.aggregate_threshold, 1.0)
                    )
                    candidates.append((score, sat_id))
                _, chosen_sat = max(candidates)
                usage[(chosen_sat, service)] += 1.0
                load[(chosen_sat, service)] += node.workload_gflops
                previous_sat = chosen_sat

        demand_signal = _service_demand_signal(window_stats, service_catalog)
        replica_targets = service_replica_targets(env, demand_signal, config)
        placements = {service: [] for service in service_catalog}
        remaining_memory, remaining_storage, remaining_slots = _resource_state(env)
        for sat_id, service in sorted(usage, key=lambda pair: (usage[pair], load[pair]), reverse=True):
            if len(placements[service]) >= replica_targets.get(service, 1):
                continue
            if sat_id in placements[service]:
                continue
            if not _can_configure(sat_id, service, service_catalog, remaining_memory, remaining_storage, remaining_slots):
                continue
            placements[service].append(sat_id)
            _consume_config(sat_id, service, service_catalog, remaining_memory, remaining_storage, remaining_slots)
        for service in service_catalog:
            if placements[service]:
                continue
            for sat_id in sorted(env.satellites, key=lambda item: feature_maps["cpu"][item], reverse=True):
                if _can_configure(sat_id, service, service_catalog, remaining_memory, remaining_storage, remaining_slots):
                    placements[service].append(sat_id)
                    _consume_config(sat_id, service, service_catalog, remaining_memory, remaining_storage, remaining_slots)
                    load[(sat_id, service)] += demand_signal.get(service, 1.0)
                    break
            if not placements[service]:
                fallback = max(env.satellites, key=lambda item: env.satellites[item].cpu_capacity_ghz)
                placements[service].append(fallback)
                load[(fallback, service)] += demand_signal.get(service, 1.0)
        return _cpu_allocation_from_load(
            algorithm_name=self.name,
            env=env,
            service_catalog=service_catalog,
            placements=placements,
            load=load,
            metadata={
                "window_id": window_stats.window_id,
                "chain_mode": "critical_path_chain",
                "implementation_note": "FloodSFCP-inspired greedy chain placement; not labeled as DRL.",
            },
        )
