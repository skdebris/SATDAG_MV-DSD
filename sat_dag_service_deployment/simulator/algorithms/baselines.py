from __future__ import annotations

import random
from collections import defaultdict

from ..models import DeploymentPlan, DeploymentWindowStats, RemoteSensingDAGRequest, SatelliteEnvironment, ServiceSpec
from ..utils import min_max_normalize
from .base import BaseDeploymentAlgorithm
from .common import place_services_by_score, satellite_feature_maps, service_replica_targets, service_role_statistics


def _service_demand_order(window_stats: DeploymentWindowStats, service_catalog: dict[str, ServiceSpec]) -> list[str]:
    return sorted(service_catalog, key=lambda service: window_stats.service_workload.get(service, 0.0), reverse=True)


class DependencyBlindAlgorithm(BaseDeploymentAlgorithm):
    name = "dependency_blind"

    def deploy(
        self,
        window_stats: DeploymentWindowStats,
        env: SatelliteEnvironment,
        service_catalog: dict[str, ServiceSpec],
        config: dict,
    ) -> DeploymentPlan:
        feature_maps = satellite_feature_maps(env)
        demand_signal = {
            service: window_stats.service_workload.get(service, 0.0) + 0.25 * window_stats.service_frequency.get(service, 0.0)
            for service in service_catalog
        }
        replica_targets = service_replica_targets(env, demand_signal, config)
        service_scores: dict[str, dict[str, float]] = defaultdict(dict)
        for service in service_catalog:
            for sat_id in env.satellites:
                service_scores[service][sat_id] = (
                    0.75 * feature_maps["cpu"][sat_id]
                    + 0.15 * feature_maps["memory"][sat_id]
                    + 0.10 * feature_maps["slots"][sat_id]
                )
        return place_services_by_score(
            algorithm_name=self.name,
            env=env,
            service_catalog=service_catalog,
            service_order=_service_demand_order(window_stats, service_catalog),
            replica_targets=replica_targets,
            satellite_scores=service_scores,
            cpu_weight_signal={service: 1.0 for service in service_catalog},
            metadata={"window_id": window_stats.window_id},
        )


def _extract_greedy_paths(request: RemoteSensingDAGRequest) -> list[list[str]]:
    adjacency = {node.task_id: [] for node in request.nodes}
    predecessors = {node.task_id: [] for node in request.nodes}
    for edge in request.edges:
        adjacency[edge.src].append(edge.dst)
        predecessors[edge.dst].append(edge.src)
    remaining = set(adjacency)
    paths: list[list[str]] = []
    while remaining:
        starts = sorted(
            node_id
            for node_id in remaining
            if not predecessors[node_id] or all(pred not in remaining for pred in predecessors[node_id])
        )
        current = starts[0] if starts else sorted(remaining)[0]
        path = [current]
        remaining.remove(current)
        while True:
            candidates = [child for child in adjacency[current] if child in remaining]
            if not candidates:
                break
            current = max(candidates, key=lambda item: len(adjacency[item]))
            path.append(current)
            remaining.remove(current)
        paths.append(path)
    return paths


class SFCPathDecompAlgorithm(BaseDeploymentAlgorithm):
    name = "sfc_path_decomp"

    def deploy(
        self,
        window_stats: DeploymentWindowStats,
        env: SatelliteEnvironment,
        service_catalog: dict[str, ServiceSpec],
        config: dict,
    ) -> DeploymentPlan:
        feature_maps = satellite_feature_maps(env)
        path_demand: dict[str, float] = defaultdict(float)
        path_traffic: dict[str, float] = defaultdict(float)
        for request in window_stats.requests:
            node_map = request.node_map()
            edge_lookup = {(edge.src, edge.dst): edge for edge in request.edges}
            for path in _extract_greedy_paths(request):
                for node_id in path:
                    path_demand[node_map[node_id].service_type] += node_map[node_id].workload_gflops
                for src, dst in zip(path, path[1:]):
                    edge = edge_lookup.get((src, dst))
                    if edge:
                        path_traffic[node_map[src].service_type] += edge.data_mb
                        path_traffic[node_map[dst].service_type] += edge.data_mb
        demand_signal = {service: path_demand.get(service, 0.0) + 0.35 * path_traffic.get(service, 0.0) for service in service_catalog}
        replica_targets = service_replica_targets(env, demand_signal, config)
        service_scores: dict[str, dict[str, float]] = defaultdict(dict)
        degree = feature_maps["degree"]
        for service in service_catalog:
            for sat_id in env.satellites:
                service_scores[service][sat_id] = 0.45 * feature_maps["cpu"][sat_id] + 0.55 * degree[sat_id]
        return place_services_by_score(
            algorithm_name=self.name,
            env=env,
            service_catalog=service_catalog,
            service_order=sorted(service_catalog, key=lambda service: demand_signal.get(service, 0.0), reverse=True),
            replica_targets=replica_targets,
            satellite_scores=service_scores,
            cpu_weight_signal=demand_signal,
            metadata={"window_id": window_stats.window_id},
        )


class GreedyResourceAlgorithm(BaseDeploymentAlgorithm):
    name = "greedy_resource"

    def deploy(
        self,
        window_stats: DeploymentWindowStats,
        env: SatelliteEnvironment,
        service_catalog: dict[str, ServiceSpec],
        config: dict,
    ) -> DeploymentPlan:
        feature_maps = satellite_feature_maps(env)
        cpu_weight_signal = {
            service: window_stats.service_workload.get(service, 0.0)
            for service in service_catalog
        }
        replica_targets = service_replica_targets(env, cpu_weight_signal, config)
        resource_score = {
            sat_id: (
                0.55 * feature_maps["cpu"][sat_id]
                + 0.20 * feature_maps["memory"][sat_id]
                + 0.15 * feature_maps["storage"][sat_id]
                + 0.10 * feature_maps["slots"][sat_id]
            )
            for sat_id in env.satellites
        }
        resource_score = min_max_normalize(resource_score)
        service_scores = defaultdict(dict)
        for service in service_catalog:
            for sat_id in env.satellites:
                service_scores[service][sat_id] = resource_score[sat_id]
        return place_services_by_score(
            algorithm_name=self.name,
            env=env,
            service_catalog=service_catalog,
            service_order=_service_demand_order(window_stats, service_catalog),
            replica_targets=replica_targets,
            satellite_scores=service_scores,
            cpu_weight_signal=cpu_weight_signal,
            metadata={"window_id": window_stats.window_id},
        )


class RandomDeployAlgorithm(BaseDeploymentAlgorithm):
    name = "random_deploy"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def deploy(
        self,
        window_stats: DeploymentWindowStats,
        env: SatelliteEnvironment,
        service_catalog: dict[str, ServiceSpec],
        config: dict,
    ) -> DeploymentPlan:
        rng = random.Random(self.seed + window_stats.window_id * 3571)
        _, _, demand_signal = service_role_statistics(window_stats)
        replica_targets = service_replica_targets(env, demand_signal, config)
        service_scores = defaultdict(dict)
        for service in service_catalog:
            for sat_id in env.satellites:
                service_scores[service][sat_id] = rng.random()
        service_order = list(service_catalog)
        rng.shuffle(service_order)
        return place_services_by_score(
            algorithm_name=self.name,
            env=env,
            service_catalog=service_catalog,
            service_order=service_order,
            replica_targets=replica_targets,
            satellite_scores=service_scores,
            cpu_weight_signal={service: 1.0 for service in service_catalog},
            metadata={"window_id": window_stats.window_id},
        )


class DRLDeployProxyAlgorithm(BaseDeploymentAlgorithm):
    name = "drl_deploy"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def deploy(
        self,
        window_stats: DeploymentWindowStats,
        env: SatelliteEnvironment,
        service_catalog: dict[str, ServiceSpec],
        config: dict,
    ) -> DeploymentPlan:
        rng = random.Random(self.seed + window_stats.window_id * 1237)
        feature_maps = satellite_feature_maps(env)
        _, _, demand_signal = service_role_statistics(window_stats)
        replica_targets = service_replica_targets(env, demand_signal, config)
        service_scores = defaultdict(dict)
        for service in service_catalog:
            service_bias = rng.uniform(0.2, 0.8)
            for sat_id in env.satellites:
                service_scores[service][sat_id] = (
                    service_bias * feature_maps["cpu"][sat_id]
                    + (1.0 - service_bias) * feature_maps["degree"][sat_id]
                )
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
                "implementation_note": "proxy baseline with fixed-score policy in place of trained DRL agent",
            },
        )
