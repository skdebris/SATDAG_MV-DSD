from __future__ import annotations

import math
from collections import defaultdict

from ..models import DeploymentPlan, DeploymentWindowStats, Satellite, SatelliteEnvironment, ServiceSpec
from ..utils import min_max_normalize, normalize_dict


def weighted_degree(graph: dict[str, dict[str, float]]) -> dict[str, float]:
    return {node: sum(neighbors.values()) for node, neighbors in graph.items()}


def satellite_feature_maps(env: SatelliteEnvironment) -> dict[str, dict[str, float]]:
    cpu = {sat_id: satellite.cpu_capacity_ghz for sat_id, satellite in env.satellites.items()}
    memory = {sat_id: satellite.memory_mb for sat_id, satellite in env.satellites.items()}
    storage = {sat_id: satellite.storage_mb for sat_id, satellite in env.satellites.items()}
    slots = {sat_id: float(satellite.container_slots) for sat_id, satellite in env.satellites.items()}
    degree = weighted_degree(env.aggregate_graph)
    for sat_id in env.satellites:
        degree.setdefault(sat_id, 0.0)
    return {
        "cpu": min_max_normalize(cpu),
        "memory": min_max_normalize(memory),
        "storage": min_max_normalize(storage),
        "slots": min_max_normalize(slots),
        "degree": min_max_normalize(degree),
    }


def service_role_statistics(window_stats: DeploymentWindowStats) -> tuple[dict[str, float], dict[str, tuple[float, float, float]], dict[str, float]]:
    cpf = normalize_dict(window_stats.critical_path_service_count)
    cp_workload = normalize_dict(window_stats.critical_path_service_weight)
    workload = normalize_dict(window_stats.service_workload)
    sync_weight = normalize_dict(window_stats.service_sync_weight)
    traffic_by_service: dict[str, float] = defaultdict(float)
    for (src_service, dst_service), volume in window_stats.service_traffic.items():
        traffic_by_service[src_service] += volume
        traffic_by_service[dst_service] += volume
    traffic = normalize_dict(dict(traffic_by_service))

    services = set(window_stats.service_workload) | set(cpf) | set(cp_workload) | set(sync_weight) | set(traffic)
    role_vectors: dict[str, tuple[float, float, float]] = {}
    combined_signal: dict[str, float] = {}
    for service in services:
        cmp_weight = cpf.get(service, 0.0) + cp_workload.get(service, 0.0)
        net_weight = traffic.get(service, 0.0)
        idle_weight = sync_weight.get(service, 0.0)
        total = cmp_weight + net_weight + idle_weight
        if total <= 0:
            role_vectors[service] = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
        else:
            role_vectors[service] = (cmp_weight / total, net_weight / total, idle_weight / total)
        combined_signal[service] = workload.get(service, 0.0) + traffic.get(service, 0.0) + cpf.get(service, 0.0)
    return cpf, role_vectors, combined_signal


def coverage_replica_targets(
    services: list[str],
    cpf: dict[str, float],
    config: dict,
) -> dict[str, int]:
    if not services:
        return {}
    nu_max = int(config["max_replicas_per_service"])
    max_cpf = max((cpf.get(service, 0.0) for service in services), default=0.0)
    targets: dict[str, int] = {}
    for service in services:
        if max_cpf <= 0:
            targets[service] = 1
            continue
        cpf_ratio = cpf.get(service, 0.0) / max_cpf
        targets[service] = max(1, math.ceil(nu_max * cpf_ratio))
    return targets


def service_replica_targets(
    env: SatelliteEnvironment,
    demand_signal: dict[str, float],
    config: dict,
) -> dict[str, int]:
    if not demand_signal:
        return {}
    normalized = min_max_normalize(demand_signal)
    max_replicas = int(config["max_replicas_per_service"])
    min_replicas = int(config["min_replicas_per_service"])
    scale = float(config["replica_scale_factor"])
    num_satellites = len(env.satellites)
    targets = {}
    for service, signal in normalized.items():
        suggested = min_replicas + round(signal * num_satellites * scale)
        targets[service] = max(min_replicas, min(max_replicas, suggested))
    return targets


def _satellite_can_host(
    satellite: Satellite,
    service: ServiceSpec,
    remaining_memory: dict[str, float],
    remaining_storage: dict[str, float],
    remaining_slots: dict[str, int],
) -> bool:
    sat_id = satellite.satellite_id
    return (
        remaining_memory[sat_id] >= service.memory_mb
        and remaining_storage[sat_id] >= service.storage_mb
        and remaining_slots[sat_id] >= 1
    )


def place_services_by_score(
    algorithm_name: str,
    env: SatelliteEnvironment,
    service_catalog: dict[str, ServiceSpec],
    service_order: list[str],
    replica_targets: dict[str, int],
    satellite_scores: dict[str, dict[str, float]],
    cpu_weight_signal: dict[str, float],
    metadata: dict,
    satellite_priority: dict[str, float] | None = None,
    score_weighted_cpu: bool = False,
) -> DeploymentPlan:
    remaining_memory = {sat_id: satellite.memory_mb for sat_id, satellite in env.satellites.items()}
    remaining_storage = {sat_id: satellite.storage_mb for sat_id, satellite in env.satellites.items()}
    remaining_slots = {sat_id: satellite.container_slots for sat_id, satellite in env.satellites.items()}
    placements: dict[str, list[str]] = {service: [] for service in service_catalog}
    placed_count = {service: 0 for service in service_catalog}

    if satellite_priority is None:
        for service in service_order:
            if service not in service_catalog:
                continue
            ranking = sorted(
                env.satellites,
                key=lambda sat_id: satellite_scores.get(service, {}).get(sat_id, 0.0),
                reverse=True,
            )
            target = replica_targets.get(service, 1)
            for sat_id in ranking:
                if len(placements[service]) >= target:
                    break
                if sat_id in placements[service]:
                    continue
                satellite = env.satellites[sat_id]
                spec = service_catalog[service]
                if not _satellite_can_host(satellite, spec, remaining_memory, remaining_storage, remaining_slots):
                    continue
                placements[service].append(sat_id)
                remaining_memory[sat_id] -= spec.memory_mb
                remaining_storage[sat_id] -= spec.storage_mb
                remaining_slots[sat_id] -= 1
                placed_count[service] += 1
        for service in service_catalog:
            if placements[service]:
                continue
            fallback = max(env.satellites, key=lambda sat_id: env.satellites[sat_id].cpu_capacity_ghz)
            placements[service].append(fallback)
            placed_count[service] = 1
    else:
        satellites_sorted = sorted(
            env.satellites,
            key=lambda sat_id: satellite_priority.get(sat_id, 0.0),
            reverse=True,
        )
        ordered_services = [service for service in service_order if service in service_catalog]
        for sat_id in satellites_sorted:
            if all(placed_count[service] >= replica_targets.get(service, 1) for service in ordered_services):
                break
            while remaining_slots[sat_id] > 0:
                ranked_services = sorted(
                    ordered_services,
                    key=lambda service: (
                        placed_count[service] < replica_targets.get(service, 1),
                        satellite_scores.get(service, {}).get(sat_id, 0.0),
                        cpu_weight_signal.get(service, 0.0),
                    ),
                    reverse=True,
                )
                placed_any = False
                for service in ranked_services:
                    if placed_count[service] >= replica_targets.get(service, 1):
                        continue
                    if sat_id in placements[service]:
                        continue
                    satellite = env.satellites[sat_id]
                    spec = service_catalog[service]
                    if not _satellite_can_host(satellite, spec, remaining_memory, remaining_storage, remaining_slots):
                        continue
                    placements[service].append(sat_id)
                    remaining_memory[sat_id] -= spec.memory_mb
                    remaining_storage[sat_id] -= spec.storage_mb
                    remaining_slots[sat_id] -= 1
                    placed_count[service] += 1
                    placed_any = True
                    break
                if not placed_any:
                    break

        for service in ordered_services:
            target = replica_targets.get(service, 1)
            if placed_count[service] >= target:
                continue
            ranking = sorted(
                env.satellites,
                key=lambda sat_id: (
                    satellite_scores.get(service, {}).get(sat_id, 0.0),
                    satellite_priority.get(sat_id, 0.0),
                ),
                reverse=True,
            )
            for sat_id in ranking:
                if placed_count[service] >= target:
                    break
                if sat_id in placements[service]:
                    continue
                satellite = env.satellites[sat_id]
                spec = service_catalog[service]
                if not _satellite_can_host(satellite, spec, remaining_memory, remaining_storage, remaining_slots):
                    continue
                placements[service].append(sat_id)
                remaining_memory[sat_id] -= spec.memory_mb
                remaining_storage[sat_id] -= spec.storage_mb
                remaining_slots[sat_id] -= 1
                placed_count[service] += 1

    for service in service_catalog:
        if placements[service]:
            continue
        fallback = max(env.satellites, key=lambda sat_id: env.satellites[sat_id].cpu_capacity_ghz)
        placements[service].append(fallback)
        placed_count[service] = 1

    cpu_allocation: dict[tuple[str, str], float] = {}
    for sat_id, satellite in env.satellites.items():
        hosted = [service for service, sats in placements.items() if sat_id in sats]
        if not hosted:
            continue
        if score_weighted_cpu:
            weights = {
                service: max(
                    0.05,
                    cpu_weight_signal.get(service, 0.0) * max(0.0, satellite_scores.get(service, {}).get(sat_id, 0.0)),
                )
                for service in hosted
            }
        else:
            weights = {service: max(0.05, cpu_weight_signal.get(service, 0.0)) for service in hosted}
        if sum(weights.values()) <= 0.0:
            weights = {service: 1.0 for service in hosted}
        total = sum(weights.values())
        for service, weight in weights.items():
            cpu_allocation[(sat_id, service)] = round(satellite.cpu_capacity_ghz * weight / total, 6)

    replica_count = {service: len(sats) for service, sats in placements.items()}
    return DeploymentPlan(
        algorithm_name=algorithm_name,
        window_id=int(metadata["window_id"]),
        service_placement={service: sorted(set(sats)) for service, sats in placements.items()},
        cpu_allocation=cpu_allocation,
        replica_count=replica_count,
        metadata=metadata,
    )


def connected_components(coalition: set[str], env: SatelliteEnvironment, constrained: bool) -> list[set[str]]:
    if not coalition:
        return []
    if not constrained:
        return [set(coalition)]
    unvisited = set(coalition)
    components: list[set[str]] = []
    while unvisited:
        root = sorted(unvisited)[0]
        unvisited.remove(root)
        stack = [root]
        component = {root}
        while stack:
            node = stack.pop()
            for neighbor in sorted(env.aggregate_graph.get(node, {})):
                if neighbor in unvisited:
                    unvisited.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return components


def coalition_value_components(
    coalition: set[str],
    env: SatelliteEnvironment,
    component_weights: tuple[float, float, float],
    feature_maps: dict[str, dict[str, float]],
    constrained: bool,
    bridge_bonus: float,
) -> tuple[float, float, float]:
    if not coalition:
        return (0.0, 0.0, 0.0)
    cmp_scale, net_scale, idle_scale = component_weights
    components = connected_components(coalition, env, constrained)
    total_cmp = 0.0
    total_net = 0.0
    total_idle = 0.0
    for component in components:
        size = len(component)
        cpu_sum = sum(feature_maps["cpu"].get(node, 0.0) for node in component)
        degree_sum = sum(feature_maps["degree"].get(node, 0.0) for node in component)
        idle_sum = sum(
            0.55 * feature_maps["memory"].get(node, 0.0)
            + 0.25 * feature_maps["slots"].get(node, 0.0)
            + 0.20 * feature_maps["storage"].get(node, 0.0)
            for node in component
        )
        internal_contact = 0.0
        for src in sorted(component):
            for dst, volume in sorted(env.aggregate_graph.get(src, {}).items()):
                if dst in component and src < dst:
                    internal_contact += volume
        internal_contact = internal_contact / max(env.aggregate_threshold, 1.0)
        pair_bonus = size * (size - 1) / 2.0
        total_cmp += cmp_scale * ((1.0 + cpu_sum) ** 1.15)
        total_net += net_scale * (0.65 * degree_sum + 0.25 * internal_contact + bridge_bonus * pair_bonus)
        total_idle += idle_scale * (idle_sum * math.log1p(size) + 0.04 * pair_bonus)
    return total_cmp, total_net, total_idle
