from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import replace
from statistics import mean

from ..models import DeploymentPlan, DeploymentWindowStats, SatelliteEnvironment, ServiceSpec
from ..utils import min_max_normalize, normalize_dict
from .base import BaseDeploymentAlgorithm
from .common import (
    coalition_value_components,
    coverage_replica_targets,
    place_services_by_score,
    satellite_feature_maps,
    service_role_statistics,
)


class CPMVDSDAlgorithm(BaseDeploymentAlgorithm):
    name = "cpmv_dsd"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def _component_weights(self, window_stats: DeploymentWindowStats) -> tuple[float, float, float]:
        total_workload = sum(window_stats.service_workload.values())
        total_traffic = sum(window_stats.service_traffic.values())
        total_sync = sum(window_stats.service_sync_weight.values())
        scaled_traffic = total_traffic / 120.0
        total = total_workload + scaled_traffic + total_sync
        if total <= 0:
            return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
        return (total_workload / total, scaled_traffic / total, total_sync / total)

    def _proposal_weights(
        self,
        env: SatelliteEnvironment,
        feature_maps: dict[str, dict[str, float]],
        use_stratified_sampling: bool,
    ) -> dict[str, float]:
        weighted_degree = feature_maps["degree"]
        cpu = feature_maps["cpu"]
        priority = {sat_id: 0.6 * cpu[sat_id] + 0.4 * weighted_degree[sat_id] for sat_id in env.satellites}
        priority = min_max_normalize(priority)
        if not use_stratified_sampling:
            return {sat_id: 1.0 for sat_id in env.satellites}
        return {sat_id: 0.12 + priority[sat_id] ** 2 for sat_id in env.satellites}

    def _sample_permutation(
        self,
        rng: random.Random,
        satellites: list[str],
        proposal_weights: dict[str, float],
    ) -> tuple[list[str], float]:
        remaining = list(satellites)
        permutation: list[str] = []
        proposal_prob = 1.0
        while remaining:
            total_weight = sum(proposal_weights[sat_id] for sat_id in remaining)
            selector = rng.random() * total_weight
            cumulative = 0.0
            chosen = remaining[-1]
            for sat_id in remaining:
                cumulative += proposal_weights[sat_id]
                if cumulative >= selector:
                    chosen = sat_id
                    break
            proposal_prob *= proposal_weights[chosen] / total_weight
            permutation.append(chosen)
            remaining.remove(chosen)
        return permutation, max(proposal_prob, 1e-30)

    def _complete_graph_environment(self, env: SatelliteEnvironment) -> SatelliteEnvironment:
        complete_graph = {
            src: {
                dst: 1.0
                for dst in env.satellites
                if dst != src
            }
            for src in env.satellites
        }
        return replace(env, aggregate_graph=complete_graph, aggregate_threshold=1.0, density_mode="complete")

    def _estimate_structured_values(
        self,
        window_stats: DeploymentWindowStats,
        env: SatelliteEnvironment,
        config: dict,
    ) -> tuple[dict[str, tuple[float, float, float]], dict]:
        rng = random.Random(self.seed + window_stats.window_id * 1009)
        feature_maps = satellite_feature_maps(env)
        component_weights = self._component_weights(window_stats)
        proposal_weights = self._proposal_weights(
            env=env,
            feature_maps=feature_maps,
            use_stratified_sampling=bool(config["use_stratified_sampling"]),
        )
        satellites = list(env.satellites)
        aggregates: dict[str, list[float]] = {sat_id: [0.0, 0.0, 0.0] for sat_id in satellites}
        raw_weights: list[float] = []
        sample_records: list[tuple[list[str], float]] = []
        oracle_calls = 0
        pruned_calls = 0
        constrained = bool(config["use_comm_graph_constraint"])
        use_structured_value = bool(config.get("use_structured_value", True))
        use_topology_features = bool(config.get("use_topology_features", constrained))
        value_env = env if use_topology_features else self._complete_graph_environment(env)
        if not use_topology_features:
            feature_maps["degree"] = {sat_id: 1.0 for sat_id in env.satellites}
        bridge_bonus = float(config["comm_graph_bridge_bonus"])
        use_cvar = bool(config.get("use_cvar", False))
        cvar_lambda = float(config.get("cvar_lambda", 0.0))
        cvar_alpha = float(config.get("cvar_alpha", 0.9))
        cvar_risk_amplifier = float(config.get("cvar_risk_amplifier", 1.0))
        topology_samples = list(config.get("cvar_topology_samples") or [])
        topology_sample_features = [satellite_feature_maps(sample_env) for sample_env in topology_samples]

        def lower_tail_mean(values: list[float], alpha: float) -> float:
            if not values:
                return 0.0
            ordered = sorted(values)
            tail_count = max(1, math.ceil((1.0 - alpha) * len(ordered)))
            return mean(ordered[:tail_count])

        def evaluate_coalition(coalition: set[str]) -> tuple[float, float, float]:
            if not use_cvar or not topology_samples or cvar_lambda <= 0.0:
                components = coalition_value_components(
                    coalition,
                    value_env,
                    component_weights,
                    feature_maps,
                    constrained=constrained,
                    bridge_bonus=bridge_bonus,
                )
                if use_structured_value:
                    return components
                return (sum(components), 0.0, 0.0)

            sample_values = []
            for sample_env, sample_features in zip(topology_samples, topology_sample_features):
                sample_values.append(
                    coalition_value_components(
                        coalition,
                        sample_env,
                        component_weights,
                        sample_features,
                        constrained=constrained,
                        bridge_bonus=bridge_bonus,
                    )
                )
            if not use_structured_value:
                scalar_values = [sum(values) for values in sample_values]
                scalar_mean = mean(scalar_values) if scalar_values else 0.0
                return (scalar_mean, 0.0, 0.0)
            component_means = tuple(mean(values[idx] for values in sample_values) for idx in range(3))
            scalar_samples = [sum(values) for values in sample_values]
            scalar_mean = mean(scalar_samples) if scalar_samples else 0.0
            scalar_tail = lower_tail_mean(scalar_samples, cvar_alpha)
            robust_scalar = scalar_mean - cvar_lambda * cvar_risk_amplifier * max(0.0, scalar_mean - scalar_tail)
            if scalar_mean <= 1e-12:
                return component_means
            scale = max(0.0, robust_scalar / scalar_mean)
            return tuple(value * scale for value in component_means)

        for _ in range(int(config["sample_size"])):
            permutation, proposal_prob = self._sample_permutation(rng, satellites, proposal_weights)
            sample_records.append((permutation, proposal_prob))
            raw_weights.append(1.0 / proposal_prob)

        clip_ratio = float(config.get("stratified_weight_clip_ratio", 0.0))
        if bool(config["use_stratified_sampling"]) and clip_ratio > 0.0 and raw_weights:
            ordered_weights = sorted(raw_weights)
            median_weight = ordered_weights[len(ordered_weights) // 2]
            cap = median_weight * clip_ratio
            raw_weights = [min(weight, cap) for weight in raw_weights]

        total_weight = sum(raw_weights)
        normalized_sample_weights = [weight / total_weight for weight in raw_weights]
        for (permutation, _), sample_weight in zip(sample_records, normalized_sample_weights):
            coalition: set[str] = set()
            value_before = (0.0, 0.0, 0.0)
            for sat_id in permutation:
                if constrained and coalition and not any(neighbor in coalition for neighbor in env.aggregate_graph.get(sat_id, {})):
                    singleton_value = evaluate_coalition({sat_id})
                    pruned_calls += 1
                    marginal = singleton_value
                else:
                    coalition_after = set(coalition)
                    coalition_after.add(sat_id)
                    value_after = evaluate_coalition(coalition_after)
                    marginal = tuple(after - before for after, before in zip(value_after, value_before))
                    value_before = value_after
                    oracle_calls += 1
                coalition.add(sat_id)
                aggregates[sat_id][0] += sample_weight * marginal[0]
                aggregates[sat_id][1] += sample_weight * marginal[1]
                aggregates[sat_id][2] += sample_weight * marginal[2]

        structured_values = {sat_id: tuple(values) for sat_id, values in aggregates.items()}
        metadata = {
            "oracle_calls": oracle_calls,
            "pruned_calls": pruned_calls,
            "pruning_rate": pruned_calls / max(pruned_calls + oracle_calls, 1),
            "component_weights": {
                "cmp": component_weights[0],
                "net": component_weights[1],
                "idle": component_weights[2],
            },
            "use_structured_value": use_structured_value,
            "use_topology_features": use_topology_features,
            "stratified_weight_clip_ratio": clip_ratio,
            "use_cvar": use_cvar,
            "cvar_lambda": cvar_lambda,
            "cvar_alpha": cvar_alpha,
            "cvar_risk_amplifier": cvar_risk_amplifier,
            "cvar_topology_sample_count": len(topology_samples),
        }
        return structured_values, metadata

    def deploy(
        self,
        window_stats: DeploymentWindowStats,
        env: SatelliteEnvironment,
        service_catalog: dict[str, ServiceSpec],
        config: dict,
    ) -> DeploymentPlan:
        cpf, role_vectors, _ = service_role_statistics(window_stats)
        structured_values, estimation_meta = self._estimate_structured_values(window_stats, env, config)
        if bool(config.get("use_cvar", False)) and config.get("cvar_topology_samples"):
            topology_samples = list(config.get("cvar_topology_samples") or [])
            lambda_cvar = float(config.get("cvar_lambda", 0.0))
            alpha = float(config.get("cvar_alpha", 0.9))
            risk_amplifier = float(config.get("cvar_risk_amplifier", 1.0))
            sample_feature_maps = [satellite_feature_maps(sample_env) for sample_env in topology_samples]
            sample_degrees = {
                sat_id: [
                    features["degree"].get(sat_id, 0.0)
                    for features in sample_feature_maps
                ]
                for sat_id in env.satellites
            }
            robust_degree = {}
            for sat_id, values in sample_degrees.items():
                ordered = sorted(values)
                tail_count = max(1, math.ceil((1.0 - alpha) * len(ordered)))
                lower_tail = mean(ordered[:tail_count])
                robust_degree[sat_id] = (1.0 - lambda_cvar) * (mean(values) if values else 0.0) + lambda_cvar * lower_tail
            robust_degree = min_max_normalize(robust_degree)
            boost_scale = min(1.5, 0.25 * risk_amplifier) * lambda_cvar
            structured_values = {
                sat_id: (
                    cmp_value,
                    net_value * (1.0 + boost_scale * robust_degree.get(sat_id, 0.0)),
                    idle_value * (1.0 + 0.35 * boost_scale * robust_degree.get(sat_id, 0.0)),
                )
                for sat_id, (cmp_value, net_value, idle_value) in structured_values.items()
            }
            estimation_meta["robust_degree"] = robust_degree
            estimation_meta["robust_degree_boost_scale"] = boost_scale
        positive_values = {
            sat_id: (
                max(0.0, cmp_value),
                max(0.0, net_value),
                max(0.0, idle_value),
            )
            for sat_id, (cmp_value, net_value, idle_value) in structured_values.items()
        }
        scalar_priority = {
            sat_id: sum(values)
            for sat_id, values in positive_values.items()
        }
        priority_for_sort = {
            sat_id: scalar_priority[sat_id]
            for sat_id in scalar_priority
        }
        priority_for_metadata = min_max_normalize(priority_for_sort)
        replica_targets = coverage_replica_targets(list(service_catalog), cpf, config)
        service_scores: dict[str, dict[str, float]] = defaultdict(dict)
        use_role_matching = bool(
            config.get("use_role_matching_deployment", config.get("use_structured_deployment", True))
        )
        cpu_weight_signal = {
            service: max(cpf.get(service, 0.0), 1e-6)
            for service in service_catalog
        }
        if use_role_matching:
            for service in service_catalog:
                rho_cmp, rho_net, rho_idle = role_vectors.get(service, (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0))
                for sat_id, (cmp_value, net_value, idle_value) in positive_values.items():
                    service_scores[service][sat_id] = (
                        rho_cmp * cmp_value
                        + rho_net * net_value
                        + rho_idle * idle_value
                    )
            service_order = sorted(service_catalog, key=lambda service: (cpf.get(service, 0.0), cpu_weight_signal.get(service, 0.0)), reverse=True)
        else:
            aggregate_score = normalize_dict(
                {
                    service: (
                        cpf.get(service, 0.0)
                        + window_stats.service_workload.get(service, 0.0)
                        + sum(
                            volume
                            for (src_service, dst_service), volume in window_stats.service_traffic.items()
                            if src_service == service or dst_service == service
                        )
                    )
                    for service in service_catalog
                }
            )
            for service in service_catalog:
                for sat_id, scalar_value in scalar_priority.items():
                    service_scores[service][sat_id] = scalar_value
            service_order = sorted(service_catalog, key=lambda service: aggregate_score.get(service, 0.0), reverse=True)
        metadata = {
            "window_id": window_stats.window_id,
            "cpf": cpf,
            "role_vectors": {
                service: {"cmp": values[0], "net": values[1], "idle": values[2]}
                for service, values in role_vectors.items()
            },
            "structured_values": {
                sat_id: {"cmp": values[0], "net": values[1], "idle": values[2]}
                for sat_id, values in structured_values.items()
            },
            "use_role_matching_deployment": use_role_matching,
            "scalar_priority": priority_for_metadata,
            "scalar_priority_raw": priority_for_sort,
            "replica_targets": replica_targets,
            **estimation_meta,
        }
        return place_services_by_score(
            algorithm_name=self.name,
            env=env,
            service_catalog=service_catalog,
            service_order=service_order,
            replica_targets=replica_targets,
            satellite_scores=service_scores,
            cpu_weight_signal=cpu_weight_signal,
            metadata=metadata,
            satellite_priority=priority_for_sort,
            score_weighted_cpu=use_role_matching,
        )
