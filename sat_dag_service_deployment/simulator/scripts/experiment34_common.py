from __future__ import annotations

import copy
import csv
import json
import math
import random
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from sat_dag_service_deployment.simulator.algorithms import build_algorithm
from sat_dag_service_deployment.simulator.data import (
    build_deployment_window_stats,
    ensure_materialized_requests,
    load_materialized_requests,
    load_service_catalog,
    split_requests_into_windows,
)
from sat_dag_service_deployment.simulator.env import build_satellite_environment
from sat_dag_service_deployment.simulator.env.constellation import _build_aggregate_graph, _build_contact_plan
from sat_dag_service_deployment.simulator.evaluation import summarize_results
from sat_dag_service_deployment.simulator.models import DeploymentPlan, DeploymentWindow, GraphSnapshot, LinkState, SatelliteEnvironment
from sat_dag_service_deployment.simulator.schedulers import build_scheduler
from sat_dag_service_deployment.simulator.utils import percentile


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def empirical_cvar(values: list[float], alpha: float = 0.9) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    threshold = percentile(ordered, alpha)
    tail = [value for value in ordered if value >= threshold]
    return mean(tail) if tail else threshold


def _days_to_minutes(value: float) -> float:
    return value * 24.0 * 60.0


def summarize_execution_results(execution_results, deployment_plans, alpha: float = 0.9, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    summary = summarize_results(execution_results, deployment_plans, metadata=metadata or {})
    completed = [result for result in execution_results if result.success]
    makespan_minutes = [_days_to_minutes(result.makespan_days) for result in completed]
    completed_deadline_misses = sum(
        1
        for result in completed
        if not bool(result.metadata.get("deadline_satisfied", result.success))
    )
    summary.update(
        {
            "cvar_makespan_minutes": empirical_cvar(makespan_minutes, alpha),
            "completed_deadline_miss_rate": completed_deadline_misses / max(len(completed), 1),
            "cvar_alpha": alpha,
        }
    )
    return summary


def perturb_environment(
    env: SatelliteEnvironment,
    p_fail: float,
    bw_sigma: float,
    seed: int,
    aggregate_kappa: float,
) -> SatelliteEnvironment:
    rng = random.Random(seed)
    snapshots: list[GraphSnapshot] = []
    for snapshot in env.snapshots:
        pair_drop: dict[tuple[str, str], bool] = {}
        pair_factor: dict[tuple[str, str], float] = {}
        adjacency: dict[str, list[LinkState]] = {sat_id: [] for sat_id in env.satellites}
        for src in sorted(snapshot.adjacency):
            for link in sorted(snapshot.adjacency[src], key=lambda item: item.dst):
                a, b = sorted((link.src, link.dst))
                pair = (a, b)
                if pair not in pair_drop:
                    pair_drop[pair] = rng.random() < p_fail
                    pair_factor[pair] = rng.lognormvariate(-0.5 * bw_sigma * bw_sigma, bw_sigma)
                if pair_drop[pair]:
                    continue
                adjacency[link.src].append(
                    LinkState(
                        src=link.src,
                        dst=link.dst,
                        capacity_mbps=max(80.0, round(link.capacity_mbps * pair_factor[pair], 4)),
                        propagation_delay_ms=link.propagation_delay_ms,
                    )
                )
        snapshots.append(
            GraphSnapshot(
                slot_index=snapshot.slot_index,
                time_offset_minutes=snapshot.time_offset_minutes,
                adjacency=adjacency,
            )
        )

    contact_plan = _build_contact_plan(
        snapshots=snapshots,
        slot_duration_minutes=env.slot_duration_minutes,
    )
    aggregate_graph, threshold = _build_aggregate_graph(
        contact_plan=contact_plan,
        aggregate_kappa=aggregate_kappa,
    )
    metadata = dict(env.metadata)
    metadata.update(
        {
            "perturb_p_fail": p_fail,
            "perturb_bw_sigma": bw_sigma,
            "perturb_seed": seed,
            "num_contact_plan_edges": len(contact_plan),
        }
    )
    return SatelliteEnvironment(
        satellites=env.satellites,
        snapshots=snapshots,
        contact_plan=contact_plan,
        aggregate_graph=aggregate_graph,
        aggregate_threshold=threshold,
        slot_duration_minutes=env.slot_duration_minutes,
        planning_horizon_minutes=env.planning_horizon_minutes,
        density_mode=env.density_mode,
        perturbation_mode=f"sampled_p{p_fail}_bw{bw_sigma}",
        metadata=metadata,
    )


def load_windows_and_catalog(config: dict):
    ensure_materialized_requests(
        arrival_trace_path=config["data"]["arrival_trace_path"],
        template_dir=config["data"]["template_dir"],
        output_path=config["data"]["job_requests_path"],
    )
    service_catalog = load_service_catalog(config["data"]["service_catalog_path"])
    requests = load_materialized_requests(config["data"]["job_requests_path"])
    windows = split_requests_into_windows(
        requests,
        deployment_period_minutes=float(config["simulation"]["deployment_period_minutes"]),
        max_windows=int(config["simulation"]["max_windows"]) if config["simulation"].get("max_windows") is not None else None,
    )
    return windows, service_catalog


def fixed_planning_stats(config: dict, windows: list[DeploymentWindow]):
    selected_windows = windows
    if config["simulation"].get("planning_window_count") is not None:
        selected_windows = windows[: int(config["simulation"]["planning_window_count"])]
    planning_requests = [request for window in selected_windows for request in window.requests]
    if config["simulation"].get("planning_max_requests") is not None:
        planning_requests = planning_requests[: int(config["simulation"]["planning_max_requests"])]
    planning_window = DeploymentWindow(
        window_id=selected_windows[0].window_id,
        start_time_days=selected_windows[0].start_time_days,
        end_time_days=selected_windows[-1].end_time_days,
        requests=planning_requests,
    )
    return build_deployment_window_stats(
        planning_window,
        max_requests_per_window=(
            int(config["simulation"]["planning_max_requests"])
            if config["simulation"].get("planning_max_requests") is not None
            else None
        ),
    )


def deploy_fixed_plan(config: dict, windows: list[DeploymentWindow], service_catalog: dict, env: SatelliteEnvironment) -> tuple[DeploymentPlan, float]:
    algorithm_seed = int(config["algorithm"]["seed"] if config["algorithm"].get("seed") is not None else config["seed"])
    algorithm = build_algorithm(config["algorithm"]["name"], seed=algorithm_seed)
    stats = fixed_planning_stats(config, windows)
    started = time.perf_counter()
    plan = algorithm.deploy(stats, env, service_catalog, config["algorithm"])
    runtime = time.perf_counter() - started
    plan.metadata.update({"deployment_mode": "fixed", "planning_request_count": stats.request_count})
    return plan, runtime


def evaluate_fixed_plan(
    config: dict,
    windows: list[DeploymentWindow],
    deployment_plan: DeploymentPlan,
    env_by_window: dict[int, SatelliteEnvironment],
    alpha: float = 0.9,
) -> tuple[dict[str, Any], list]:
    scheduler = build_scheduler(config["scheduler"]["name"])
    execution_results = []
    max_requests = (
        int(config["simulation"]["max_requests_per_window"])
        if config["simulation"].get("max_requests_per_window") is not None
        else None
    )
    for window in windows:
        scheduler.reset_window(window)
        stats = build_deployment_window_stats(window, max_requests_per_window=max_requests)
        env = env_by_window[window.window_id]
        for request in stats.requests:
            execution_results.append(
                scheduler.schedule(
                    request=request,
                    deployment_plan=deployment_plan,
                    env=env,
                    current_time=request.arrival_time_days,
                    config=config["scheduler"],
                )
            )
    summary = summarize_execution_results(
        execution_results,
        [deployment_plan],
        alpha=alpha,
        metadata={
            "algorithm": deployment_plan.algorithm_name,
            "scheduler": config["scheduler"]["name"],
            "deadline_minutes": config["simulation"].get("t_max_minutes", 30.0),
        },
    )
    return summary, execution_results


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    return mean(values), (0.0 if len(values) == 1 else pstdev(values))


def topk_jaccard(rankings: list[list[str]], k: int) -> float:
    if len(rankings) < 2:
        return 1.0
    samples = [set(ranking[:k]) for ranking in rankings]
    scores: list[float] = []
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            union = samples[i] | samples[j]
            scores.append(len(samples[i] & samples[j]) / max(len(union), 1))
    return mean(scores) if scores else 1.0


def contribution_variance(score_by_seed: list[dict[str, float]]) -> float:
    if len(score_by_seed) < 2:
        return 0.0
    satellites = sorted({sat_id for scores in score_by_seed for sat_id in scores})
    variances: list[float] = []
    for sat_id in satellites:
        values = [scores.get(sat_id, 0.0) for scores in score_by_seed]
        variances.append(pstdev(values) ** 2)
    return mean(variances) if variances else 0.0


def write_json_csv(output_file: Path, payload: dict[str, Any], rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with output_file.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
