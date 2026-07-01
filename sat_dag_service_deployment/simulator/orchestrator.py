from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .algorithms import build_algorithm
from .config import load_config
from .data import (
    build_deployment_window_stats,
    ensure_materialized_requests,
    load_materialized_requests,
    load_service_catalog,
    split_requests_into_windows,
)
from .env import build_satellite_environment
from .evaluation import summarize_results
from .models import DeploymentPlan, DeploymentWindow, DeploymentWindowStats, ExecutionResult
from .schedulers import build_scheduler
from .utils import dump_json, ensure_dir


def _serialize_deployment_plan(plan: DeploymentPlan) -> dict[str, Any]:
    return {
        "algorithm_name": plan.algorithm_name,
        "window_id": plan.window_id,
        "service_placement": plan.service_placement,
        "cpu_allocation": {
            f"{sat_id}|{service}": value
            for (sat_id, service), value in plan.cpu_allocation.items()
        },
        "replica_count": plan.replica_count,
        "metadata": plan.metadata,
    }


def _serialize_execution_result(result: ExecutionResult) -> dict[str, Any]:
    return {
        "request_id": result.request_id,
        "window_id": result.window_id,
        "success": result.success,
        "arrival_time_days": result.arrival_time_days,
        "start_time_days": result.start_time_days,
        "finish_time_days": result.finish_time_days,
        "makespan_days": result.makespan_days,
        "cp_delay_breakdown_days": result.cp_delay_breakdown_days,
        "cpu_utilization": result.cpu_utilization,
        "failure_reason": result.failure_reason,
        "metadata": result.metadata,
    }


def _prepare_requests(config: dict) -> tuple[Path, list]:
    request_path = Path(config["data"]["job_requests_path"])
    if config["simulation"].get("request_source_mode", "materialized") == "materialized":
        if config["simulation"].get("persist_materialized_requests", True):
            ensure_materialized_requests(
                arrival_trace_path=config["data"]["arrival_trace_path"],
                template_dir=config["data"]["template_dir"],
                output_path=request_path,
            )
        elif not request_path.exists():
            ensure_materialized_requests(
                arrival_trace_path=config["data"]["arrival_trace_path"],
                template_dir=config["data"]["template_dir"],
                output_path=request_path,
            )
    requests = load_materialized_requests(request_path)
    return request_path, requests


def _build_fixed_planning_stats(
    windows,
    planning_window_count: int | None,
    planning_max_requests: int | None,
) -> DeploymentWindowStats:
    selected_windows = windows if planning_window_count is None else windows[:planning_window_count]
    if not selected_windows:
        raise ValueError("Fixed deployment mode requires at least one non-empty execution window.")
    planning_requests = [
        request
        for window in selected_windows
        for request in window.requests
    ]
    if planning_max_requests is not None:
        planning_requests = planning_requests[:planning_max_requests]
    planning_window = DeploymentWindow(
        window_id=selected_windows[0].window_id,
        start_time_days=selected_windows[0].start_time_days,
        end_time_days=selected_windows[-1].end_time_days,
        requests=planning_requests,
    )
    return build_deployment_window_stats(
        planning_window,
        max_requests_per_window=planning_max_requests,
    )


def run_simulation(config_or_path: dict | str | Path | None = None) -> dict[str, Any]:
    if config_or_path is None:
        config = load_config()
    elif isinstance(config_or_path, (str, Path)):
        config = load_config(config_or_path)
    else:
        config = config_or_path

    service_catalog = load_service_catalog(config["data"]["service_catalog_path"])
    request_dataset_path, requests = _prepare_requests(config)
    windows = split_requests_into_windows(
        requests,
        deployment_period_minutes=float(config["simulation"]["deployment_period_minutes"]),
        max_windows=int(config["simulation"]["max_windows"]) if config["simulation"].get("max_windows") is not None else None,
    )

    scheduler = build_scheduler(config["scheduler"]["name"])
    algorithm_seed = (
        int(config["algorithm"]["seed"])
        if config["algorithm"].get("seed") is not None
        else int(config["seed"])
    )
    environment_seed = (
        int(config["environment"]["seed"])
        if config["environment"].get("seed") is not None
        else int(config["seed"])
    )
    algorithm = build_algorithm(config["algorithm"]["name"], seed=algorithm_seed)
    deployment_plans: list[DeploymentPlan] = []
    execution_results: list[ExecutionResult] = []
    per_window_stats = []
    started = time.perf_counter()
    deployment_mode = str(config["simulation"].get("deployment_mode", "fixed"))
    planning_runtime_seconds = 0.0
    deployment_plan_fixed: DeploymentPlan | None = None

    if deployment_mode == "fixed":
        planning_stats = _build_fixed_planning_stats(
            windows=windows,
            planning_window_count=(
                int(config["simulation"]["planning_window_count"])
                if config["simulation"].get("planning_window_count") is not None
                else None
            ),
            planning_max_requests=(
                int(config["simulation"]["planning_max_requests"])
                if config["simulation"].get("planning_max_requests") is not None
                else None
            ),
        )
        planning_reference_window_id = int(config["simulation"].get("planning_reference_window_id", 0))
        planning_env = build_satellite_environment(
            config=config["environment"],
            window_id=planning_reference_window_id,
            seed=environment_seed,
        )
        algorithm_started = time.perf_counter()
        deployment_plan_fixed = algorithm.deploy(
            window_stats=planning_stats,
            env=planning_env,
            service_catalog=service_catalog,
            config=config["algorithm"],
        )
        planning_runtime_seconds = time.perf_counter() - algorithm_started
        deployment_plan_fixed.metadata.update(
            {
                "deployment_mode": "fixed",
                "planning_request_count": planning_stats.request_count,
                "planning_window_start_id": planning_stats.window_id,
                "planning_window_count": (
                    len(windows)
                    if config["simulation"].get("planning_window_count") is None
                    else int(config["simulation"]["planning_window_count"])
                ),
                "planning_reference_window_id": planning_reference_window_id,
            }
        )
        deployment_plans.append(deployment_plan_fixed)

    for window in windows:
        scheduler.reset_window(window)
        window_stats = build_deployment_window_stats(
            window,
            max_requests_per_window=int(config["simulation"]["max_requests_per_window"])
            if config["simulation"].get("max_requests_per_window") is not None
            else None,
        )
        env = build_satellite_environment(
            config=config["environment"],
            window_id=window.window_id,
            seed=environment_seed,
        )
        if deployment_mode == "fixed":
            deployment_plan = deployment_plan_fixed
            algorithm_runtime = 0.0
        else:
            algorithm_started = time.perf_counter()
            deployment_plan = algorithm.deploy(
                window_stats=window_stats,
                env=env,
                service_catalog=service_catalog,
                config=config["algorithm"],
            )
            algorithm_runtime = time.perf_counter() - algorithm_started
            deployment_plans.append(deployment_plan)
        window_started = time.perf_counter()
        for request in window_stats.requests:
            result = scheduler.schedule(
                request=request,
                deployment_plan=deployment_plan,
                env=env,
                current_time=request.arrival_time_days,
                config=config["scheduler"],
            )
            execution_results.append(result)
        per_window_stats.append(
            {
                "window_id": window.window_id,
                "request_count": len(window_stats.requests),
                "arrival_rate_per_day": window_stats.arrival_rate_per_day,
                "algorithm_runtime_seconds": algorithm_runtime,
                "deployment_recomputed": deployment_mode != "fixed",
                "window_runtime_seconds": time.perf_counter() - window_started,
            }
        )

    total_runtime = time.perf_counter() - started
    output_dir = ensure_dir(Path(config["output"]["output_dir"]) / config["output"]["run_name"])
    summary = summarize_results(
        execution_results=execution_results,
        deployment_plans=deployment_plans,
        metadata={
            "algorithm": config["algorithm"]["name"],
            "algorithm_seed": algorithm_seed,
            "scheduler": config["scheduler"]["name"],
            "deployment_mode": deployment_mode,
            "density_mode": config["environment"]["density_mode"],
            "environment_seed": environment_seed,
            "perturbation_mode": config["environment"]["perturbation_mode"],
            "total_runtime_seconds": total_runtime,
            "planning_runtime_seconds": planning_runtime_seconds,
            "deployment_recomputations": len(deployment_plans),
            "window_count": len(windows),
            "per_window_stats": per_window_stats,
        },
    )
    summary_path = output_dir / "summary.json"
    dump_json(summary_path, summary)
    dump_json(output_dir / "deployment_plans.json", [_serialize_deployment_plan(plan) for plan in deployment_plans])
    if config["output"].get("save_per_request_results", True):
        dump_json(output_dir / "per_request_results.json", [_serialize_execution_result(result) for result in execution_results])

    manifest = {
        "config": config,
        "request_dataset_path": str(request_dataset_path),
        "summary_path": str(summary_path),
        "output_dir": str(output_dir),
    }
    dump_json(output_dir / "manifest.json", manifest)
    return {"summary": summary, "manifest": manifest}
