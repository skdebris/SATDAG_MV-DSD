from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    return mean(values), (0.0 if len(values) == 1 else pstdev(values))


def cpu_utilization_proxy(per_request_results: list[dict[str, Any]]) -> float:
    samples: list[float] = []
    for result in per_request_results:
        values = list((result.get("cpu_utilization") or {}).values())
        if values:
            samples.append(sum(values) / len(values))
    return mean(samples) if samples else 0.0


def summarize_seed_runs(seed_runs: list[dict[str, Any]]) -> dict[str, Any]:
    fields = {
        "task_completion_rate": [run["summary"]["task_completion_rate"] for run in seed_runs],
        "deadline_satisfaction_ratio": [
            run["summary"].get("deadline_satisfaction_ratio", run["summary"]["task_completion_rate"])
            for run in seed_runs
        ],
        "raw_task_completion_rate": [
            run["summary"].get("raw_task_completion_rate", run["summary"]["task_completion_rate"])
            for run in seed_runs
        ],
        "mean_normalized_tardiness": [
            run["summary"].get("mean_normalized_tardiness", 0.0)
            for run in seed_runs
        ],
        "mean_makespan_minutes": [run["summary"]["mean_makespan_minutes"] for run in seed_runs],
        "p95_makespan_minutes": [run["summary"]["p95_makespan_minutes"] for run in seed_runs],
        "p99_makespan_minutes": [run["summary"]["p99_makespan_minutes"] for run in seed_runs],
        "mean_cp_cmp_minutes": [run["summary"]["mean_cp_cmp_minutes"] for run in seed_runs],
        "mean_cp_net_minutes": [run["summary"]["mean_cp_net_minutes"] for run in seed_runs],
        "mean_cp_idle_minutes": [run["summary"]["mean_cp_idle_minutes"] for run in seed_runs],
        "failed_count": [float(run["summary"]["failed_count"]) for run in seed_runs],
        "total_runtime_seconds": [run["summary"]["metadata"]["total_runtime_seconds"] for run in seed_runs],
        "planning_runtime_seconds": [run["summary"]["metadata"]["planning_runtime_seconds"] for run in seed_runs],
        "cpu_utilization_proxy": [run["cpu_utilization_proxy"] for run in seed_runs],
    }
    output: dict[str, Any] = {}
    for field, values in fields.items():
        avg, std = mean_std(values)
        output[f"{field}_mean"] = avg
        output[f"{field}_std"] = std
    return output


def write_outputs(output_file: Path, payload: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = output_file.with_suffix(".csv")
    fieldnames = [
        "experiment",
        "scenario",
        "algorithm",
        "scheduler",
        "group",
        "seed_count",
        "task_completion_rate_mean",
        "task_completion_rate_std",
        "deadline_satisfaction_ratio_mean",
        "deadline_satisfaction_ratio_std",
        "raw_task_completion_rate_mean",
        "raw_task_completion_rate_std",
        "mean_normalized_tardiness_mean",
        "mean_normalized_tardiness_std",
        "mean_makespan_minutes_mean",
        "mean_makespan_minutes_std",
        "p95_makespan_minutes_mean",
        "p95_makespan_minutes_std",
        "p99_makespan_minutes_mean",
        "p99_makespan_minutes_std",
        "mean_cp_cmp_minutes_mean",
        "mean_cp_net_minutes_mean",
        "mean_cp_idle_minutes_mean",
        "cpu_utilization_proxy_mean",
        "planning_runtime_seconds_mean",
        "total_runtime_seconds_mean",
        "failed_count_mean",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
