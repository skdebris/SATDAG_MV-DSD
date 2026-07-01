from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = ROOT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sat_dag_service_deployment.simulator.config import load_config
from sat_dag_service_deployment.simulator.orchestrator import run_simulation


DEFAULT_BASE_CONFIG = ROOT_DIR / "configs" / "default_simulation.json"
DEFAULT_SCENARIO_DIR = ROOT_DIR / "configs" / "scenarios"
DEFAULT_OUTPUT_FILE = ROOT_DIR / "simulator" / "outputs" / "experiment4_m_sweep_fixed.json"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _safe_rel_error(value: float, reference: float) -> float:
    if reference == 0:
        return 0.0 if value == 0 else math.inf
    return abs(value - reference) / abs(reference)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiment 4: fixed-deployment sample-size sweep.")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--scenario-dir", type=Path, default=DEFAULT_SCENARIO_DIR)
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=["normal_nominal", "sparse_topology_stress", "sparse_fluctuation_stress"],
    )
    parser.add_argument("--sample-sizes", nargs="*", type=int, default=[50, 100, 200, 300, 500, 800])
    parser.add_argument("--algorithm-seeds", nargs="*", type=int, default=[20260427, 20260428, 20260429])
    parser.add_argument("--environment-seed", type=int, default=20260427)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--run-prefix", type=str, default="exp4_m")
    parser.add_argument("--tcr-tol", type=float, default=0.01)
    parser.add_argument("--makespan-rel-tol", type=float, default=0.03)
    return parser


def _load_deployment_metadata(output_dir: Path) -> dict[str, Any]:
    plans = json.loads((output_dir / "deployment_plans.json").read_text(encoding="utf-8"))
    return plans[0]["metadata"]


def main() -> None:
    args = build_parser().parse_args()
    base_config = load_config(args.base_config)
    raw_records: list[dict[str, Any]] = []

    for scenario_name in args.scenarios:
        scenario_path = args.scenario_dir / f"{scenario_name}.json"
        scenario_override = json.loads(scenario_path.read_text(encoding="utf-8"))
        for sample_size in args.sample_sizes:
            for algorithm_seed in args.algorithm_seeds:
                config = _deep_merge(base_config, scenario_override)
                config["environment"]["seed"] = args.environment_seed
                config["algorithm"]["seed"] = algorithm_seed
                config["algorithm"]["name"] = "cpmv_dsd"
                config["algorithm"]["sample_size"] = sample_size
                config["output"]["run_name"] = (
                    f"{args.run_prefix}_{scenario_name}_M{sample_size}_aseed{algorithm_seed}_eseed{args.environment_seed}"
                )
                result = run_simulation(config)
                summary = result["summary"]
                output_dir = Path(result["manifest"]["output_dir"])
                deployment_metadata = _load_deployment_metadata(output_dir)
                raw_records.append(
                    {
                        "scenario_name": scenario_name,
                        "scenario_path": str(scenario_path),
                        "sample_size": sample_size,
                        "algorithm_seed": algorithm_seed,
                        "environment_seed": args.environment_seed,
                        "task_completion_rate": summary["task_completion_rate"],
                        "deadline_satisfaction_ratio": summary.get("deadline_satisfaction_ratio", summary["task_completion_rate"]),
                        "raw_task_completion_rate": summary.get("raw_task_completion_rate", summary["task_completion_rate"]),
                        "mean_makespan_minutes": summary["mean_makespan_minutes"],
                        "p95_makespan_minutes": summary["p95_makespan_minutes"],
                        "planning_runtime_seconds": summary["metadata"]["planning_runtime_seconds"],
                        "total_runtime_seconds": summary["metadata"]["total_runtime_seconds"],
                        "oracle_calls": deployment_metadata.get("oracle_calls", 0),
                        "pruned_calls": deployment_metadata.get("pruned_calls", 0),
                        "pruning_rate": deployment_metadata.get("pruning_rate", 0.0),
                    }
                )

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in raw_records:
        grouped[(record["scenario_name"], record["sample_size"])].append(record)

    aggregates: list[dict[str, Any]] = []
    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (scenario_name, sample_size), records in sorted(grouped.items()):
        aggregate = {
            "scenario_name": scenario_name,
            "sample_size": sample_size,
            "seed_count": len(records),
            "task_completion_rate_mean": mean(record["task_completion_rate"] for record in records),
            "task_completion_rate_std": pstdev(record["task_completion_rate"] for record in records),
            "deadline_satisfaction_ratio_mean": mean(record["deadline_satisfaction_ratio"] for record in records),
            "raw_task_completion_rate_mean": mean(record["raw_task_completion_rate"] for record in records),
            "mean_makespan_minutes_mean": mean(record["mean_makespan_minutes"] for record in records),
            "mean_makespan_minutes_std": pstdev(record["mean_makespan_minutes"] for record in records),
            "p95_makespan_minutes_mean": mean(record["p95_makespan_minutes"] for record in records),
            "planning_runtime_seconds_mean": mean(record["planning_runtime_seconds"] for record in records),
            "total_runtime_seconds_mean": mean(record["total_runtime_seconds"] for record in records),
            "oracle_calls_mean": mean(record["oracle_calls"] for record in records),
            "pruned_calls_mean": mean(record["pruned_calls"] for record in records),
            "pruning_rate_mean": mean(record["pruning_rate"] for record in records),
        }
        aggregates.append(aggregate)
        by_scenario[scenario_name].append(aggregate)

    recommendation_checks: list[dict[str, Any]] = []
    for scenario_name, rows in by_scenario.items():
        rows.sort(key=lambda item: item["sample_size"])
        reference = max(rows, key=lambda item: item["sample_size"])
        for row in rows:
            row["reference_sample_size"] = reference["sample_size"]
            row["tcr_gap_vs_reference"] = abs(row["task_completion_rate_mean"] - reference["task_completion_rate_mean"])
            row["makespan_rel_error_vs_reference"] = _safe_rel_error(
                row["mean_makespan_minutes_mean"],
                reference["mean_makespan_minutes_mean"],
            )
            recommendation_checks.append(row)

    recommended_sample_size: int | None = None
    for sample_size in sorted(set(args.sample_sizes)):
        eligible = [row for row in recommendation_checks if row["sample_size"] == sample_size]
        if not eligible:
            continue
        if all(
            row["tcr_gap_vs_reference"] <= args.tcr_tol
            and row["makespan_rel_error_vs_reference"] <= args.makespan_rel_tol
            for row in eligible
        ):
            recommended_sample_size = sample_size
            break
    if recommended_sample_size is None:
        recommended_sample_size = max(args.sample_sizes)

    output = {
        "experiment": "experiment4_m_sweep_fixed",
        "scenarios": args.scenarios,
        "sample_sizes": args.sample_sizes,
        "algorithm_seeds": args.algorithm_seeds,
        "environment_seed": args.environment_seed,
        "selection_rule": {
            "tcr_tolerance": args.tcr_tol,
            "makespan_relative_tolerance": args.makespan_rel_tol,
            "reference_sample_size": max(args.sample_sizes),
        },
        "recommended_sample_size": recommended_sample_size,
        "raw_records": raw_records,
        "aggregates": aggregates,
    }

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = args.output_file.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "scenario",
                "sample_size",
                "seed_count",
                "tcr_mean",
                "tcr_std",
                "dsr_mean",
                "raw_tcr_mean",
                "mean_makespan_mean",
                "mean_makespan_std",
                "p95_makespan_mean",
                "planning_runtime_mean",
                "total_runtime_mean",
                "oracle_calls_mean",
                "pruned_calls_mean",
                "pruning_rate_mean",
                "tcr_gap_vs_reference",
                "makespan_rel_error_vs_reference",
            ]
        )
        for row in sorted(aggregates, key=lambda item: (item["scenario_name"], item["sample_size"])):
            writer.writerow(
                [
                    row["scenario_name"],
                    row["sample_size"],
                    row["seed_count"],
                    row["task_completion_rate_mean"],
                    row["task_completion_rate_std"],
                    row["deadline_satisfaction_ratio_mean"],
                    row["raw_task_completion_rate_mean"],
                    row["mean_makespan_minutes_mean"],
                    row["mean_makespan_minutes_std"],
                    row["p95_makespan_minutes_mean"],
                    row["planning_runtime_seconds_mean"],
                    row["total_runtime_seconds_mean"],
                    row["oracle_calls_mean"],
                    row["pruned_calls_mean"],
                    row["pruning_rate_mean"],
                    row.get("tcr_gap_vs_reference", 0.0),
                    row.get("makespan_rel_error_vs_reference", 0.0),
                ]
            )

    print(
        json.dumps(
            {
                "output_json": str(args.output_file),
                "output_csv": str(csv_path),
                "recommended_sample_size": recommended_sample_size,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
