from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = ROOT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiment2_common import cpu_utilization_proxy, deep_merge, summarize_seed_runs, write_outputs
from sat_dag_service_deployment.simulator.config import load_config
from sat_dag_service_deployment.simulator.orchestrator import run_simulation


DEFAULT_BASE_CONFIG = ROOT_DIR / "configs" / "default_simulation.json"
DEFAULT_SCENARIO_DIR = ROOT_DIR / "configs" / "scenarios"
DEFAULT_OUTPUT_FILE = ROOT_DIR / "simulator" / "outputs" / "exp2" / "aggregate" / "experiment2_main_comparison_M500_seed3.json"
DEFAULT_REQUEST_DATASET = ROOT_DIR / "dag_data_set" / "dataset" / "job_requests" / "requests_T30days.jsonl"


def _days_to_minutes(value: float) -> float:
    return value * 24.0 * 60.0


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    return mean(values), (0.0 if len(values) == 1 else pstdev(values))


def _percentile(sorted_values: list[float], ratio: float) -> float:
    if not sorted_values:
        return 0.0
    index = int(round((len(sorted_values) - 1) * ratio))
    index = min(max(index, 0), len(sorted_values) - 1)
    return sorted_values[index]


def _load_request_metadata(dataset_path: Path) -> dict[str, dict[str, str]]:
    mapping: dict[str, dict[str, str]] = {}
    with dataset_path.open(encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            mapping[payload["request_id"]] = {
                "dag_type": payload["dag_type"],
                "subarchetype": payload["subarchetype"],
            }
    return mapping


def _has_deadline_metadata(results: list[dict[str, Any]]) -> bool:
    return any(not result.get("metadata", {}).get("deadline_missing", True) for result in results)


def _deadline_satisfied(result: dict[str, Any], deadlines_present: bool) -> bool:
    if not deadlines_present:
        return bool(result["success"])
    return bool(result.get("metadata", {}).get("deadline_satisfied", False))


def _group_metrics(per_request_results: list[dict[str, Any]], request_meta: dict[str, dict[str, str]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in per_request_results:
        grouped[request_meta[result["request_id"]]["dag_type"]].append(result)

    output: dict[str, dict[str, Any]] = {}
    for dag_type, results in sorted(grouped.items()):
        deadlines_present = _has_deadline_metadata(results)
        raw_completed = [result for result in results if result["success"]]
        deadline_completed = [result for result in results if _deadline_satisfied(result, deadlines_present)]
        makespans = sorted(_days_to_minutes(result["makespan_days"]) for result in raw_completed)
        cp_cmp = [_days_to_minutes(result["cp_delay_breakdown_days"]["cmp"]) for result in raw_completed]
        cp_net = [_days_to_minutes(result["cp_delay_breakdown_days"]["net"]) for result in raw_completed]
        cp_idle = [_days_to_minutes(result["cp_delay_breakdown_days"]["idle"]) for result in raw_completed]
        tardiness = [float(result.get("metadata", {}).get("normalized_tardiness", 0.0)) for result in results]
        failures = Counter(result["failure_reason"] for result in results if result["failure_reason"])
        output[dag_type] = {
            "request_count": len(results),
            "task_completion_rate": len(deadline_completed) / max(len(results), 1),
            "deadline_satisfaction_ratio": len(deadline_completed) / max(len(results), 1),
            "raw_task_completion_rate": len(raw_completed) / max(len(results), 1),
            "mean_normalized_tardiness": sum(tardiness) / len(tardiness) if tardiness else 0.0,
            "mean_makespan_minutes": sum(makespans) / len(makespans) if makespans else 0.0,
            "p95_makespan_minutes": _percentile(makespans, 0.95) if makespans else 0.0,
            "p99_makespan_minutes": _percentile(makespans, 0.99) if makespans else 0.0,
            "mean_cp_cmp_minutes": sum(cp_cmp) / len(cp_cmp) if cp_cmp else 0.0,
            "mean_cp_net_minutes": sum(cp_net) / len(cp_net) if cp_net else 0.0,
            "mean_cp_idle_minutes": sum(cp_idle) / len(cp_idle) if cp_idle else 0.0,
            "failed_count": len(results) - len(deadline_completed),
            "failure_reasons": dict(failures),
        }
    return output


def _aggregate_group_rows(seed_runs: list[dict[str, Any]], group: str) -> dict[str, Any]:
    summaries = [run["by_dag_type"][group] for run in seed_runs if group in run["by_dag_type"]]
    fields = [
        "task_completion_rate",
        "deadline_satisfaction_ratio",
        "raw_task_completion_rate",
        "mean_normalized_tardiness",
        "mean_makespan_minutes",
        "p95_makespan_minutes",
        "p99_makespan_minutes",
        "mean_cp_cmp_minutes",
        "mean_cp_net_minutes",
        "mean_cp_idle_minutes",
        "failed_count",
    ]
    output: dict[str, Any] = {}
    for field in fields:
        values = [float(summary[field]) for summary in summaries]
        avg, std = _mean_std(values)
        output[f"{field}_mean"] = avg
        output[f"{field}_std"] = std
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiment 2-A: literature-style deployment baseline comparison.")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--scenario-dir", type=Path, default=DEFAULT_SCENARIO_DIR)
    parser.add_argument("--request-dataset", type=Path, default=DEFAULT_REQUEST_DATASET)
    parser.add_argument("--scenarios", nargs="*", default=["normal_nominal", "sparse_topology_stress"])
    parser.add_argument(
        "--algorithms",
        nargs="*",
        default=[
            "cpmv_dsd",
            "jsdts_aos_sat",
            "ondoc_sat",
            "floodsfcp_greedy",
            "greedy_resource",
        ],
    )
    parser.add_argument("--scheduler", type=str, default="heft")
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--seeds", nargs="*", type=int, default=[20260427, 20260428, 20260429])
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--max-requests-per-window", type=int, default=None)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--run-prefix", type=str, default="exp2_main")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    base_config = load_config(args.base_config)
    request_meta = _load_request_metadata(args.request_dataset)
    records: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []

    for scenario_name in args.scenarios:
        scenario_override = json.loads((args.scenario_dir / f"{scenario_name}.json").read_text(encoding="utf-8"))
        for algorithm_name in args.algorithms:
            seed_runs = []
            for seed in args.seeds:
                config = deep_merge(base_config, scenario_override)
                config["environment"]["seed"] = seed
                config["algorithm"]["seed"] = seed
                config["algorithm"]["name"] = algorithm_name
                config["algorithm"]["sample_size"] = args.sample_size
                config["scheduler"]["name"] = args.scheduler
                if args.max_windows is not None:
                    config["simulation"]["max_windows"] = args.max_windows
                if args.max_requests_per_window is not None:
                    config["simulation"]["max_requests_per_window"] = args.max_requests_per_window
                config["output"]["output_dir"] = str(ROOT_DIR / "simulator" / "outputs" / "exp2" / "runs")
                config["output"]["run_name"] = f"{args.run_prefix}_{scenario_name}_{algorithm_name}_{args.scheduler}_seed{seed}"
                print(
                    f"[exp2] scenario={scenario_name} algorithm={algorithm_name} scheduler={args.scheduler} seed={seed} M={args.sample_size}",
                    flush=True,
                )
                result = run_simulation(config)
                output_dir = Path(result["manifest"]["output_dir"])
                per_request = json.loads((output_dir / "per_request_results.json").read_text(encoding="utf-8"))
                seed_runs.append(
                    {
                        "seed": seed,
                        "summary": result["summary"],
                        "by_dag_type": _group_metrics(per_request, request_meta),
                        "cpu_utilization_proxy": cpu_utilization_proxy(per_request),
                        "output_dir": str(output_dir),
                    }
                )
            aggregate = summarize_seed_runs(seed_runs)
            record = {
                "scenario_name": scenario_name,
                "algorithm_name": algorithm_name,
                "scheduler_name": args.scheduler,
                "sample_size": args.sample_size,
                "seeds": args.seeds,
                "seed_runs": seed_runs,
                "aggregate": aggregate,
            }
            records.append(record)
            csv_rows.append(
                {
                    "experiment": "experiment2_main_comparison",
                    "scenario": scenario_name,
                    "algorithm": algorithm_name,
                    "scheduler": args.scheduler,
                    "group": "overall",
                    "seed_count": len(seed_runs),
                    **aggregate,
                }
            )
            dag_groups = sorted({group for run in seed_runs for group in run["by_dag_type"]})
            for dag_group in dag_groups:
                dag_aggregate = _aggregate_group_rows(seed_runs, dag_group)
                csv_rows.append(
                    {
                        "experiment": "experiment2_main_comparison",
                        "scenario": scenario_name,
                        "algorithm": algorithm_name,
                        "scheduler": args.scheduler,
                        "group": dag_group,
                        "seed_count": len(seed_runs),
                        **dag_aggregate,
                    }
                )

    payload = {
        "experiment": "experiment2_main_comparison",
        "scheduler": args.scheduler,
        "scenarios": args.scenarios,
        "algorithms": args.algorithms,
        "sample_size": args.sample_size,
        "results": records,
    }
    write_outputs(args.output_file, payload, csv_rows)
    print(json.dumps({"output_json": str(args.output_file), "output_csv": str(args.output_file.with_suffix(".csv"))}, indent=2))


if __name__ == "__main__":
    main()
