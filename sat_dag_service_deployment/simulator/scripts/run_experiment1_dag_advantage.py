from __future__ import annotations

import argparse
import copy
import csv
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

from sat_dag_service_deployment.simulator.config import load_config
from sat_dag_service_deployment.simulator.orchestrator import run_simulation


DEFAULT_BASE_CONFIG = ROOT_DIR / "configs" / "default_simulation.json"
DEFAULT_SCENARIO_DIR = ROOT_DIR / "configs" / "scenarios"
DEFAULT_OUTPUT_FILE = ROOT_DIR / "simulator" / "outputs" / "exp1" / "aggregate" / "experiment1_architectural_advantage_M500_seed3.json"
DEFAULT_REQUEST_DATASET = ROOT_DIR / "dag_data_set" / "dataset" / "job_requests" / "requests_T30days.jsonl"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _percentile(sorted_values: list[float], ratio: float) -> float:
    if not sorted_values:
        return 0.0
    index = int(round((len(sorted_values) - 1) * ratio))
    index = min(max(index, 0), len(sorted_values) - 1)
    return sorted_values[index]


def _days_to_minutes(value: float) -> float:
    return value * 24.0 * 60.0


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    return mean(values), (0.0 if len(values) == 1 else pstdev(values))


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
    return any(
        not result.get("metadata", {}).get("deadline_missing", True)
        for result in results
    )


def _deadline_satisfied(result: dict[str, Any], deadlines_present: bool) -> bool:
    if not deadlines_present:
        return bool(result["success"])
    return bool(result.get("metadata", {}).get("deadline_satisfied", False))


def _group_metrics(per_request_results: list[dict[str, Any]], request_meta: dict[str, dict[str, str]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in per_request_results:
        dag_type = request_meta[result["request_id"]]["dag_type"]
        groups[dag_type].append(result)

    output: dict[str, dict[str, Any]] = {}
    for dag_type, results in sorted(groups.items()):
        deadlines_present = _has_deadline_metadata(results)
        raw_completed = [result for result in results if result["success"]]
        deadline_completed = [
            result
            for result in results
            if _deadline_satisfied(result, deadlines_present)
        ]
        makespans = sorted(_days_to_minutes(result["makespan_days"]) for result in raw_completed)
        cp_cmp = [_days_to_minutes(result["cp_delay_breakdown_days"]["cmp"]) for result in raw_completed]
        cp_net = [_days_to_minutes(result["cp_delay_breakdown_days"]["net"]) for result in raw_completed]
        cp_idle = [_days_to_minutes(result["cp_delay_breakdown_days"]["idle"]) for result in raw_completed]
        failures = Counter(result["failure_reason"] for result in results if result["failure_reason"])
        tardiness = [
            float(result.get("metadata", {}).get("normalized_tardiness", 0.0))
            for result in results
        ]
        output[dag_type] = {
            "request_count": len(results),
            "completed_count": len(deadline_completed),
            "failed_count": len(results) - len(deadline_completed),
            "task_completion_rate": len(deadline_completed) / max(len(results), 1),
            "deadline_satisfaction_ratio": len(deadline_completed) / max(len(results), 1),
            "raw_completed_count": len(raw_completed),
            "raw_task_completion_rate": len(raw_completed) / max(len(results), 1),
            "mean_normalized_tardiness": sum(tardiness) / len(tardiness) if tardiness else 0.0,
            "mean_makespan_minutes": sum(makespans) / len(makespans) if makespans else None,
            "p95_makespan_minutes": _percentile(makespans, 0.95) if makespans else None,
            "mean_cp_cmp_minutes": sum(cp_cmp) / len(cp_cmp) if cp_cmp else None,
            "mean_cp_net_minutes": sum(cp_net) / len(cp_net) if cp_net else None,
            "mean_cp_idle_minutes": sum(cp_idle) / len(cp_idle) if cp_idle else None,
            "failure_reasons": dict(failures),
        }
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiment 1: DAG-aware deployment advantage under fixed deployment.")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--scenario-dir", type=Path, default=DEFAULT_SCENARIO_DIR)
    parser.add_argument("--request-dataset", type=Path, default=DEFAULT_REQUEST_DATASET)
    parser.add_argument("--scenarios", nargs="*", default=["sparse_topology_stress"])
    parser.add_argument("--algorithms", nargs="*", default=["cpmv_dsd", "dependency_blind", "sfc_path_decomp"])
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--seeds", nargs="*", type=int, default=[20260427, 20260428, 20260429])
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--run-prefix", type=str, default="exp1_arch")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    base_config = load_config(args.base_config)
    request_meta = _load_request_metadata(args.request_dataset)
    results: list[dict[str, Any]] = []

    for scenario_name in args.scenarios:
        scenario_path = args.scenario_dir / f"{scenario_name}.json"
        scenario_override = json.loads(scenario_path.read_text(encoding="utf-8"))
        for algorithm_name in args.algorithms:
            seed_runs: list[dict[str, Any]] = []
            for seed in args.seeds:
                config = _deep_merge(base_config, scenario_override)
                config["environment"]["seed"] = seed
                config["algorithm"]["seed"] = seed
                config["algorithm"]["name"] = algorithm_name
                config["algorithm"]["sample_size"] = args.sample_size
                config["output"]["output_dir"] = str(ROOT_DIR / "simulator" / "outputs" / "exp1" / "runs")
                config["output"]["run_name"] = f"{args.run_prefix}_{scenario_name}_{algorithm_name}_seed{seed}"
                print(
                    f"[exp1] scenario={scenario_name} algorithm={algorithm_name} seed={seed} M={args.sample_size}",
                    flush=True,
                )
                result = run_simulation(config)
                output_dir = Path(result["manifest"]["output_dir"])
                summary = result["summary"]
                per_request_results = json.loads((output_dir / "per_request_results.json").read_text(encoding="utf-8"))
                seed_runs.append(
                    {
                        "seed": seed,
                        "overall": summary,
                        "by_dag_type": _group_metrics(per_request_results, request_meta),
                    }
                )
            overall_tcr = [run["overall"]["task_completion_rate"] for run in seed_runs]
            overall_mean = [run["overall"]["mean_makespan_minutes"] for run in seed_runs]
            overall_p95 = [run["overall"]["p95_makespan_minutes"] for run in seed_runs]
            overall_cmp = [run["overall"]["mean_cp_cmp_minutes"] for run in seed_runs]
            overall_net = [run["overall"]["mean_cp_net_minutes"] for run in seed_runs]
            overall_idle = [run["overall"]["mean_cp_idle_minutes"] for run in seed_runs]
            overall_failed = [run["overall"]["failed_count"] for run in seed_runs]
            overall_dsr = [run["overall"].get("deadline_satisfaction_ratio", run["overall"]["task_completion_rate"]) for run in seed_runs]
            overall_raw_tcr = [run["overall"].get("raw_task_completion_rate", run["overall"]["task_completion_rate"]) for run in seed_runs]
            overall_tardiness = [run["overall"].get("mean_normalized_tardiness", 0.0) for run in seed_runs]
            dag_groups = sorted({group for run in seed_runs for group in run["by_dag_type"]})
            by_dag_type_aggregate: dict[str, dict[str, Any]] = {}
            for dag_type in dag_groups:
                rows = [run["by_dag_type"][dag_type] for run in seed_runs if dag_type in run["by_dag_type"]]
                tcr_values = [row["task_completion_rate"] for row in rows]
                mean_values = [row["mean_makespan_minutes"] for row in rows if row["mean_makespan_minutes"] is not None]
                p95_values = [row["p95_makespan_minutes"] for row in rows if row["p95_makespan_minutes"] is not None]
                cmp_values = [row["mean_cp_cmp_minutes"] for row in rows if row["mean_cp_cmp_minutes"] is not None]
                net_values = [row["mean_cp_net_minutes"] for row in rows if row["mean_cp_net_minutes"] is not None]
                idle_values = [row["mean_cp_idle_minutes"] for row in rows if row["mean_cp_idle_minutes"] is not None]
                failed_values = [row["failed_count"] for row in rows]
                dsr_values = [row["deadline_satisfaction_ratio"] for row in rows]
                raw_tcr_values = [row["raw_task_completion_rate"] for row in rows]
                tardiness_values = [row["mean_normalized_tardiness"] for row in rows]
                tcr_mean, tcr_std = _mean_std(tcr_values)
                dsr_mean, dsr_std = _mean_std(dsr_values)
                raw_tcr_mean, raw_tcr_std = _mean_std(raw_tcr_values)
                tardiness_mean, tardiness_std = _mean_std(tardiness_values)
                mean_makespan_mean, mean_makespan_std = _mean_std(mean_values)
                by_dag_type_aggregate[dag_type] = {
                    "task_completion_rate_mean": tcr_mean,
                    "task_completion_rate_std": tcr_std,
                    "deadline_satisfaction_ratio_mean": dsr_mean,
                    "deadline_satisfaction_ratio_std": dsr_std,
                    "raw_task_completion_rate_mean": raw_tcr_mean,
                    "raw_task_completion_rate_std": raw_tcr_std,
                    "mean_normalized_tardiness_mean": tardiness_mean,
                    "mean_normalized_tardiness_std": tardiness_std,
                    "mean_makespan_minutes_mean": mean_makespan_mean if mean_values else None,
                    "mean_makespan_minutes_std": mean_makespan_std if mean_values else None,
                    "p95_makespan_minutes_mean": (sum(p95_values) / len(p95_values) if p95_values else None),
                    "mean_cp_cmp_minutes_mean": (sum(cmp_values) / len(cmp_values) if cmp_values else None),
                    "mean_cp_net_minutes_mean": (sum(net_values) / len(net_values) if net_values else None),
                    "mean_cp_idle_minutes_mean": (sum(idle_values) / len(idle_values) if idle_values else None),
                    "failed_count_mean": sum(failed_values) / len(failed_values),
                }
            results.append(
                {
                    "scenario_name": scenario_name,
                    "scenario_path": str(scenario_path),
                    "algorithm_name": algorithm_name,
                    "sample_size": args.sample_size,
                    "seeds": args.seeds,
                    "raw_seed_runs": seed_runs,
                    "overall": {
                        "task_completion_rate_mean": sum(overall_tcr) / len(overall_tcr),
                        "task_completion_rate_std": 0.0 if len(overall_tcr) == 1 else pstdev(overall_tcr),
                        "deadline_satisfaction_ratio_mean": sum(overall_dsr) / len(overall_dsr),
                        "deadline_satisfaction_ratio_std": 0.0 if len(overall_dsr) == 1 else pstdev(overall_dsr),
                        "raw_task_completion_rate_mean": sum(overall_raw_tcr) / len(overall_raw_tcr),
                        "raw_task_completion_rate_std": 0.0 if len(overall_raw_tcr) == 1 else pstdev(overall_raw_tcr),
                        "mean_normalized_tardiness_mean": sum(overall_tardiness) / len(overall_tardiness),
                        "mean_normalized_tardiness_std": 0.0 if len(overall_tardiness) == 1 else pstdev(overall_tardiness),
                        "mean_makespan_minutes_mean": sum(overall_mean) / len(overall_mean),
                        "mean_makespan_minutes_std": 0.0 if len(overall_mean) == 1 else pstdev(overall_mean),
                        "p95_makespan_minutes_mean": sum(overall_p95) / len(overall_p95),
                        "mean_cp_cmp_minutes_mean": sum(overall_cmp) / len(overall_cmp),
                        "mean_cp_net_minutes_mean": sum(overall_net) / len(overall_net),
                        "mean_cp_idle_minutes_mean": sum(overall_idle) / len(overall_idle),
                        "failed_count_mean": sum(overall_failed) / len(overall_failed),
                    },
                    "by_dag_type": by_dag_type_aggregate,
                }
            )

    output = {
        "experiment": "experiment1_dag_advantage_fixed",
        "sample_size": args.sample_size,
        "scenarios": args.scenarios,
        "algorithms": args.algorithms,
        "results": results,
    }
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = args.output_file.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "scenario",
                "algorithm",
                "group",
                "seed_count",
                "tcr_mean",
                "tcr_std",
                "deadline_satisfaction_ratio_mean",
                "deadline_satisfaction_ratio_std",
                "raw_task_completion_rate_mean",
                "raw_task_completion_rate_std",
                "mean_normalized_tardiness_mean",
                "mean_normalized_tardiness_std",
                "mean_makespan_minutes_mean",
                "mean_makespan_minutes_std",
                "p95_makespan_minutes_mean",
                "mean_cp_cmp_minutes_mean",
                "mean_cp_net_minutes_mean",
                "mean_cp_idle_minutes_mean",
                "failed_count_mean",
            ]
        )
        for record in results:
            overall = record["overall"]
            writer.writerow(
                [
                    record["scenario_name"],
                    record["algorithm_name"],
                    "overall",
                    len(record["seeds"]),
                    overall["task_completion_rate_mean"],
                    overall["task_completion_rate_std"],
                    overall["deadline_satisfaction_ratio_mean"],
                    overall["deadline_satisfaction_ratio_std"],
                    overall["raw_task_completion_rate_mean"],
                    overall["raw_task_completion_rate_std"],
                    overall["mean_normalized_tardiness_mean"],
                    overall["mean_normalized_tardiness_std"],
                    overall["mean_makespan_minutes_mean"],
                    overall["mean_makespan_minutes_std"],
                    overall["p95_makespan_minutes_mean"],
                    overall["mean_cp_cmp_minutes_mean"],
                    overall["mean_cp_net_minutes_mean"],
                    overall["mean_cp_idle_minutes_mean"],
                    overall["failed_count_mean"],
                ]
            )
            for dag_type, metrics in record["by_dag_type"].items():
                writer.writerow(
                    [
                        record["scenario_name"],
                        record["algorithm_name"],
                        dag_type,
                        len(record["seeds"]),
                        metrics["task_completion_rate_mean"],
                        metrics["task_completion_rate_std"],
                        metrics["deadline_satisfaction_ratio_mean"],
                        metrics["deadline_satisfaction_ratio_std"],
                        metrics["raw_task_completion_rate_mean"],
                        metrics["raw_task_completion_rate_std"],
                        metrics["mean_normalized_tardiness_mean"],
                        metrics["mean_normalized_tardiness_std"],
                        metrics["mean_makespan_minutes_mean"],
                        metrics["mean_makespan_minutes_std"],
                        metrics["p95_makespan_minutes_mean"],
                        metrics["mean_cp_cmp_minutes_mean"],
                        metrics["mean_cp_net_minutes_mean"],
                        metrics["mean_cp_idle_minutes_mean"],
                        metrics["failed_count_mean"],
                    ]
                )

    print(json.dumps({"output_json": str(args.output_file), "output_csv": str(csv_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
