from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = ROOT_DIR.parent
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from experiment34_common import deep_merge, deploy_fixed_plan, evaluate_fixed_plan, load_windows_and_catalog, write_json_csv
from sat_dag_service_deployment.simulator.config import load_config
from sat_dag_service_deployment.simulator.env import build_satellite_environment
from sat_dag_service_deployment.simulator.models import DeploymentWindow


DEFAULT_BASE_CONFIG = ROOT_DIR / "configs" / "default_simulation.json"
DEFAULT_SCENARIO_DIR = ROOT_DIR / "configs" / "scenarios"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "simulator" / "outputs" / "exp3" / "aggregate"

VARIANTS: dict[str, dict[str, Any]] = {
    "full": {
        "label": "Full",
        "use_comm_graph_constraint": True,
        "use_topology_features": True,
        "use_structured_value": True,
        "use_structured_deployment": True,
        "use_role_matching_deployment": True,
        "use_stratified_sampling": True,
    },
    "shapley": {
        "label": "Shapley",
        "use_comm_graph_constraint": False,
        "use_topology_features": False,
        "use_structured_value": True,
        "use_structured_deployment": True,
        "use_role_matching_deployment": True,
        "use_stratified_sampling": True,
    },
    "no_struct": {
        "label": "No-Struct",
        "use_comm_graph_constraint": True,
        "use_topology_features": True,
        "use_structured_value": False,
        "use_structured_deployment": False,
        "use_role_matching_deployment": False,
        "use_stratified_sampling": True,
    },
    "no_strat": {
        "label": "No-Strat",
        "use_comm_graph_constraint": True,
        "use_topology_features": True,
        "use_structured_value": True,
        "use_structured_deployment": True,
        "use_role_matching_deployment": True,
        "use_stratified_sampling": False,
    },
}

DAG_LABELS = {
    "chain_like": "Chain",
    "wide_shallow": "Wide",
    "general": "Mixed",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiment 3: MV-DSD mechanism ablation.")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--scenario-dir", type=Path, default=DEFAULT_SCENARIO_DIR)
    parser.add_argument("--scenario", type=str, default="sparse_topology_stress")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--sample-sizes", nargs="*", type=int, default=[50, 100, 200, 300, 500])
    parser.add_argument("--variants", nargs="*", default=["full", "shapley", "no_struct", "no_strat"])
    parser.add_argument("--dag-types", nargs="*", default=["chain_like", "wide_shallow", "general"])
    parser.add_argument("--seeds", nargs="*", type=int, default=[20260427, 20260428, 20260429])
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--max-requests-per-window", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser


def _mean_std_ci(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    avg = mean(values)
    std = 0.0 if len(values) == 1 else pstdev(values)
    ci95 = 0.0 if len(values) == 1 else 1.96 * std / math.sqrt(len(values))
    return avg, std, ci95


def _prepare_config(base_config: dict[str, Any], scenario_override: dict[str, Any], seed: int, sample_size: int, variant: str, args: argparse.Namespace) -> dict[str, Any]:
    config = deep_merge(base_config, scenario_override)
    if args.max_windows is not None:
        config["simulation"]["max_windows"] = args.max_windows
    if args.max_requests_per_window is not None:
        config["simulation"]["max_requests_per_window"] = args.max_requests_per_window
    config["seed"] = seed
    config["environment"]["seed"] = seed
    config["algorithm"]["seed"] = seed
    config["algorithm"]["name"] = "cpmv_dsd"
    config["algorithm"]["sample_size"] = sample_size
    config["algorithm"]["use_cvar"] = False
    config["scheduler"]["name"] = "heft"
    for key, value in VARIANTS[variant].items():
        if key != "label":
            config["algorithm"][key] = value
    return config


def _filter_windows_by_dag_type(windows: list[DeploymentWindow], dag_type: str) -> list[DeploymentWindow]:
    filtered = []
    for window in windows:
        requests = [request for request in window.requests if request.dag_type == dag_type]
        if requests:
            filtered.append(
                DeploymentWindow(
                    window_id=window.window_id,
                    start_time_days=window.start_time_days,
                    end_time_days=window.end_time_days,
                    requests=requests,
                )
            )
    return filtered


def _run_once(config: dict[str, Any], variant: str, eval_dag_type: str | None = None) -> dict[str, Any]:
    windows, service_catalog = load_windows_and_catalog(config)
    eval_windows = _filter_windows_by_dag_type(windows, eval_dag_type) if eval_dag_type else windows
    env_seed = int(config["environment"]["seed"] if config["environment"].get("seed") is not None else config["seed"])
    planning_env = build_satellite_environment(
        config["environment"],
        window_id=int(config["simulation"].get("planning_reference_window_id", 0)),
        seed=env_seed,
    )
    plan, planning_runtime = deploy_fixed_plan(config, windows, service_catalog, planning_env)
    envs = {
        window.window_id: build_satellite_environment(config["environment"], window.window_id, env_seed)
        for window in eval_windows
    }
    summary, _ = evaluate_fixed_plan(config, eval_windows, plan, envs, alpha=0.9)
    return {
        "variant": variant,
        "variant_label": VARIANTS[variant]["label"],
        "summary": summary,
        "planning_runtime_seconds": planning_runtime,
        "metadata": plan.metadata,
        "eval_dag_type": eval_dag_type,
    }


def _summarize(seed_runs: list[dict[str, Any]]) -> dict[str, Any]:
    fields = {
        "deadline_satisfaction_ratio": [run["summary"].get("deadline_satisfaction_ratio", run["summary"]["task_completion_rate"]) for run in seed_runs],
        "raw_task_completion_rate": [run["summary"].get("raw_task_completion_rate", run["summary"]["task_completion_rate"]) for run in seed_runs],
        "mean_normalized_tardiness": [run["summary"].get("mean_normalized_tardiness", 0.0) for run in seed_runs],
        "mean_makespan_minutes": [run["summary"]["mean_makespan_minutes"] for run in seed_runs],
        "p95_makespan_minutes": [run["summary"]["p95_makespan_minutes"] for run in seed_runs],
        "mean_cp_cmp_minutes": [run["summary"]["mean_cp_cmp_minutes"] for run in seed_runs],
        "mean_cp_net_minutes": [run["summary"]["mean_cp_net_minutes"] for run in seed_runs],
        "mean_cp_idle_minutes": [run["summary"]["mean_cp_idle_minutes"] for run in seed_runs],
        "planning_runtime_seconds": [float(run["planning_runtime_seconds"]) for run in seed_runs],
        "oracle_calls": [float(run["metadata"].get("oracle_calls", 0.0)) for run in seed_runs],
        "pruned_calls": [float(run["metadata"].get("pruned_calls", 0.0)) for run in seed_runs],
        "pruning_rate": [float(run["metadata"].get("pruning_rate", 0.0)) for run in seed_runs],
        "failed_count": [float(run["summary"]["failed_count"]) for run in seed_runs],
    }
    output: dict[str, Any] = {}
    for field, values in fields.items():
        avg, std, ci95 = _mean_std_ci(values)
        output[f"{field}_mean"] = avg
        output[f"{field}_std"] = std
        output[f"{field}_ci95"] = ci95
    return output


def _write(output_file: Path, payload: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "study",
        "variant",
        "variant_label",
        "dag_type",
        "dag_label",
        "sample_size",
        "seed_count",
        "deadline_satisfaction_ratio_mean",
        "deadline_satisfaction_ratio_std",
        "deadline_satisfaction_ratio_ci95",
        "raw_task_completion_rate_mean",
        "mean_normalized_tardiness_mean",
        "mean_normalized_tardiness_std",
        "mean_normalized_tardiness_ci95",
        "mean_makespan_minutes_mean",
        "p95_makespan_minutes_mean",
        "p95_makespan_minutes_std",
        "p95_makespan_minutes_ci95",
        "mean_cp_cmp_minutes_mean",
        "mean_cp_net_minutes_mean",
        "mean_cp_idle_minutes_mean",
        "planning_runtime_seconds_mean",
        "oracle_calls_mean",
        "pruned_calls_mean",
        "pruning_rate_mean",
        "failed_count_mean",
    ]
    write_json_csv(output_file, payload, rows, fieldnames)


def main() -> None:
    args = build_parser().parse_args()
    unknown = [variant for variant in args.variants if variant not in VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    base_config = load_config(args.base_config)
    scenario_override = json.loads((args.scenario_dir / f"{args.scenario}.json").read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def progress(message: str) -> None:
        if not args.quiet:
            print(message, flush=True)

    overall_rows = []
    overall_records = []
    for variant in args.variants:
        seed_runs = []
        for seed in args.seeds:
            progress(f"[exp3-overall] variant={variant} seed={seed} M={args.sample_size}")
            config = _prepare_config(base_config, scenario_override, seed, args.sample_size, variant, args)
            seed_runs.append(_run_once(config, variant))
        aggregate = _summarize(seed_runs)
        record = {"variant": variant, "variant_label": VARIANTS[variant]["label"], "seed_runs": seed_runs, "aggregate": aggregate}
        overall_records.append(record)
        overall_rows.append(
            {
                "study": "overall",
                "variant": variant,
                "variant_label": VARIANTS[variant]["label"],
                "dag_type": "",
                "dag_label": "",
                "sample_size": args.sample_size,
                "seed_count": len(seed_runs),
                **aggregate,
            }
        )
    _write(
        args.output_dir / f"experiment3_overall_ablation_M{args.sample_size}_seed{len(args.seeds)}.json",
        {"experiment": "experiment3_overall_ablation", "scenario": args.scenario, "sample_size": args.sample_size, "results": overall_records},
        overall_rows,
    )

    dag_rows = []
    dag_records = []
    for dag_type in args.dag_types:
        for variant in ["full", "shapley", "no_struct"]:
            seed_runs = []
            for seed in args.seeds:
                progress(f"[exp3-dag] dag={dag_type} variant={variant} seed={seed} M={args.sample_size}")
                config = _prepare_config(base_config, scenario_override, seed, args.sample_size, variant, args)
                seed_runs.append(_run_once(config, variant, eval_dag_type=dag_type))
            aggregate = _summarize(seed_runs)
            dag_records.append({"dag_type": dag_type, "variant": variant, "seed_runs": seed_runs, "aggregate": aggregate})
            dag_rows.append(
                {
                    "study": "dagtype",
                    "variant": variant,
                    "variant_label": VARIANTS[variant]["label"],
                    "dag_type": dag_type,
                    "dag_label": DAG_LABELS.get(dag_type, dag_type),
                    "sample_size": args.sample_size,
                    "seed_count": len(seed_runs),
                    **aggregate,
                }
            )
    _write(
        args.output_dir / f"experiment3_dagtype_ablation_M{args.sample_size}_seed{len(args.seeds)}.json",
        {"experiment": "experiment3_dagtype_ablation", "scenario": args.scenario, "sample_size": args.sample_size, "results": dag_records},
        dag_rows,
    )

    budget_rows = []
    budget_records = []
    for sample_size in args.sample_sizes:
        for variant in ["full", "no_strat"]:
            seed_runs = []
            for seed in args.seeds:
                progress(f"[exp3-budget] variant={variant} seed={seed} M={sample_size}")
                config = _prepare_config(base_config, scenario_override, seed, sample_size, variant, args)
                seed_runs.append(_run_once(config, variant))
            aggregate = _summarize(seed_runs)
            budget_records.append({"sample_size": sample_size, "variant": variant, "seed_runs": seed_runs, "aggregate": aggregate})
            budget_rows.append(
                {
                    "study": "budget",
                    "variant": variant,
                    "variant_label": VARIANTS[variant]["label"],
                    "dag_type": "",
                    "dag_label": "",
                    "sample_size": sample_size,
                    "seed_count": len(seed_runs),
                    **aggregate,
                }
            )
    _write(
        args.output_dir / f"experiment3_sampling_budget_seed{len(args.seeds)}.json",
        {"experiment": "experiment3_sampling_budget", "scenario": args.scenario, "sample_sizes": args.sample_sizes, "results": budget_records},
        budget_rows,
    )
    print(json.dumps({"output_dir": str(args.output_dir), "rows": len(overall_rows) + len(dag_rows) + len(budget_rows)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
