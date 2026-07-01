from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
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
DEFAULT_OUTPUT_DIR = ROOT_DIR / "simulator" / "outputs" / "exp5" / "aggregate"

SCENARIO_LABELS = {
    "sparse_topology_stress": "sparse",
    "normal_nominal": "normal",
    "dense_reference": "dense",
}

DAG_LABELS = {
    "chain_like": "chain",
    "wide_shallow": "wide-shallow",
    "general": "mixed",
}

VARIANTS: dict[str, dict[str, Any]] = {
    "full": {
        "label": "Full MV-DSD",
        "use_comm_graph_constraint": True,
        "use_topology_features": True,
        "use_structured_value": True,
        "use_structured_deployment": True,
        "use_role_matching_deployment": True,
        "use_stratified_sampling": True,
        "stratified_weight_clip_ratio": 8.0,
    },
    "no_topology": {
        "label": "No-Topology",
        "use_comm_graph_constraint": False,
        "use_topology_features": False,
        "use_structured_value": True,
        "use_structured_deployment": True,
        "use_role_matching_deployment": True,
        "use_stratified_sampling": True,
        "stratified_weight_clip_ratio": 8.0,
    },
    "no_structured_value": {
        "label": "No-Structured-Value",
        "use_comm_graph_constraint": True,
        "use_topology_features": True,
        "use_structured_value": False,
        "use_structured_deployment": False,
        "use_role_matching_deployment": False,
        "use_stratified_sampling": True,
        "stratified_weight_clip_ratio": 8.0,
    },
    "no_stratified_sampling": {
        "label": "No-Stratified-Sampling",
        "use_comm_graph_constraint": True,
        "use_topology_features": True,
        "use_structured_value": True,
        "use_structured_deployment": True,
        "use_role_matching_deployment": True,
        "use_stratified_sampling": False,
        "stratified_weight_clip_ratio": 0.0,
    },
    "legacy_shapley_rank": {
        "label": "Legacy-Shapley-Rank",
        "use_comm_graph_constraint": False,
        "use_topology_features": False,
        "use_structured_value": False,
        "use_structured_deployment": False,
        "use_role_matching_deployment": False,
        "use_stratified_sampling": False,
        "stratified_weight_clip_ratio": 0.0,
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiment 5: MV-DSD internal ablation study.")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--scenario-dir", type=Path, default=DEFAULT_SCENARIO_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--studies", nargs="*", default=["density", "dagtype"], choices=["density", "dagtype"])
    parser.add_argument("--density-scenarios", nargs="*", default=["sparse_topology_stress", "normal_nominal", "dense_reference"])
    parser.add_argument("--dag-scenario", type=str, default="normal_nominal")
    parser.add_argument("--dag-types", nargs="*", default=["chain_like", "wide_shallow", "general"])
    parser.add_argument("--variants", nargs="*", default=["full", "no_topology", "no_structured_value", "no_stratified_sampling"])
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--seeds", nargs="*", type=int, default=[20260427, 20260428, 20260429])
    parser.add_argument("--max-windows", type=int, default=6)
    parser.add_argument("--max-requests-per-window", type=int, default=24)
    parser.add_argument("--output-prefix", type=str, default="experiment5_ablation_M200_seed3")
    parser.add_argument("--quiet", action="store_true")
    return parser


def _mean_std_ci(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    avg = mean(values)
    std = 0.0 if len(values) == 1 else pstdev(values)
    ci95 = 0.0 if len(values) == 1 else 1.96 * std / math.sqrt(len(values))
    return avg, std, ci95


def _prepare_config(
    base_config: dict[str, Any],
    scenario_override: dict[str, Any],
    seed: int,
    sample_size: int,
    variant_name: str,
    max_windows: int | None,
    max_requests_per_window: int | None,
) -> dict[str, Any]:
    variant = VARIANTS[variant_name]
    config = deep_merge(base_config, scenario_override)
    if max_windows is not None:
        config["simulation"]["max_windows"] = max_windows
    if max_requests_per_window is not None:
        config["simulation"]["max_requests_per_window"] = max_requests_per_window
    config["seed"] = seed
    config["environment"]["seed"] = seed
    config["algorithm"]["seed"] = seed
    config["algorithm"]["name"] = "cpmv_dsd"
    config["algorithm"]["sample_size"] = sample_size
    config["algorithm"]["use_cvar"] = False
    config["algorithm"]["cvar_lambda"] = 0.0
    config["scheduler"]["name"] = "heft"
    for key, value in variant.items():
        if key != "label":
            config["algorithm"][key] = value
    return config


def _filter_windows_by_dag_type(windows: list[DeploymentWindow], dag_type: str) -> list[DeploymentWindow]:
    filtered = []
    for window in windows:
        requests = [request for request in window.requests if request.dag_type == dag_type]
        if not requests:
            continue
        filtered.append(
            DeploymentWindow(
                window_id=window.window_id,
                start_time_days=window.start_time_days,
                end_time_days=window.end_time_days,
                requests=requests,
            )
        )
    return filtered


def _run_one(
    config: dict[str, Any],
    scenario_name: str,
    variant_name: str,
    eval_dag_type: str | None = None,
) -> dict[str, Any]:
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
    plan_meta = plan.metadata
    return {
        "scenario_name": scenario_name,
        "variant_name": variant_name,
        "variant_label": VARIANTS[variant_name]["label"],
        "eval_dag_type": eval_dag_type,
        "summary": summary,
        "planning_runtime_seconds": planning_runtime,
        "oracle_calls": plan_meta.get("oracle_calls", 0),
        "pruned_calls": plan_meta.get("pruned_calls", 0),
        "pruning_rate": plan_meta.get("pruning_rate", 0.0),
        "component_weights": plan_meta.get("component_weights", {}),
        "use_structured_value": plan_meta.get("use_structured_value"),
        "use_role_matching_deployment": plan_meta.get("use_role_matching_deployment"),
    }


def _summarize_group(seed_runs: list[dict[str, Any]]) -> dict[str, Any]:
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
        "mean_makespan_minutes": [run["summary"]["mean_makespan_minutes"] for run in seed_runs],
        "p95_makespan_minutes": [run["summary"]["p95_makespan_minutes"] for run in seed_runs],
        "p99_makespan_minutes": [run["summary"]["p99_makespan_minutes"] for run in seed_runs],
        "mean_cp_cmp_minutes": [run["summary"]["mean_cp_cmp_minutes"] for run in seed_runs],
        "mean_cp_net_minutes": [run["summary"]["mean_cp_net_minutes"] for run in seed_runs],
        "mean_cp_idle_minutes": [run["summary"]["mean_cp_idle_minutes"] for run in seed_runs],
        "failed_count": [float(run["summary"]["failed_count"]) for run in seed_runs],
        "planning_runtime_seconds": [float(run["planning_runtime_seconds"]) for run in seed_runs],
        "oracle_calls": [float(run["oracle_calls"]) for run in seed_runs],
        "pruned_calls": [float(run["pruned_calls"]) for run in seed_runs],
        "pruning_rate": [float(run["pruning_rate"]) for run in seed_runs],
    }
    output: dict[str, Any] = {}
    for field, values in fields.items():
        avg, std, ci95 = _mean_std_ci(values)
        output[f"{field}_mean"] = avg
        output[f"{field}_std"] = std
        output[f"{field}_ci95"] = ci95
    return output


def _attach_relative_metrics(rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if row["variant"] == "full":
            grouped[(row["study"], row["case"])] = row
    for row in rows:
        full = grouped.get((row["study"], row["case"]))
        if not full:
            continue
        full_mean = max(float(full["mean_makespan_minutes_mean"]), 1e-9)
        row["normalized_makespan_vs_full"] = float(row["mean_makespan_minutes_mean"]) / full_mean
        row["gap_vs_full"] = (float(row["mean_makespan_minutes_mean"]) - full_mean) / full_mean
        if row["variant"] == "full":
            row["main_degraded_component"] = "none"
            continue
        component_deltas = {
            "cmp": float(row["mean_cp_cmp_minutes_mean"]) - float(full["mean_cp_cmp_minutes_mean"]),
            "net": float(row["mean_cp_net_minutes_mean"]) - float(full["mean_cp_net_minutes_mean"]),
            "idle": float(row["mean_cp_idle_minutes_mean"]) - float(full["mean_cp_idle_minutes_mean"]),
        }
        row["main_degraded_component"] = max(component_deltas, key=component_deltas.get)


def _write_study(output_file: Path, study: str, records: list[dict[str, Any]], rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    fieldnames = [
        "study",
        "case",
        "case_label",
        "variant",
        "variant_label",
        "seed_count",
        "sample_size",
        "task_completion_rate_mean",
        "task_completion_rate_std",
        "task_completion_rate_ci95",
        "deadline_satisfaction_ratio_mean",
        "deadline_satisfaction_ratio_std",
        "deadline_satisfaction_ratio_ci95",
        "raw_task_completion_rate_mean",
        "raw_task_completion_rate_std",
        "raw_task_completion_rate_ci95",
        "mean_makespan_minutes_mean",
        "mean_makespan_minutes_std",
        "mean_makespan_minutes_ci95",
        "p95_makespan_minutes_mean",
        "p95_makespan_minutes_std",
        "p95_makespan_minutes_ci95",
        "p99_makespan_minutes_mean",
        "mean_cp_cmp_minutes_mean",
        "mean_cp_net_minutes_mean",
        "mean_cp_idle_minutes_mean",
        "normalized_makespan_vs_full",
        "gap_vs_full",
        "main_degraded_component",
        "planning_runtime_seconds_mean",
        "oracle_calls_mean",
        "pruned_calls_mean",
        "pruning_rate_mean",
        "failed_count_mean",
    ]
    payload = {
        "experiment": "experiment5_internal_ablation",
        "study": study,
        "sample_size": args.sample_size,
        "seeds": args.seeds,
        "max_windows": args.max_windows,
        "max_requests_per_window": args.max_requests_per_window,
        "variants": {name: VARIANTS[name] for name in args.variants},
        "results": records,
    }
    write_json_csv(output_file, payload, rows, fieldnames)


def run_density_study(args: argparse.Namespace, base_config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = []
    rows = []
    for scenario_name in args.density_scenarios:
        scenario_override = json.loads((args.scenario_dir / f"{scenario_name}.json").read_text(encoding="utf-8"))
        for variant_name in args.variants:
            seed_runs = []
            for seed in args.seeds:
                if not args.quiet:
                    print(f"[exp5-density] scenario={scenario_name} variant={variant_name} seed={seed}", flush=True)
                config = _prepare_config(
                    base_config,
                    scenario_override,
                    seed,
                    args.sample_size,
                    variant_name,
                    args.max_windows,
                    args.max_requests_per_window,
                )
                seed_runs.append(_run_one(config, scenario_name, variant_name))
            aggregate = _summarize_group(seed_runs)
            record = {
                "study": "density",
                "case": scenario_name,
                "case_label": SCENARIO_LABELS.get(scenario_name, scenario_name),
                "variant": variant_name,
                "variant_label": VARIANTS[variant_name]["label"],
                "seed_runs": seed_runs,
                "aggregate": aggregate,
            }
            records.append(record)
            rows.append(
                {
                    "study": "density",
                    "case": scenario_name,
                    "case_label": SCENARIO_LABELS.get(scenario_name, scenario_name),
                    "variant": variant_name,
                    "variant_label": VARIANTS[variant_name]["label"],
                    "seed_count": len(seed_runs),
                    "sample_size": args.sample_size,
                    **aggregate,
                }
            )
    _attach_relative_metrics(rows)
    return records, rows


def run_dagtype_study(args: argparse.Namespace, base_config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = []
    rows = []
    scenario_override = json.loads((args.scenario_dir / f"{args.dag_scenario}.json").read_text(encoding="utf-8"))
    for dag_type in args.dag_types:
        for variant_name in args.variants:
            seed_runs = []
            for seed in args.seeds:
                if not args.quiet:
                    print(f"[exp5-dagtype] dag_type={dag_type} variant={variant_name} seed={seed}", flush=True)
                config = _prepare_config(
                    base_config,
                    scenario_override,
                    seed,
                    args.sample_size,
                    variant_name,
                    args.max_windows,
                    args.max_requests_per_window,
                )
                seed_runs.append(_run_one(config, args.dag_scenario, variant_name, eval_dag_type=dag_type))
            aggregate = _summarize_group(seed_runs)
            record = {
                "study": "dagtype",
                "case": dag_type,
                "case_label": DAG_LABELS.get(dag_type, dag_type),
                "variant": variant_name,
                "variant_label": VARIANTS[variant_name]["label"],
                "seed_runs": seed_runs,
                "aggregate": aggregate,
            }
            records.append(record)
            rows.append(
                {
                    "study": "dagtype",
                    "case": dag_type,
                    "case_label": DAG_LABELS.get(dag_type, dag_type),
                    "variant": variant_name,
                    "variant_label": VARIANTS[variant_name]["label"],
                    "seed_count": len(seed_runs),
                    "sample_size": args.sample_size,
                    **aggregate,
                }
            )
    _attach_relative_metrics(rows)
    return records, rows


def main() -> None:
    args = build_parser().parse_args()
    unknown_variants = [variant for variant in args.variants if variant not in VARIANTS]
    if unknown_variants:
        raise ValueError(f"Unknown variants: {unknown_variants}")

    base_config = load_config(args.base_config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suite_records: dict[str, list[dict[str, Any]]] = {}
    suite_rows: dict[str, list[dict[str, Any]]] = {}

    if "density" in args.studies:
        records, rows = run_density_study(args, base_config)
        suite_records["density"] = records
        suite_rows["density"] = rows
        _write_study(args.output_dir / f"{args.output_prefix}_density.json", "density", records, rows, args)

    if "dagtype" in args.studies:
        records, rows = run_dagtype_study(args, base_config)
        suite_records["dagtype"] = records
        suite_rows["dagtype"] = rows
        _write_study(args.output_dir / f"{args.output_prefix}_dagtype.json", "dagtype", records, rows, args)

    combined_rows = [row for rows in suite_rows.values() for row in rows]
    combined_payload = {
        "experiment": "experiment5_internal_ablation",
        "sample_size": args.sample_size,
        "seeds": args.seeds,
        "max_windows": args.max_windows,
        "max_requests_per_window": args.max_requests_per_window,
        "studies": args.studies,
        "variants": {name: VARIANTS[name] for name in args.variants},
        "results": suite_records,
    }
    combined_file = args.output_dir / f"{args.output_prefix}_suite.json"
    combined_file.write_text(json.dumps(combined_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "suite_json": str(combined_file), "row_count": len(combined_rows)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
