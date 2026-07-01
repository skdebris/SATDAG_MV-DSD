from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = ROOT_DIR.parent
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from experiment34_common import (
    contribution_variance,
    deep_merge,
    deploy_fixed_plan,
    evaluate_fixed_plan,
    load_windows_and_catalog,
    mean_std,
    topk_jaccard,
    write_json_csv,
)
from sat_dag_service_deployment.simulator.config import load_config
from sat_dag_service_deployment.simulator.env import build_satellite_environment


DEFAULT_BASE_CONFIG = ROOT_DIR / "configs" / "default_simulation.json"
DEFAULT_SCENARIO_DIR = ROOT_DIR / "configs" / "scenarios"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "simulator" / "outputs" / "exp4" / "aggregate"

VARIANTS = {
    "full": {
        "use_stratified_sampling": True,
        "use_comm_graph_constraint": True,
        "use_topology_features": True,
    },
    "uniform_sampling": {
        "use_stratified_sampling": False,
        "use_comm_graph_constraint": True,
        "use_topology_features": True,
    },
    "no_pruning": {
        "use_stratified_sampling": True,
        "use_comm_graph_constraint": False,
        "use_topology_features": True,
    },
    "naive_shapley": {
        "use_stratified_sampling": False,
        "use_comm_graph_constraint": False,
        "use_topology_features": False,
    },
}

N_CONFIGS = {
    40: {"num_planes": 4, "satellites_per_plane": 10},
    80: {"num_planes": 8, "satellites_per_plane": 10},
    120: {"num_planes": 10, "satellites_per_plane": 12},
    160: {"num_planes": 10, "satellites_per_plane": 16},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiment 4: MV-DSD scalability and estimation efficiency.")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--scenario-dir", type=Path, default=DEFAULT_SCENARIO_DIR)
    parser.add_argument("--scenario", type=str, default="normal_nominal")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-sizes", nargs="*", type=int, default=[50, 100, 200, 300, 500])
    parser.add_argument("--selected-m", type=int, default=500)
    parser.add_argument("--n-values", nargs="*", type=int, default=[40, 80, 120, 160])
    parser.add_argument("--kappa-values", nargs="*", type=float, default=[0.10, 0.50, 0.70, 0.90, 0.99, 1.01])
    parser.add_argument("--seeds", nargs="*", type=int, default=[20260427, 20260428, 20260429])
    parser.add_argument("--max-windows", type=int, default=6)
    parser.add_argument("--max-requests-per-window", type=int, default=24)
    parser.add_argument("--quiet", action="store_true")
    return parser


def _prepare_config(
    base_config: dict[str, Any],
    scenario_override: dict[str, Any],
    seed: int,
    sample_size: int,
    variant: str,
    args: argparse.Namespace,
    n_satellites: int | None = None,
    kappa: float | None = None,
) -> dict[str, Any]:
    config = deep_merge(base_config, scenario_override)
    if args.max_windows is not None:
        config["simulation"]["max_windows"] = args.max_windows
    if args.max_requests_per_window is not None:
        config["simulation"]["max_requests_per_window"] = args.max_requests_per_window
    if n_satellites is not None:
        config["environment"].update(N_CONFIGS[int(n_satellites)])
    if kappa is not None:
        config["environment"]["aggregate_kappa"] = float(kappa)
    config["seed"] = seed
    config["environment"]["seed"] = seed
    config["algorithm"]["seed"] = seed
    config["algorithm"]["name"] = "cpmv_dsd"
    config["algorithm"]["sample_size"] = sample_size
    config["algorithm"]["use_cvar"] = False
    config["scheduler"]["name"] = "heft"
    for key, value in VARIANTS[variant].items():
        config["algorithm"][key] = value
    return config


def _graph_stats(env: Any) -> dict[str, float]:
    node_count = len(env.satellites)
    degree_sum = sum(len(env.aggregate_graph.get(sat_id, {})) for sat_id in env.satellites)
    directed_edges = degree_sum
    max_directed = max(node_count * (node_count - 1), 1)
    return {
        "avg_degree_g_hat": degree_sum / max(node_count, 1),
        "edge_ratio_g_hat": directed_edges / max_directed,
        "aggregate_threshold": float(getattr(env, "aggregate_threshold", 0.0)),
    }


def _run_once(config: dict[str, Any]) -> dict[str, Any]:
    windows, service_catalog = load_windows_and_catalog(config)
    env_seed = int(config["environment"]["seed"] if config["environment"].get("seed") is not None else config["seed"])
    planning_env = build_satellite_environment(
        config["environment"],
        window_id=int(config["simulation"].get("planning_reference_window_id", 0)),
        seed=env_seed,
    )
    plan, planning_runtime = deploy_fixed_plan(config, windows, service_catalog, planning_env)
    envs = {
        window.window_id: build_satellite_environment(config["environment"], window.window_id, env_seed)
        for window in windows
    }
    summary, _ = evaluate_fixed_plan(config, windows, plan, envs, alpha=0.9)
    scalar_priority = plan.metadata.get("scalar_priority_raw") or plan.metadata.get("scalar_priority") or {}
    ranking = sorted(scalar_priority, key=lambda sat_id: scalar_priority[sat_id], reverse=True)
    return {
        "summary": summary,
        "planning_runtime_seconds": planning_runtime,
        "metadata": plan.metadata,
        "scalar_priority": scalar_priority,
        "ranking": ranking,
        "graph_stats": _graph_stats(planning_env),
    }


def _aggregate_group(records: list[dict[str, Any]], k_ratio: float = 0.2) -> dict[str, Any]:
    runtime_values = [record["planning_runtime_seconds"] for record in records]
    oracle_values = [float(record["metadata"].get("oracle_calls", 0)) for record in records]
    pruned_values = [float(record["metadata"].get("pruned_calls", 0)) for record in records]
    pruning_values = [float(record["metadata"].get("pruning_rate", 0.0)) for record in records]
    mean_values = [record["summary"]["mean_makespan_minutes"] for record in records]
    p95_values = [record["summary"]["p95_makespan_minutes"] for record in records]
    dsr_values = [
        record["summary"].get("deadline_satisfaction_ratio", record["summary"]["task_completion_rate"])
        for record in records
    ]
    raw_tcr_values = [
        record["summary"].get("raw_task_completion_rate", record["summary"]["task_completion_rate"])
        for record in records
    ]
    tardiness_values = [record["summary"].get("mean_normalized_tardiness", 0.0) for record in records]
    avg_degree_values = [record["graph_stats"]["avg_degree_g_hat"] for record in records]
    edge_ratio_values = [record["graph_stats"]["edge_ratio_g_hat"] for record in records]
    aggregate_threshold_values = [record["graph_stats"]["aggregate_threshold"] for record in records]
    k = max(1, round(len(records[0]["ranking"]) * k_ratio)) if records and records[0]["ranking"] else 1
    runtime_mean, runtime_std = mean_std(runtime_values)
    return {
        "seed_count": len(records),
        "planning_runtime_seconds_mean": runtime_mean,
        "planning_runtime_seconds_std": runtime_std,
        "oracle_calls_mean": mean(oracle_values) if oracle_values else 0.0,
        "pruned_calls_mean": mean(pruned_values) if pruned_values else 0.0,
        "pruning_rate_mean": mean(pruning_values) if pruning_values else 0.0,
        "contribution_variance": contribution_variance([record["scalar_priority"] for record in records]),
        "topk_jaccard": topk_jaccard([record["ranking"] for record in records], k=k),
        "topk_k": k,
        "mean_makespan_minutes_mean": mean(mean_values) if mean_values else 0.0,
        "p95_makespan_minutes_mean": mean(p95_values) if p95_values else 0.0,
        "deadline_satisfaction_ratio_mean": mean(dsr_values) if dsr_values else 0.0,
        "raw_task_completion_rate_mean": mean(raw_tcr_values) if raw_tcr_values else 0.0,
        "mean_normalized_tardiness_mean": mean(tardiness_values) if tardiness_values else 0.0,
        "avg_degree_g_hat_mean": mean(avg_degree_values) if avg_degree_values else 0.0,
        "edge_ratio_g_hat_mean": mean(edge_ratio_values) if edge_ratio_values else 0.0,
        "aggregate_threshold_mean": mean(aggregate_threshold_values) if aggregate_threshold_values else 0.0,
    }


def _write_section(output_dir: Path, name: str, payload: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "section",
        "variant",
        "sample_size",
        "n_satellites",
        "aggregate_kappa",
        "seed_count",
        "planning_runtime_seconds_mean",
        "planning_runtime_seconds_std",
        "oracle_calls_mean",
        "pruned_calls_mean",
        "pruning_rate_mean",
        "contribution_variance",
        "topk_jaccard",
        "topk_k",
        "mean_makespan_minutes_mean",
        "p95_makespan_minutes_mean",
        "deadline_satisfaction_ratio_mean",
        "raw_task_completion_rate_mean",
        "mean_normalized_tardiness_mean",
        "avg_degree_g_hat_mean",
        "edge_ratio_g_hat_mean",
        "aggregate_threshold_mean",
    ]
    write_json_csv(output_dir / f"{name}.json", payload, rows, fieldnames)


def main() -> None:
    args = build_parser().parse_args()
    base_config = load_config(args.base_config)
    scenario_override = json.loads((args.scenario_dir / f"{args.scenario}.json").read_text(encoding="utf-8"))
    sections: dict[str, list[dict[str, Any]]] = {
        "exp4a_sampling_selection": [],
        "exp4b_scalability": [],
        "exp4c_kappa_sensitivity": [],
    }
    raw_records: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def progress(message: str) -> None:
        if not args.quiet:
            print(message, flush=True)

    for sample_size in args.sample_sizes:
        for variant in ["full", "uniform_sampling", "naive_shapley"]:
            records = []
            for seed in args.seeds:
                progress(f"[exp4-M] variant={variant} seed={seed} M={sample_size}")
                config = _prepare_config(base_config, scenario_override, seed, sample_size, variant, args)
                result = _run_once(config)
                records.append(result)
                raw_records["m_sweep"].append({"sample_size": sample_size, "variant": variant, "seed": seed, **result})
            sections["exp4a_sampling_selection"].append(
                {
                    "section": "exp4a_sampling_selection",
                    "variant": variant,
                    "sample_size": sample_size,
                    "n_satellites": 80,
                    "aggregate_kappa": scenario_override.get("environment", {}).get("aggregate_kappa", 0.25),
                    **_aggregate_group(records),
                }
            )

    for n_value in args.n_values:
        for variant in ["full", "uniform_sampling", "no_pruning"]:
            records = []
            for seed in args.seeds:
                progress(f"[exp4-N] variant={variant} seed={seed} N={n_value} M={args.selected_m}")
                config = _prepare_config(base_config, scenario_override, seed, args.selected_m, variant, args, n_satellites=n_value)
                result = _run_once(config)
                records.append(result)
                raw_records["n_sweep"].append({"n_satellites": n_value, "variant": variant, "seed": seed, **result})
            sections["exp4b_scalability"].append(
                {
                    "section": "exp4b_scalability",
                    "variant": variant,
                    "sample_size": args.selected_m,
                    "n_satellites": n_value,
                    "aggregate_kappa": scenario_override.get("environment", {}).get("aggregate_kappa", 0.25),
                    **_aggregate_group(records),
                }
            )

    for kappa in args.kappa_values:
        records = []
        for seed in args.seeds:
            progress(f"[exp4-kappa] seed={seed} kappa={kappa:.2f} M={args.selected_m}")
            config = _prepare_config(base_config, scenario_override, seed, args.selected_m, "full", args, kappa=kappa)
            result = _run_once(config)
            records.append(result)
            raw_records["kappa_sweep"].append({"aggregate_kappa": kappa, "variant": "full", "seed": seed, **result})
        sections["exp4c_kappa_sensitivity"].append(
            {
                "section": "exp4c_kappa_sensitivity",
                "variant": "full",
                "sample_size": args.selected_m,
                "n_satellites": 80,
                "aggregate_kappa": kappa,
                **_aggregate_group(records),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in sections.items():
        payload = {
            "experiment": name,
            "scenario": args.scenario,
            "selected_m": args.selected_m,
            "seeds": args.seeds,
            "max_windows": args.max_windows,
            "max_requests_per_window": args.max_requests_per_window,
            "results": rows,
        }
        _write_section(args.output_dir, name, payload, rows)

    combined_payload = {
        "experiment": "experiment4_efficiency_suite",
        "scenario": args.scenario,
        "selected_m": args.selected_m,
        "sample_sizes": args.sample_sizes,
        "n_values": args.n_values,
        "kappa_values": args.kappa_values,
        "seeds": args.seeds,
        "max_windows": args.max_windows,
        "max_requests_per_window": args.max_requests_per_window,
        "sections": sections,
    }
    (args.output_dir / "experiment4_efficiency_suite.json").write_text(
        json.dumps(combined_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(args.output_dir), "section_files": [f"{name}.csv" for name in sections]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
