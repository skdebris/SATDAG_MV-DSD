from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = ROOT_DIR.parent
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from experiment34_common import (
    deep_merge,
    deploy_fixed_plan,
    evaluate_fixed_plan,
    load_windows_and_catalog,
    summarize_execution_results,
    write_json_csv,
)
from run_experiment3_cvar_sweep import (
    FLUCTUATION_CONFIG,
    _aggregate_seed_runs,
    _sample_envs_by_window,
    _sample_envs_for_planning,
)
from sat_dag_service_deployment.simulator.config import load_config
from sat_dag_service_deployment.simulator.env import build_satellite_environment


DEFAULT_BASE_CONFIG = ROOT_DIR / "configs" / "default_simulation.json"
DEFAULT_SCENARIO_DIR = ROOT_DIR / "configs" / "scenarios"
DEFAULT_OUTPUT_FILE = ROOT_DIR / "simulator" / "outputs" / "exp5" / "aggregate" / "experiment5_cvar_sweep_M500_seed3.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiment 5: CVaR-aware MV-DSD risk extension.")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--scenario-dir", type=Path, default=DEFAULT_SCENARIO_DIR)
    parser.add_argument("--scenario", type=str, default="sparse_fluctuation_stress")
    parser.add_argument("--lambda-values", nargs="*", type=float, default=[0.0, 0.1, 0.3, 0.5, 0.7, 1.0])
    parser.add_argument("--fluctuation-level", type=str, default="high", choices=sorted(FLUCTUATION_CONFIG))
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=0.90)
    parser.add_argument("--risk-amplifier", type=float, default=4.0)
    parser.add_argument("--seeds", nargs="*", type=int, default=[20260427, 20260428, 20260429])
    parser.add_argument("--train-topology-samples", type=int, default=4)
    parser.add_argument("--test-topology-samples", type=int, default=8)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--max-requests-per-window", type=int, default=None)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--quiet", action="store_true")
    return parser


def _prepare_base_config(args: argparse.Namespace) -> dict[str, Any]:
    base_config = load_config(args.base_config)
    scenario_override = json.loads((args.scenario_dir / f"{args.scenario}.json").read_text(encoding="utf-8"))
    config = deep_merge(base_config, scenario_override)
    config["algorithm"]["name"] = "cpmv_dsd"
    config["algorithm"]["sample_size"] = args.sample_size
    config["scheduler"]["name"] = "heft"
    if args.max_windows is not None:
        config["simulation"]["max_windows"] = args.max_windows
    if args.max_requests_per_window is not None:
        config["simulation"]["max_requests_per_window"] = args.max_requests_per_window
    return config


def main() -> None:
    args = build_parser().parse_args()
    base_config = _prepare_base_config(args)
    level = args.fluctuation_level
    records: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    baseline_by_seed: dict[int, dict[str, float]] = {}
    total_deployments = len(args.lambda_values) * len(args.seeds)
    deployment_index = 0

    def progress(message: str) -> None:
        if not args.quiet:
            print(message, flush=True)

    progress(
        "[exp5] start "
        f"scenario={args.scenario} level={level} M={args.sample_size} "
        f"lambdas={args.lambda_values} seeds={args.seeds} "
        f"train_samples={args.train_topology_samples} test_samples={args.test_topology_samples}"
    )

    lambda_records: list[dict[str, Any]] = []
    for lam in args.lambda_values:
        progress(f"[exp5] lambda={lam} begin")
        seed_runs: list[dict[str, Any]] = []
        for seed in args.seeds:
            deployment_index += 1
            progress(f"[exp5] deploy {deployment_index}/{total_deployments}: lambda={lam} seed={seed}")
            config = deep_merge(base_config, {})
            config["seed"] = seed
            config["environment"]["seed"] = seed
            config["algorithm"]["seed"] = seed
            config["algorithm"]["use_cvar"] = lam > 0.0
            config["algorithm"]["cvar_lambda"] = lam
            config["algorithm"]["cvar_alpha"] = args.alpha
            config["algorithm"]["cvar_risk_amplifier"] = args.risk_amplifier
            windows, service_catalog = load_windows_and_catalog(config)
            env_seed = int(config["environment"]["seed"])
            planning_env = build_satellite_environment(
                config["environment"],
                window_id=int(config["simulation"].get("planning_reference_window_id", 0)),
                seed=env_seed,
            )
            if lam > 0.0:
                config["algorithm"]["cvar_topology_samples"] = _sample_envs_for_planning(
                    planning_env,
                    config["environment"],
                    level=level,
                    seed=seed + 3001,
                    count=args.train_topology_samples,
                )
            plan, planning_runtime = deploy_fixed_plan(config, windows, service_catalog, planning_env)
            progress(
                "[exp5] deployed "
                f"lambda={lam} seed={seed} runtime={planning_runtime:.3f}s "
                f"oracle={plan.metadata.get('oracle_calls', 0)} pruned={plan.metadata.get('pruned_calls', 0)}"
            )

            all_results = []
            for sample_idx in range(args.test_topology_samples):
                progress(f"[exp5] evaluate lambda={lam} seed={seed} topology_sample={sample_idx + 1}/{args.test_topology_samples}")
                envs = _sample_envs_by_window(config, windows, level, seed + 9001, sample_idx)
                summary, execution_results = evaluate_fixed_plan(config, windows, plan, envs, alpha=args.alpha)
                all_results.extend(execution_results)
                progress(
                    "[exp5] sample "
                    f"lambda={lam} seed={seed} sample={sample_idx + 1}/{args.test_topology_samples} "
                    f"DSR={summary.get('deadline_satisfaction_ratio', summary['task_completion_rate']):.4f} "
                    f"mean={summary['mean_makespan_minutes']:.3f} "
                    f"p95={summary['p95_makespan_minutes']:.3f} "
                    f"cvar={summary['cvar_makespan_minutes']:.3f}"
                )
            combined_summary = summarize_execution_results(
                all_results,
                [plan],
                alpha=args.alpha,
                metadata={
                    "algorithm": plan.algorithm_name,
                    "scheduler": config["scheduler"]["name"],
                    "topology_samples": args.test_topology_samples,
                },
            )
            if lam == 0.0:
                baseline_by_seed[seed] = {
                    "mean_makespan_minutes": combined_summary["mean_makespan_minutes"],
                    "p95_makespan_minutes": combined_summary["p95_makespan_minutes"],
                    "cvar_makespan_minutes": combined_summary["cvar_makespan_minutes"],
                }
            seed_runs.append(
                {
                    "seed": seed,
                    "summary": combined_summary,
                    "planning_runtime_seconds": planning_runtime,
                    "deployment_metadata": plan.metadata,
                }
            )
            progress(
                "[exp5] seed aggregate "
                f"lambda={lam} seed={seed} "
                f"DSR={combined_summary.get('deadline_satisfaction_ratio', combined_summary['task_completion_rate']):.4f} "
                f"mean={combined_summary['mean_makespan_minutes']:.3f} "
                f"p95={combined_summary['p95_makespan_minutes']:.3f} "
                f"cvar={combined_summary['cvar_makespan_minutes']:.3f}"
            )

        lambda_records.append(
            {
                "fluctuation_level": level,
                "lambda_cvar": lam,
                "alpha": args.alpha,
                "seed_runs": seed_runs,
            }
        )

    for record in lambda_records:
        aggregate = _aggregate_seed_runs(record["seed_runs"], baseline_by_seed)
        record["aggregate"] = aggregate
        records.append(record)
        progress(
            "[exp5] lambda aggregate "
            f"lambda={record['lambda_cvar']} "
            f"DSR={aggregate['deadline_satisfaction_ratio_mean']:.4f} "
            f"mean={aggregate['mean_makespan_minutes_mean']:.3f} "
            f"p95={aggregate['p95_makespan_minutes_mean']:.3f} "
            f"cvar={aggregate['cvar_makespan_minutes_mean']:.3f} "
            f"tail_reduction={100.0 * aggregate['tail_reduction_ratio_mean']:.2f}%"
        )
        csv_rows.append(
            {
                "fluctuation_level": record["fluctuation_level"],
                "lambda_cvar": record["lambda_cvar"],
                "alpha": record["alpha"],
                **aggregate,
            }
        )

    payload = {
        "experiment": "experiment5_cvar_sweep",
        "scenario": args.scenario,
        "sample_size": args.sample_size,
        "lambda_values": args.lambda_values,
        "fluctuation_level": level,
        "train_topology_samples": args.train_topology_samples,
        "test_topology_samples": args.test_topology_samples,
        "risk_amplifier": args.risk_amplifier,
        "fluctuation_config": FLUCTUATION_CONFIG[level],
        "results": records,
    }
    fieldnames = [
        "fluctuation_level",
        "lambda_cvar",
        "alpha",
        "task_completion_rate_mean",
        "deadline_satisfaction_ratio_mean",
        "raw_task_completion_rate_mean",
        "mean_makespan_minutes_mean",
        "p95_makespan_minutes_mean",
        "p99_makespan_minutes_mean",
        "cvar_makespan_minutes_mean",
        "deadline_miss_rate_mean",
        "tail_reduction_ratio_mean",
        "cvar_reduction_ratio_mean",
        "mean_penalty_ratio_mean",
        "planning_runtime_seconds_mean",
        "oracle_calls_mean",
        "pruned_calls_mean",
        "pruning_rate_mean",
    ]
    write_json_csv(args.output_file, payload, csv_rows, fieldnames)
    progress(f"[exp5] finished output_json={args.output_file} output_csv={args.output_file.with_suffix('.csv')}")
    print(json.dumps({"output_json": str(args.output_file), "output_csv": str(args.output_file.with_suffix(".csv"))}, indent=2))


if __name__ == "__main__":
    main()
