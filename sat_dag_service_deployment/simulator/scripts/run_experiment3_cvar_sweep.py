from __future__ import annotations

import argparse
import json
import sys
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
    deep_merge,
    deploy_fixed_plan,
    evaluate_fixed_plan,
    load_windows_and_catalog,
    mean_std,
    perturb_environment,
    summarize_execution_results,
    write_json_csv,
)
from sat_dag_service_deployment.simulator.config import load_config
from sat_dag_service_deployment.simulator.env import build_satellite_environment


DEFAULT_BASE_CONFIG = ROOT_DIR / "configs" / "default_simulation.json"
DEFAULT_SCENARIO_DIR = ROOT_DIR / "configs" / "scenarios"
DEFAULT_OUTPUT_FILE = ROOT_DIR / "simulator" / "outputs" / "exp3" / "aggregate" / "experiment3_cvar_sweep_M200_seed3.json"

FLUCTUATION_CONFIG = {
    "low": {"p_fail": 0.02, "bw_sigma": 0.05},
    "medium": {"p_fail": 0.08, "bw_sigma": 0.15},
    "high": {"p_fail": 0.22, "bw_sigma": 0.40},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiment 3: CVaR robustness under topology fluctuations.")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--scenario-dir", type=Path, default=DEFAULT_SCENARIO_DIR)
    parser.add_argument("--scenario", type=str, default="normal_nominal")
    parser.add_argument("--lambda-values", nargs="*", type=float, default=[0.0, 0.3, 0.5, 1.0])
    parser.add_argument("--fluctuation-levels", nargs="*", default=["low", "high"])
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.90)
    parser.add_argument("--risk-amplifier", type=float, default=4.0)
    parser.add_argument("--seeds", nargs="*", type=int, default=[20260427, 20260428, 20260429])
    parser.add_argument("--train-topology-samples", type=int, default=4)
    parser.add_argument("--test-topology-samples", type=int, default=8)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--max-requests-per-window", type=int, default=None)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--quiet", action="store_true", help="Suppress per-run progress messages.")
    return parser


def _sample_envs_for_planning(base_env, env_config: dict, level: str, seed: int, count: int):
    noise = FLUCTUATION_CONFIG[level]
    return [
        perturb_environment(
            base_env,
            p_fail=noise["p_fail"],
            bw_sigma=noise["bw_sigma"],
            seed=seed + idx * 104729,
            aggregate_kappa=float(env_config["aggregate_kappa"]),
        )
        for idx in range(count)
    ]


def _sample_envs_by_window(config: dict, windows, level: str, seed: int, sample_idx: int):
    noise = FLUCTUATION_CONFIG[level]
    env_seed = int(config["environment"]["seed"] if config["environment"].get("seed") is not None else config["seed"])
    envs = {}
    for window in windows:
        base_env = build_satellite_environment(config["environment"], window_id=window.window_id, seed=env_seed)
        envs[window.window_id] = perturb_environment(
            base_env,
            p_fail=noise["p_fail"],
            bw_sigma=noise["bw_sigma"],
            seed=seed + sample_idx * 65537 + window.window_id * 9176,
            aggregate_kappa=float(config["environment"]["aggregate_kappa"]),
        )
    return envs


def _aggregate_seed_runs(seed_runs: list[dict[str, Any]], baseline_by_seed: dict[int, dict[str, float]]) -> dict[str, Any]:
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
        "cvar_makespan_minutes": [run["summary"]["cvar_makespan_minutes"] for run in seed_runs],
        "deadline_miss_rate": [run["summary"]["deadline_miss_rate"] for run in seed_runs],
        "planning_runtime_seconds": [run["planning_runtime_seconds"] for run in seed_runs],
        "oracle_calls": [float(run["deployment_metadata"].get("oracle_calls", 0)) for run in seed_runs],
        "pruned_calls": [float(run["deployment_metadata"].get("pruned_calls", 0)) for run in seed_runs],
        "pruning_rate": [float(run["deployment_metadata"].get("pruning_rate", 0.0)) for run in seed_runs],
    }
    output: dict[str, Any] = {}
    for field, values in fields.items():
        avg, std = mean_std(values)
        output[f"{field}_mean"] = avg
        output[f"{field}_std"] = std

    tail_reductions = []
    mean_penalties = []
    cvar_reductions = []
    for run in seed_runs:
        seed = run["seed"]
        baseline = baseline_by_seed.get(seed)
        if not baseline:
            continue
        summary = run["summary"]
        tail_reductions.append((baseline["p95_makespan_minutes"] - summary["p95_makespan_minutes"]) / max(baseline["p95_makespan_minutes"], 1e-9))
        cvar_reductions.append((baseline["cvar_makespan_minutes"] - summary["cvar_makespan_minutes"]) / max(baseline["cvar_makespan_minutes"], 1e-9))
        mean_penalties.append((summary["mean_makespan_minutes"] - baseline["mean_makespan_minutes"]) / max(baseline["mean_makespan_minutes"], 1e-9))
    output["tail_reduction_ratio_mean"] = mean(tail_reductions) if tail_reductions else 0.0
    output["cvar_reduction_ratio_mean"] = mean(cvar_reductions) if cvar_reductions else 0.0
    output["mean_penalty_ratio_mean"] = mean(mean_penalties) if mean_penalties else 0.0
    return output


def main() -> None:
    args = build_parser().parse_args()
    base_config = load_config(args.base_config)
    scenario_override = json.loads((args.scenario_dir / f"{args.scenario}.json").read_text(encoding="utf-8"))
    base_config = deep_merge(base_config, scenario_override)
    base_config["algorithm"]["name"] = "cpmv_dsd"
    base_config["algorithm"]["sample_size"] = args.sample_size
    base_config["scheduler"]["name"] = "heft"
    if args.max_windows is not None:
        base_config["simulation"]["max_windows"] = args.max_windows
    if args.max_requests_per_window is not None:
        base_config["simulation"]["max_requests_per_window"] = args.max_requests_per_window

    records: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    total_deployments = len(args.fluctuation_levels) * len(args.lambda_values) * len(args.seeds)
    deployment_index = 0

    def progress(message: str) -> None:
        if not args.quiet:
            print(message, flush=True)

    progress(
        "[exp3] start "
        f"scenario={args.scenario} M={args.sample_size} "
        f"levels={args.fluctuation_levels} lambdas={args.lambda_values} "
        f"seeds={args.seeds} train_topology_samples={args.train_topology_samples} "
        f"test_topology_samples={args.test_topology_samples}"
    )

    for level in args.fluctuation_levels:
        progress(f"[exp3] fluctuation level={level} noise={FLUCTUATION_CONFIG[level]}")
        baseline_by_seed: dict[int, dict[str, float]] = {}
        level_records: list[dict[str, Any]] = []
        for lam in args.lambda_values:
            progress(f"[exp3] lambda={lam} begin")
            seed_runs: list[dict[str, Any]] = []
            for seed in args.seeds:
                deployment_index += 1
                progress(f"[exp3] deploy {deployment_index}/{total_deployments}: level={level} lambda={lam} seed={seed}")
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
                    "[exp3] deployed "
                    f"level={level} lambda={lam} seed={seed} "
                    f"planning_runtime={planning_runtime:.3f}s "
                    f"oracle={plan.metadata.get('oracle_calls', 0)} pruned={plan.metadata.get('pruned_calls', 0)}"
                )

                sample_summaries = []
                all_results = []
                for sample_idx in range(args.test_topology_samples):
                    progress(
                        "[exp3] evaluate "
                        f"level={level} lambda={lam} seed={seed} "
                        f"topology_sample={sample_idx + 1}/{args.test_topology_samples}"
                    )
                    envs = _sample_envs_by_window(config, windows, level, seed + 9001, sample_idx)
                    summary, execution_results = evaluate_fixed_plan(config, windows, plan, envs, alpha=args.alpha)
                    sample_summaries.append(summary)
                    all_results.extend(execution_results)
                    progress(
                        "[exp3] sample summary "
                        f"level={level} lambda={lam} seed={seed} "
                        f"topology_sample={sample_idx + 1}/{args.test_topology_samples} "
                        f"TCR={summary['task_completion_rate']:.4f} "
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
                    "[exp3] seed aggregate "
                    f"level={level} lambda={lam} seed={seed} "
                    f"TCR={combined_summary['task_completion_rate']:.4f} "
                    f"mean={combined_summary['mean_makespan_minutes']:.3f} "
                    f"p95={combined_summary['p95_makespan_minutes']:.3f} "
                    f"cvar={combined_summary['cvar_makespan_minutes']:.3f}"
                )

            record = {
                "fluctuation_level": level,
                "lambda_cvar": lam,
                "alpha": args.alpha,
                "seed_runs": seed_runs,
            }
            level_records.append(record)

        for record in level_records:
            aggregate = _aggregate_seed_runs(record["seed_runs"], baseline_by_seed)
            record["aggregate"] = aggregate
            records.append(record)
            progress(
                "[exp3] lambda aggregate "
                f"level={record['fluctuation_level']} lambda={record['lambda_cvar']} "
                f"TCR={aggregate['task_completion_rate_mean']:.4f} "
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
        "experiment": "experiment3_cvar_sweep",
        "scenario": args.scenario,
        "sample_size": args.sample_size,
        "lambda_values": args.lambda_values,
        "fluctuation_levels": args.fluctuation_levels,
        "train_topology_samples": args.train_topology_samples,
        "test_topology_samples": args.test_topology_samples,
        "risk_amplifier": args.risk_amplifier,
        "fluctuation_config": FLUCTUATION_CONFIG,
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
    progress(f"[exp3] finished output_json={args.output_file} output_csv={args.output_file.with_suffix('.csv')}")
    print(json.dumps({"output_json": str(args.output_file), "output_csv": str(args.output_file.with_suffix(".csv"))}, indent=2))


if __name__ == "__main__":
    main()
