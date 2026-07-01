from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
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
DEFAULT_OUTPUT_FILE = ROOT_DIR / "simulator" / "outputs" / "experiment2_scheduler_ablation.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiment 2-B: scheduler ablation for deployment algorithms.")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--scenario-dir", type=Path, default=DEFAULT_SCENARIO_DIR)
    parser.add_argument("--scenarios", nargs="*", default=["normal_nominal"])
    parser.add_argument("--algorithms", nargs="*", default=["cpmv_dsd", "jsdts_aos_sat", "ondoc_sat"])
    parser.add_argument("--schedulers", nargs="*", default=["heft", "peft", "ccp_mdag"])
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--seeds", nargs="*", type=int, default=[20260427, 20260428, 20260429])
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--max-requests-per-window", type=int, default=None)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--run-prefix", type=str, default="exp2_sched")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    base_config = load_config(args.base_config)
    records: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []

    for scenario_name in args.scenarios:
        scenario_override = json.loads((args.scenario_dir / f"{scenario_name}.json").read_text(encoding="utf-8"))
        for scheduler_name in args.schedulers:
            for algorithm_name in args.algorithms:
                seed_runs = []
                for seed in args.seeds:
                    config = deep_merge(base_config, scenario_override)
                    config["environment"]["seed"] = seed
                    config["algorithm"]["seed"] = seed
                    config["algorithm"]["name"] = algorithm_name
                    config["algorithm"]["sample_size"] = args.sample_size
                    config["algorithm"]["scheduler_name"] = scheduler_name
                    config["scheduler"]["name"] = scheduler_name
                    if args.max_windows is not None:
                        config["simulation"]["max_windows"] = args.max_windows
                    if args.max_requests_per_window is not None:
                        config["simulation"]["max_requests_per_window"] = args.max_requests_per_window
                    config["output"]["run_name"] = f"{args.run_prefix}_{scenario_name}_{algorithm_name}_{scheduler_name}_seed{seed}"
                    result = run_simulation(config)
                    output_dir = Path(result["manifest"]["output_dir"])
                    per_request = json.loads((output_dir / "per_request_results.json").read_text(encoding="utf-8"))
                    seed_runs.append(
                        {
                            "seed": seed,
                            "summary": result["summary"],
                            "cpu_utilization_proxy": cpu_utilization_proxy(per_request),
                            "output_dir": str(output_dir),
                        }
                    )
                aggregate = summarize_seed_runs(seed_runs)
                record = {
                    "scenario_name": scenario_name,
                    "algorithm_name": algorithm_name,
                    "scheduler_name": scheduler_name,
                    "sample_size": args.sample_size,
                    "seeds": args.seeds,
                    "seed_runs": seed_runs,
                    "aggregate": aggregate,
                }
                records.append(record)
                csv_rows.append(
                    {
                        "experiment": "experiment2_scheduler_ablation",
                        "scenario": scenario_name,
                        "algorithm": algorithm_name,
                        "scheduler": scheduler_name,
                        "seed_count": len(seed_runs),
                        **aggregate,
                    }
                )

    payload = {
        "experiment": "experiment2_scheduler_ablation",
        "evaluation_mode": "cross_scheduler_robustness",
        "scenarios": args.scenarios,
        "algorithms": args.algorithms,
        "schedulers": args.schedulers,
        "sample_size": args.sample_size,
        "results": records,
    }
    write_outputs(args.output_file, payload, csv_rows)
    print(json.dumps({"output_json": str(args.output_file), "output_csv": str(args.output_file.with_suffix(".csv"))}, indent=2))


if __name__ == "__main__":
    main()
