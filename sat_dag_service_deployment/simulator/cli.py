from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import DEFAULT_CONFIG, load_config
from .data import ensure_materialized_requests
from .orchestrator import run_simulation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MV-DSD satellite DAG simulation CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    materialize = subparsers.add_parser("materialize-requests", help="Build full DAG job instances from arrivals and templates")
    materialize.add_argument("--config", type=str, default=None)

    run = subparsers.add_parser("run", help="Run an end-to-end simulation")
    run.add_argument("--config", type=str, default=None)
    run.add_argument("--run-name", type=str, default=None)
    run.add_argument("--algorithm", type=str, default=None)
    run.add_argument("--scheduler", type=str, default=None)
    run.add_argument("--density-mode", type=str, default=None)
    run.add_argument("--perturbation-mode", type=str, default=None)
    run.add_argument("--max-windows", type=int, default=None)
    run.add_argument("--max-requests-per-window", type=int, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config) if getattr(args, "config", None) else load_config()

    if args.command == "materialize-requests":
        output_path, summary_path = ensure_materialized_requests(
            arrival_trace_path=config["data"]["arrival_trace_path"],
            template_dir=config["data"]["template_dir"],
            output_path=config["data"]["job_requests_path"],
        )
        print(json.dumps({"request_dataset_path": str(output_path), "summary_path": str(summary_path)}, ensure_ascii=False, indent=2))
        return

    if args.run_name:
        config["output"]["run_name"] = args.run_name
    if args.algorithm:
        config["algorithm"]["name"] = args.algorithm
    if args.scheduler:
        config["scheduler"]["name"] = args.scheduler
    if args.density_mode:
        config["environment"]["density_mode"] = args.density_mode
    if args.perturbation_mode:
        config["environment"]["perturbation_mode"] = args.perturbation_mode
    if args.max_windows is not None:
        config["simulation"]["max_windows"] = args.max_windows
    if args.max_requests_per_window is not None:
        config["simulation"]["max_requests_per_window"] = args.max_requests_per_window

    result = run_simulation(config)
    print(json.dumps(result["manifest"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
