from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = ROOT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sat_dag_service_deployment.simulator.config import load_config
from sat_dag_service_deployment.simulator.orchestrator import run_simulation

DEFAULT_BASE_CONFIG = ROOT_DIR / "configs" / "default_simulation.json"
DEFAULT_SCENARIO_DIR = ROOT_DIR / "configs" / "scenarios"
DEFAULT_OUTPUT_FILE = ROOT_DIR / "simulator" / "outputs" / "scenario_suite_summary.json"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _scenario_files(scenario_dir: Path, names: list[str] | None) -> list[Path]:
    if names:
        files = [scenario_dir / f"{name}.json" for name in names]
    else:
        files = sorted(scenario_dir.glob("*.json"))
    missing = [path for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing scenario config(s): {', '.join(str(path) for path in missing)}")
    return files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a calibrated scenario suite for MV-DSD experiments.")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--scenario-dir", type=Path, default=DEFAULT_SCENARIO_DIR)
    parser.add_argument("--scenarios", nargs="*", default=None, help="Scenario config stems without .json")
    parser.add_argument(
        "--algorithms",
        nargs="*",
        default=["cpmv_dsd", "dependency_blind", "sfc_path_decomp"],
        help="Algorithms to run for each scenario",
    )
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--run-prefix", type=str, default="scene_suite")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    base_config = load_config(args.base_config)
    results: list[dict[str, Any]] = []

    for scenario_path in _scenario_files(args.scenario_dir, args.scenarios):
        scenario_name = scenario_path.stem
        scenario_override = json.loads(scenario_path.read_text(encoding="utf-8"))
        scenario_record: dict[str, Any] = {
            "scenario_name": scenario_name,
            "scenario_path": str(scenario_path),
            "override": scenario_override,
            "algorithms": {},
        }
        for algorithm_name in args.algorithms:
            config = _deep_merge(base_config, scenario_override)
            config["algorithm"]["name"] = algorithm_name
            config["output"]["run_name"] = f"{args.run_prefix}_{scenario_name}_{algorithm_name}"
            summary = run_simulation(config)["summary"]
            scenario_record["algorithms"][algorithm_name] = {
                "task_completion_rate": summary["task_completion_rate"],
                "mean_makespan_minutes": summary["mean_makespan_minutes"],
                "p95_makespan_minutes": summary["p95_makespan_minutes"],
                "mean_cp_cmp_minutes": summary["mean_cp_cmp_minutes"],
                "mean_cp_net_minutes": summary["mean_cp_net_minutes"],
                "mean_cp_idle_minutes": summary["mean_cp_idle_minutes"],
                "failed_count": summary["failed_count"],
                "cross_sat_traffic_mb": summary["cross_sat_traffic_mb"],
            }
        if "cpmv_dsd" in scenario_record["algorithms"] and "dependency_blind" in scenario_record["algorithms"]:
            cpmv = scenario_record["algorithms"]["cpmv_dsd"]
            blind = scenario_record["algorithms"]["dependency_blind"]
            scenario_record["delta_vs_dependency_blind"] = {
                "tcr_gain": cpmv["task_completion_rate"] - blind["task_completion_rate"],
                "mean_makespan_delta": cpmv["mean_makespan_minutes"] - blind["mean_makespan_minutes"],
                "net_delay_delta": cpmv["mean_cp_net_minutes"] - blind["mean_cp_net_minutes"],
                "failed_count_delta": cpmv["failed_count"] - blind["failed_count"],
            }
        results.append(scenario_record)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = args.output_file.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "scenario",
                "algorithm",
                "tcr",
                "mean_makespan_minutes",
                "p95_makespan_minutes",
                "mean_cp_cmp_minutes",
                "mean_cp_net_minutes",
                "mean_cp_idle_minutes",
                "failed_count",
                "cross_sat_traffic_mb",
            ]
        )
        for scenario_record in results:
            for algorithm_name, summary in scenario_record["algorithms"].items():
                writer.writerow(
                    [
                        scenario_record["scenario_name"],
                        algorithm_name,
                        summary["task_completion_rate"],
                        summary["mean_makespan_minutes"],
                        summary["p95_makespan_minutes"],
                        summary["mean_cp_cmp_minutes"],
                        summary["mean_cp_net_minutes"],
                        summary["mean_cp_idle_minutes"],
                        summary["failed_count"],
                        summary["cross_sat_traffic_mb"],
                    ]
                )

    print(json.dumps({"output_json": str(args.output_file), "output_csv": str(csv_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
