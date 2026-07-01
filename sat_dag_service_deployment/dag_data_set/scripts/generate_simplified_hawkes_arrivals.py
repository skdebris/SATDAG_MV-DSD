from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict, deque
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import median
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = BASE_DIR / "dataset"
ARRIVAL_DIR = DATASET_DIR / "arrival_traces"
TEMPLATES_DIR = DATASET_DIR / "dag_templates"

BALANCED_EVENTS_PATH = ARRIVAL_DIR / "hawkes_fitting_input_balanced_events.json"
SIMPLIFIED_PARAMS_PATH = ARRIVAL_DIR / "simplified_hawkes_params.json"
SIMULATED_TRACE_PATH = ARRIVAL_DIR / "simulated_arrivals_balanced_T30days.json"
SIMULATED_SUMMARY_PATH = ARRIVAL_DIR / "simulated_arrivals_balanced_T30days_summary.json"

RNG_SEED = 20260427
SIMULATION_HORIZON_DAYS = 30.0


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def poisson_sample(rng: random.Random, lam: float) -> int:
    if lam <= 0:
        return 0
    if lam < 30.0:
        threshold = math.exp(-lam)
        product = 1.0
        count = 0
        while product > threshold:
            count += 1
            product *= rng.random()
        return count - 1

    std = math.sqrt(lam)
    return max(0, int(round(rng.gauss(lam, std))))


def load_template_pools() -> dict[str, list[str]]:
    pools: dict[str, list[str]] = defaultdict(list)
    for json_path in sorted(TEMPLATES_DIR.glob("*/*.json")):
        if json_path.name == "dag_instances_summary.json":
            continue
        parts = json_path.stem.split("_")
        if len(parts) < 2:
            continue
        subarchetype = parts[1]
        pools[subarchetype].append(json_path.stem)
    return pools


def estimate_global_hawkes_params(events: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    by_sub: dict[str, list[datetime]] = defaultdict(list)
    for event in events:
        by_sub[event["assigned_subarchetype"]].append(parse_timestamp(event["timestamp"]))

    params: dict[str, dict[str, float]] = {}
    for subarchetype, timestamps in sorted(by_sub.items()):
        timestamps.sort()
        gaps_days = [
            (timestamps[index] - timestamps[index - 1]).total_seconds() / 86400.0
            for index in range(1, len(timestamps))
            if timestamps[index] > timestamps[index - 1]
        ]

        if len(gaps_days) >= 2:
            mean_gap = sum(gaps_days) / len(gaps_days)
            variance = sum((gap - mean_gap) ** 2 for gap in gaps_days) / len(gaps_days)
            std_gap = math.sqrt(variance)
            cv = std_gap / mean_gap if mean_gap > 0 else 1.0
            branching_ratio = clamp((cv - 1.0) / (cv + 1.0), 0.05, 0.65)
            median_gap = median(gaps_days)
            beta_per_day = clamp(1.0 / max(median_gap, 1.0 / 24.0), 1.0 / 30.0, 1.0)
        else:
            mean_gap = 7.0
            cv = 1.0
            branching_ratio = 0.2
            beta_per_day = 1.0 / 7.0

        params[subarchetype] = {
            "branching_ratio_alpha": round(branching_ratio, 4),
            "beta_per_day": round(beta_per_day, 4),
            "mean_interarrival_days": round(mean_gap, 4),
            "cv_interarrival": round(cv, 4),
        }
    return params


def estimate_region_baselines(
    events: list[dict[str, Any]],
    global_params: dict[str, dict[str, float]],
) -> tuple[dict[str, dict[str, float]], float, datetime, datetime]:
    timestamps = [parse_timestamp(event["timestamp"]) for event in events]
    start_time = min(timestamps)
    end_time = max(timestamps)
    horizon_days = max((end_time - start_time).total_seconds() / 86400.0, 1.0)

    counts: dict[str, Counter] = defaultdict(Counter)
    for event in events:
        counts[event["region_id"]][event["assigned_subarchetype"]] += 1

    baseline_rates: dict[str, dict[str, float]] = defaultdict(dict)
    for region_id, region_counts in counts.items():
        for subarchetype, count in sorted(region_counts.items()):
            alpha = global_params[subarchetype]["branching_ratio_alpha"]
            empirical_rate = count / horizon_days
            mu = empirical_rate * (1.0 - alpha)
            baseline_rates[region_id][subarchetype] = round(mu, 6)
    return baseline_rates, horizon_days, start_time, end_time


def simulate_immigrants(rng: random.Random, mu_per_day: float, horizon_days: float) -> list[float]:
    if mu_per_day <= 0:
        return []
    events: list[float] = []
    time_cursor = 0.0
    while True:
        time_cursor += rng.expovariate(mu_per_day)
        if time_cursor > horizon_days:
            break
        events.append(time_cursor)
    return events


def simulate_process(
    rng: random.Random,
    region_id: str,
    subarchetype: str,
    mu_per_day: float,
    alpha: float,
    beta_per_day: float,
    horizon_days: float,
) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    immigrants = simulate_immigrants(rng, mu_per_day, horizon_days)
    queue: deque[tuple[float, str | None, int]] = deque()

    for immigrant_time in immigrants:
        queue.append((immigrant_time, None, 0))

    local_index = 0
    while queue:
        event_time, parent_id, generation = queue.popleft()
        local_index += 1
        event_id = f"{region_id}_{subarchetype}_{local_index:06d}"
        trace.append(
            {
                "sim_event_id": event_id,
                "region_id": region_id,
                "assigned_subarchetype": subarchetype,
                "simulation_time_days": round(event_time, 6),
                "generation": generation,
                "parent_event_id": parent_id,
                "event_source": "immigrant" if generation == 0 else "offspring",
            }
        )

        child_count = poisson_sample(rng, alpha)
        for _ in range(child_count):
            delay = rng.expovariate(beta_per_day)
            child_time = event_time + delay
            if child_time <= horizon_days:
                queue.append((child_time, event_id, generation + 1))

    trace.sort(key=lambda item: item["simulation_time_days"])
    return trace


def assign_templates(
    trace: list[dict[str, Any]],
    template_pools: dict[str, list[str]],
    simulation_start_time: datetime,
    rng: random.Random,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for index, record in enumerate(sorted(trace, key=lambda item: item["simulation_time_days"]), start=1):
        subarchetype = record["assigned_subarchetype"]
        template_id = rng.choice(template_pools[subarchetype])
        timestamp = simulation_start_time + timedelta(days=record["simulation_time_days"])
        enriched.append(
            {
                "task_request_id": f"req_{index:06d}",
                "event_id": record["sim_event_id"],
                "timestamp": timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z"),
                "simulation_time_days": record["simulation_time_days"],
                "region_id": record["region_id"],
                "subarchetype": subarchetype,
                "dag_archetype": "chain_like" if subarchetype.startswith("1") else ("wide_shallow" if subarchetype.startswith("2") else "general"),
                "dag_instance_id": template_id,
                "event_source": record["event_source"],
                "generation": record["generation"],
                "parent_event_id": record["parent_event_id"],
            }
        )
    return enriched


def build_summary(
    params: dict[str, dict[str, float]],
    baselines: dict[str, dict[str, float]],
    observed_horizon_days: float,
    simulation_start_time: datetime,
    arrivals: list[dict[str, Any]],
) -> dict[str, Any]:
    counts_by_sub = Counter(item["subarchetype"] for item in arrivals)
    counts_by_region = Counter(item["region_id"] for item in arrivals)
    return {
        "generator_seed": RNG_SEED,
        "simulation_horizon_days": SIMULATION_HORIZON_DAYS,
        "observed_horizon_days": round(observed_horizon_days, 4),
        "simulation_start_utc": simulation_start_time.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "global_hawkes_params": params,
        "active_region_processes": sum(len(value) for value in baselines.values()),
        "simulated_arrival_count": len(arrivals),
        "counts_by_subarchetype": dict(sorted(counts_by_sub.items())),
        "top_regions": [
            {"region_id": region_id, "count": count}
            for region_id, count in counts_by_region.most_common(20)
        ],
    }


def main() -> None:
    rng = random.Random(RNG_SEED)
    balanced_events = load_json(BALANCED_EVENTS_PATH)
    template_pools = load_template_pools()

    global_params = estimate_global_hawkes_params(balanced_events)
    baselines, observed_horizon_days, observed_start, observed_end = estimate_region_baselines(
        balanced_events,
        global_params,
    )
    simulation_start_time = (observed_end + timedelta(days=1)).astimezone(UTC).replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )

    process_trace: list[dict[str, Any]] = []
    for region_id, region_baselines in sorted(baselines.items()):
        for subarchetype, mu_per_day in sorted(region_baselines.items()):
            alpha = global_params[subarchetype]["branching_ratio_alpha"]
            beta_per_day = global_params[subarchetype]["beta_per_day"]
            process_trace.extend(
                simulate_process(
                    rng,
                    region_id,
                    subarchetype,
                    mu_per_day,
                    alpha,
                    beta_per_day,
                    SIMULATION_HORIZON_DAYS,
                )
            )

    arrivals = assign_templates(process_trace, template_pools, simulation_start_time, rng)

    param_payload = {
        "generator_seed": RNG_SEED,
        "simulation_horizon_days": SIMULATION_HORIZON_DAYS,
        "observed_window": {
            "start_utc": observed_start.astimezone(UTC).isoformat().replace("+00:00", "Z"),
            "end_utc": observed_end.astimezone(UTC).isoformat().replace("+00:00", "Z"),
            "horizon_days": round(observed_horizon_days, 4),
        },
        "global_hawkes_params": global_params,
        "region_baseline_mu_per_day": baselines,
    }
    dump_json(SIMPLIFIED_PARAMS_PATH, param_payload)
    dump_json(SIMULATED_TRACE_PATH, arrivals)
    dump_json(
        SIMULATED_SUMMARY_PATH,
        build_summary(global_params, baselines, observed_horizon_days, simulation_start_time, arrivals),
    )


if __name__ == "__main__":
    main()
