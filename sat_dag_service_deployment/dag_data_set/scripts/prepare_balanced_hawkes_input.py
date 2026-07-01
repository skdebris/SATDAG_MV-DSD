from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
ARRIVAL_DIR = BASE_DIR / "dataset" / "arrival_traces"
NORMALIZED_EVENTS_PATH = ARRIVAL_DIR / "eonet_last365_normalized_events.json"
MAPPING_CONFIG_PATH = ARRIVAL_DIR / "dag_category_mapping_config.json"
WEIGHTED_PROJECTION_PATH = ARRIVAL_DIR / "hawkes_weighted_projection_records.json"
BALANCED_EVENTS_JSON_PATH = ARRIVAL_DIR / "hawkes_fitting_input_balanced_events.json"
BALANCED_EVENTS_CSV_PATH = ARRIVAL_DIR / "hawkes_fitting_input_balanced_events.csv"
BALANCED_DAILY_CSV_PATH = ARRIVAL_DIR / "hawkes_fitting_input_balanced_region_daily.csv"
SUMMARY_PATH = ARRIVAL_DIR / "hawkes_balancing_summary.json"

RNG_SEED = 20260427


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def ordered_subarchetypes(config: dict[str, Any]) -> list[str]:
    return sorted(config["subarchetypes"].keys())


def build_projection_records(
    normalized_events: list[dict[str, Any]],
    mapping_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    projection_records: list[dict[str, Any]] = []
    candidate_pools: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for record in normalized_events:
        category = record["category"]
        if category not in mapping_config["categories"]:
            continue

        config_entry = mapping_config["categories"][category]
        primary_sub = config_entry["primary_subarchetype"]
        balanced_weights = config_entry["balanced_projection_weights"]

        for subarchetype, weight in balanced_weights.items():
            projection_record = {
                "source_record_id": record["record_id"],
                "event_id": record["event_id"],
                "timestamp": record["timestamp"],
                "region_id": record["region_id"],
                "region_row": record["region_row"],
                "region_col": record["region_col"],
                "latitude": record["latitude"],
                "longitude": record["longitude"],
                "category": category,
                "primary_subarchetype": primary_sub,
                "projected_subarchetype": subarchetype,
                "projected_archetype": mapping_config["subarchetypes"][subarchetype]["archetype"],
                "projection_weight": weight,
                "title": record["title"],
            }
            projection_records.append(projection_record)
            candidate_pools[subarchetype].append(projection_record)

    projection_records.sort(
        key=lambda item: (
            item["timestamp"],
            item["projected_subarchetype"],
            item["source_record_id"],
        )
    )
    return projection_records, candidate_pools


def target_counts(total_records: int, subarchetypes: list[str]) -> dict[str, int]:
    base = total_records // len(subarchetypes)
    remainder = total_records % len(subarchetypes)
    counts = {}
    for index, subarchetype in enumerate(subarchetypes):
        counts[subarchetype] = base + (1 if index < remainder else 0)
    return counts


def balanced_resample(
    projection_pools: dict[str, list[dict[str, Any]]],
    mapping_config: dict[str, Any],
    total_records: int,
) -> list[dict[str, Any]]:
    rng = random.Random(RNG_SEED)
    subarchetypes = ordered_subarchetypes(mapping_config)
    targets = target_counts(total_records, subarchetypes)
    balanced_records: list[dict[str, Any]] = []

    for subarchetype in subarchetypes:
        pool = projection_pools.get(subarchetype, [])
        if not pool:
            raise ValueError(f"No projection candidates available for subarchetype {subarchetype}")

        weights = [entry["projection_weight"] for entry in pool]
        sampled = rng.choices(pool, weights=weights, k=targets[subarchetype])
        for sample_index, record in enumerate(sampled, start=1):
            balanced_records.append(
                {
                    "balanced_record_id": f"{subarchetype}_{sample_index:05d}",
                    "source_record_id": record["source_record_id"],
                    "event_id": record["event_id"],
                    "category": record["category"],
                    "title": record["title"],
                    "timestamp": record["timestamp"],
                    "region_id": record["region_id"],
                    "region_row": record["region_row"],
                    "region_col": record["region_col"],
                    "latitude": record["latitude"],
                    "longitude": record["longitude"],
                    "primary_subarchetype": record["primary_subarchetype"],
                    "assigned_subarchetype": subarchetype,
                    "assigned_archetype": mapping_config["subarchetypes"][subarchetype]["archetype"],
                    "projection_weight": record["projection_weight"],
                }
            )

    balanced_records.sort(
        key=lambda item: (
            item["timestamp"],
            item["assigned_subarchetype"],
            item["balanced_record_id"],
        )
    )
    return balanced_records


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_daily_region_table(balanced_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str, str], int] = defaultdict(int)
    for record in balanced_records:
        date = record["timestamp"][:10]
        key = (date, record["region_id"], record["assigned_subarchetype"])
        counts[key] += 1

    rows = [
        {
            "date": date,
            "region_id": region_id,
            "assigned_subarchetype": subarchetype,
            "count": count,
        }
        for (date, region_id, subarchetype), count in sorted(counts.items())
    ]
    return rows


def build_summary(
    normalized_events: list[dict[str, Any]],
    projection_records: list[dict[str, Any]],
    balanced_records: list[dict[str, Any]],
    mapping_config: dict[str, Any],
) -> dict[str, Any]:
    weighted_support = defaultdict(float)
    for record in projection_records:
        weighted_support[record["projected_subarchetype"]] += float(record["projection_weight"])

    balanced_counts = Counter(record["assigned_subarchetype"] for record in balanced_records)
    category_counts = Counter(record["category"] for record in balanced_records)
    unique_sources_per_sub = defaultdict(set)
    for record in balanced_records:
        unique_sources_per_sub[record["assigned_subarchetype"]].add(record["source_record_id"])

    return {
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "seed": RNG_SEED,
        "raw_normalized_record_count": len(normalized_events),
        "projection_record_count": len(projection_records),
        "balanced_record_count": len(balanced_records),
        "weighted_support_by_subarchetype": {
            key: round(weighted_support[key], 3)
            for key in ordered_subarchetypes(mapping_config)
        },
        "balanced_counts_by_subarchetype": {
            key: balanced_counts[key]
            for key in ordered_subarchetypes(mapping_config)
        },
        "unique_source_records_by_subarchetype": {
            key: len(unique_sources_per_sub[key])
            for key in ordered_subarchetypes(mapping_config)
        },
        "balanced_counts_by_category": dict(sorted(category_counts.items())),
    }


def main() -> None:
    normalized_events = load_json(NORMALIZED_EVENTS_PATH)
    mapping_config = load_json(MAPPING_CONFIG_PATH)

    projection_records, candidate_pools = build_projection_records(normalized_events, mapping_config)
    dump_json(WEIGHTED_PROJECTION_PATH, projection_records)

    balanced_records = balanced_resample(candidate_pools, mapping_config, len(normalized_events))
    dump_json(BALANCED_EVENTS_JSON_PATH, balanced_records)

    write_csv(
        BALANCED_EVENTS_CSV_PATH,
        balanced_records,
        [
            "balanced_record_id",
            "source_record_id",
            "event_id",
            "category",
            "title",
            "timestamp",
            "region_id",
            "region_row",
            "region_col",
            "latitude",
            "longitude",
            "primary_subarchetype",
            "assigned_subarchetype",
            "assigned_archetype",
            "projection_weight",
        ],
    )

    daily_rows = build_daily_region_table(balanced_records)
    write_csv(
        BALANCED_DAILY_CSV_PATH,
        daily_rows,
        ["date", "region_id", "assigned_subarchetype", "count"],
    )

    summary = build_summary(normalized_events, projection_records, balanced_records, mapping_config)
    dump_json(SUMMARY_PATH, summary)


if __name__ == "__main__":
    main()
