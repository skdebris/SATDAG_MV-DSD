from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen


BASE_DIR = Path(__file__).resolve().parents[1]
ARRIVAL_DIR = BASE_DIR / "dataset" / "arrival_traces"
MAPPING_CONFIG_PATH = ARRIVAL_DIR / "dag_category_mapping_config.json"
RAW_EVENTS_PATH = ARRIVAL_DIR / "eonet_raw_last365_events.json"
RAW_CATEGORIES_PATH = ARRIVAL_DIR / "eonet_categories.json"
NORMALIZED_EVENTS_PATH = ARRIVAL_DIR / "eonet_last365_normalized_events.json"
SUMMARY_PATH = ARRIVAL_DIR / "eonet_mapping_summary.json"

GRID_STEP_DEGREES = 6.0
GRID_LAT_CELLS = int(180 / GRID_STEP_DEGREES)
GRID_LON_CELLS = int(360 / GRID_STEP_DEGREES)

SUBARCHETYPE_TO_ARCHETYPE = {
    "1A": "chain_like",
    "1B": "chain_like",
    "2A": "wide_shallow",
    "2B": "wide_shallow",
    "3A": "general",
    "3B": "general",
    "3C": "general",
}

CATEGORY_TO_SUBARCHETYPE = {
    "wildfires": "3B",
    "severe storms": "3B",
    "volcanoes": "1A",
    "floods": "2B",
    "drought": "1B",
    "sea and lake ice": "3C",
    "landslides": "1A",
    "earthquakes": "3A",
    "dust and haze": "2A",
    "snow": "3C",
    "temperature extremes": "1B",
    "water color": "3C",
    "manmade": "3A",
}


def fetch_json(base_url: str, params: dict[str, Any]) -> dict[str, Any]:
    url = f"{base_url}?{urlencode(params)}"
    with urlopen(url) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_events_by_category(categories: list[dict[str, Any]], days: int, limit: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    merged_events: dict[str, dict[str, Any]] = {}
    counts: dict[str, int] = {}
    for category in categories:
        category_id = category.get("id")
        if not category_id:
            continue
        payload = fetch_json(
            "https://eonet.gsfc.nasa.gov/api/v3/events",
            {"days": days, "status": "all", "limit": limit, "category": category_id},
        )
        events = payload.get("events", [])
        counts[category_id] = len(events)
        for event in events:
            event_id = str(event.get("id", f"{category_id}:{len(merged_events)}"))
            merged_events[event_id] = event
    return list(merged_events.values()), counts


def load_mapping_config() -> dict[str, Any] | None:
    if MAPPING_CONFIG_PATH.exists():
        return json.loads(MAPPING_CONFIG_PATH.read_text(encoding="utf-8"))
    return None


def normalize_category_name(category_name: str) -> str:
    return " ".join(category_name.strip().lower().split())


def category_to_subarchetype(category_name: str, mapping_config: dict[str, Any] | None) -> str:
    if mapping_config and category_name in mapping_config.get("categories", {}):
        return mapping_config["categories"][category_name]["primary_subarchetype"]

    normalized = normalize_category_name(category_name)
    if normalized in CATEGORY_TO_SUBARCHETYPE:
        return CATEGORY_TO_SUBARCHETYPE[normalized]
    if "wildfire" in normalized or "storm" in normalized:
        return "3B"
    if "flood" in normalized:
        return "2B"
    if "drought" in normalized or "temperature" in normalized:
        return "1B"
    if "ice" in normalized or "water" in normalized or "snow" in normalized:
        return "3C"
    if "volcano" in normalized or "landslide" in normalized:
        return "1A"
    if "earthquake" in normalized or "manmade" in normalized:
        return "3A"
    if "dust" in normalized or "haze" in normalized:
        return "2A"
    return "3A"


def flatten_coordinates(value: Any) -> list[tuple[float, float]]:
    if not isinstance(value, list):
        return []
    if len(value) >= 2 and all(isinstance(item, (int, float)) for item in value[:2]):
        return [(float(value[0]), float(value[1]))]

    points: list[tuple[float, float]] = []
    for item in value:
        points.extend(flatten_coordinates(item))
    return points


def extract_representative_point(geometry: dict[str, Any]) -> tuple[float, float] | None:
    coordinates = geometry.get("coordinates")
    points = flatten_coordinates(coordinates)
    if not points:
        return None

    lon = sum(point[0] for point in points) / len(points)
    lat = sum(point[1] for point in points) / len(points)
    return lat, lon


def to_region(lat: float, lon: float) -> tuple[str, int, int]:
    lat_index = min(max(int((lat + 90.0) / GRID_STEP_DEGREES), 0), GRID_LAT_CELLS - 1)
    lon_index = min(max(int((lon + 180.0) / GRID_STEP_DEGREES), 0), GRID_LON_CELLS - 1)
    return f"r{lat_index:02d}_{lon_index:02d}", lat_index, lon_index


def parse_eonet_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def normalize_events(
    events: list[dict[str, Any]],
    cutoff_utc: datetime,
    mapping_config: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    normalized_records: list[dict[str, Any]] = []
    category_counter = Counter()
    subarchetype_counter = Counter()
    archetype_counter = Counter()
    region_counter = Counter()

    for event in events:
        categories = event.get("categories") or []
        if not categories:
            continue

        category_name = categories[0].get("title", "unknown")
        subarchetype = category_to_subarchetype(category_name, mapping_config)
        archetype = SUBARCHETYPE_TO_ARCHETYPE[subarchetype]

        for geometry_index, geometry in enumerate(event.get("geometry", [])):
            point = extract_representative_point(geometry)
            if point is None:
                continue

            lat, lon = point
            region_id, lat_index, lon_index = to_region(lat, lon)
            timestamp = geometry.get("date") or event.get("closed") or event.get("geometryDate")
            parsed_timestamp = parse_eonet_timestamp(timestamp)
            if parsed_timestamp is None or parsed_timestamp < cutoff_utc:
                continue
            record_id = f"{event.get('id', 'unknown')}:{geometry_index}"
            source_titles = [source.get("id") for source in event.get("sources", [])]

            normalized_records.append(
                {
                    "record_id": record_id,
                    "event_id": event.get("id"),
                    "title": event.get("title"),
                    "category": category_name,
                    "primary_subarchetype": subarchetype,
                    "primary_archetype": archetype,
                    "timestamp": parsed_timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z"),
                    "latitude": round(lat, 4),
                    "longitude": round(lon, 4),
                    "region_id": region_id,
                    "region_row": lat_index,
                    "region_col": lon_index,
                    "geometry_type": geometry.get("type"),
                    "source_ids": source_titles,
                    "closed": event.get("closed"),
                    "link": event.get("link"),
                }
            )
            category_counter[category_name] += 1
            subarchetype_counter[subarchetype] += 1
            archetype_counter[archetype] += 1
            region_counter[region_id] += 1

    normalized_records.sort(key=lambda item: (item["timestamp"] or "", item["record_id"]))
    summary = {
        "total_normalized_records": len(normalized_records),
        "counts_by_category": dict(sorted(category_counter.items())),
        "counts_by_subarchetype": dict(sorted(subarchetype_counter.items())),
        "counts_by_archetype": dict(sorted(archetype_counter.items())),
        "top_regions": [
            {"region_id": region_id, "count": count}
            for region_id, count in region_counter.most_common(20)
        ],
    }
    return normalized_records, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and normalize NASA EONET event data.")
    parser.add_argument("--days", type=int, default=365, help="Look-back window in days.")
    parser.add_argument("--limit", type=int, default=5000, help="Maximum events returned by the EONET API.")
    args = parser.parse_args()

    ARRIVAL_DIR.mkdir(parents=True, exist_ok=True)

    categories_payload = fetch_json("https://eonet.gsfc.nasa.gov/api/v3/categories", {})
    mapping_config = load_mapping_config()
    categories = categories_payload.get("categories", [])
    events, per_category_counts = fetch_events_by_category(categories, args.days, args.limit)
    fetched_at = datetime.now(UTC)
    cutoff_utc = fetched_at.replace(microsecond=0) - timedelta(days=args.days)
    events_payload = {
        "title": "EONET Events",
        "description": "Merged category-wise EONET event collection.",
        "link": "https://eonet.gsfc.nasa.gov/api/v3/events",
        "days": args.days,
        "limit": args.limit,
        "cutoff_utc": cutoff_utc.isoformat().replace("+00:00", "Z"),
        "per_category_event_counts": per_category_counts,
        "events": events,
    }

    RAW_CATEGORIES_PATH.write_text(json.dumps(categories_payload, indent=2), encoding="utf-8")
    RAW_EVENTS_PATH.write_text(json.dumps(events_payload, indent=2), encoding="utf-8")

    normalized_records, normalization_summary = normalize_events(events, cutoff_utc, mapping_config)
    NORMALIZED_EVENTS_PATH.write_text(json.dumps(normalized_records, indent=2), encoding="utf-8")

    summary_payload = {
        "fetched_at_utc": fetched_at.isoformat(),
        "cutoff_utc": cutoff_utc.isoformat().replace("+00:00", "Z"),
        "days": args.days,
        "limit": args.limit,
        "grid_step_degrees": GRID_STEP_DEGREES,
        "grid_shape": {"lat_cells": GRID_LAT_CELLS, "lon_cells": GRID_LON_CELLS},
        "raw_event_count": len(events),
        "per_category_event_counts": per_category_counts,
        "category_to_subarchetype": CATEGORY_TO_SUBARCHETYPE,
        "mapping_config_used": mapping_config is not None,
        "normalization_summary": normalization_summary,
    }
    SUMMARY_PATH.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
