from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def utc_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def safe_mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * q
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def normalize_dict(values: dict[str, float]) -> dict[str, float]:
    total = sum(values.values())
    if total <= 0:
        return {key: 0.0 for key in values}
    return {key: value / total for key, value in values.items()}


def min_max_normalize(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    lower = min(values.values())
    upper = max(values.values())
    if math.isclose(lower, upper):
        return {key: 1.0 for key in values}
    scale = upper - lower
    return {key: (value - lower) / scale for key, value in values.items()}


def topological_sort(node_ids: Iterable[str], edges: Iterable[tuple[str, str]]) -> list[str]:
    node_ids = list(node_ids)
    adjacency: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
    indegree = {node_id: 0 for node_id in node_ids}
    for src, dst in edges:
        adjacency[src].append(dst)
        indegree[dst] += 1

    queue = sorted(node_id for node_id, degree in indegree.items() if degree == 0)
    order: list[str] = []
    while queue:
        node_id = queue.pop(0)
        order.append(node_id)
        for child in adjacency[node_id]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
                queue.sort()
    if len(order) != len(adjacency):
        raise ValueError("Graph contains a cycle")
    return order


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
