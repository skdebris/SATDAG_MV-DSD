from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from ..utils import dump_json, ensure_dir
from .loaders import load_arrival_trace, load_template_library


MODALITY_BY_SUBARCHETYPE = {
    "1A": ["optical"],
    "1B": ["optical", "multispectral"],
    "2A": ["optical", "sar"],
    "2B": ["optical", "sar", "hyperspectral"],
    "3A": ["multispectral", "sar"],
    "3B": ["optical", "sar", "thermal"],
    "3C": ["optical", "sar", "hyperspectral", "thermal"],
}


def _template_node_io(template: dict[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
    incoming: dict[str, float] = {node["id"]: 0.0 for node in template["nodes"]}
    outgoing: dict[str, float] = {node["id"]: 0.0 for node in template["nodes"]}
    for edge in template["edges"]:
        incoming[edge["dst"]] += float(edge["data_MB"])
        outgoing[edge["src"]] += float(edge["data_MB"])
    return incoming, outgoing


def _build_request_payload(arrival: dict[str, Any], template: dict[str, Any]) -> dict[str, Any]:
    incoming, outgoing = _template_node_io(template)
    nodes = []
    for node in template["nodes"]:
        nodes.append(
            {
                "task_id": node["id"],
                "service_type": node["service_type"],
                "service_id": node["service_id"],
                "service_name": node["service_name"],
                "processing_level": node["processing_level"],
                "workload_gflops": float(node["workload_GFLOPs"]),
                "memory_mb": float(node["memory_MB"]),
                "storage_mb": float(node["storage_MB"]),
                "eta_gflops_per_ghz_s": float(node["eta_gflops_per_ghz_s"]),
                "input_size_mb": round(incoming[node["id"]], 4),
                "output_size_mb": round(outgoing[node["id"]], 4),
                "metadata": {"tag": node["tag"]},
            }
        )

    edges = []
    for edge in template["edges"]:
        edges.append(
            {
                "src": edge["src"],
                "dst": edge["dst"],
                "data_mb": float(edge["data_MB"]),
                "src_service_type": edge["src_service_type"],
                "dst_service_type": edge["dst_service_type"],
            }
        )

    return {
        "request_id": arrival["task_request_id"],
        "event_id": arrival["event_id"],
        "region_id": arrival["region_id"],
        "arrival_time_days": float(arrival["simulation_time_days"]),
        "arrival_timestamp_utc": arrival["timestamp"],
        "dag_type": arrival["dag_archetype"],
        "subarchetype": arrival["subarchetype"],
        "dag_instance_id": arrival["dag_instance_id"],
        "source_modalities": MODALITY_BY_SUBARCHETYPE.get(arrival["subarchetype"], ["optical"]),
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "generation": arrival.get("generation", 0),
            "event_source": arrival.get("event_source"),
            "parent_event_id": arrival.get("parent_event_id"),
            "template_metadata": template["metadata"],
        },
    }


def ensure_materialized_requests(
    arrival_trace_path: str | Path,
    template_dir: str | Path,
    output_path: str | Path,
) -> tuple[Path, Path]:
    output_path = Path(output_path)
    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
    if output_path.exists() and summary_path.exists():
        return output_path, summary_path

    ensure_dir(output_path.parent)
    arrivals = load_arrival_trace(arrival_trace_path)
    templates = load_template_library(template_dir)

    subtype_counts = Counter()
    dag_type_counts = Counter()
    with output_path.open("w", encoding="utf-8") as handle:
        for arrival in arrivals:
            template = templates[arrival["dag_instance_id"]]
            payload = _build_request_payload(arrival, template)
            subtype_counts[payload["subarchetype"]] += 1
            dag_type_counts[payload["dag_type"]] += 1
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    summary = {
        "request_count": len(arrivals),
        "subarchetype_distribution": dict(sorted(subtype_counts.items())),
        "dag_type_distribution": dict(sorted(dag_type_counts.items())),
        "source_arrival_trace": str(arrival_trace_path),
        "source_template_dir": str(template_dir),
        "materialized_output": str(output_path),
    }
    dump_json(summary_path, summary)
    return output_path, summary_path

