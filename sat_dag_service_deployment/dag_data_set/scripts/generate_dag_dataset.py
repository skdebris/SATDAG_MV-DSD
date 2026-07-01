from __future__ import annotations

import json
import random
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = BASE_DIR / "dataset"
TEMPLATES_DIR = DATASET_DIR / "dag_templates"
CHAIN_DIR = TEMPLATES_DIR / "chain_like"
WIDE_DIR = TEMPLATES_DIR / "wide_shallow"
GENERAL_DIR = TEMPLATES_DIR / "general"
README_PATH = DATASET_DIR / "README.md"
SERVICE_LIBRARY_PATH = DATASET_DIR / "service_library.json"
SUMMARY_PATH = TEMPLATES_DIR / "dag_instances_summary.json"

RNG_SEED = 20260427
P_BRANCH = 0.3
P_MERGE = 0.4


@dataclass(frozen=True)
class ServiceSpec:
    service_id: str
    short_name: str
    name: str
    processing_level: str
    workload_distribution: dict[str, float | str]
    memory_mb: int
    storage_mb: int
    eta_gflops_per_ghz_s: float


SERVICE_LIBRARY: list[ServiceSpec] = [
    ServiceSpec("f1", "DA", "Data Acquisition", "L0", {"type": "uniform", "min": 1.0, "max": 5.0}, 200, 100, 0.9),
    ServiceSpec("f2", "RC", "Radiometric Correction", "L0-L1A", {"type": "normal", "mu": 12.0, "sigma": 3.0}, 500, 200, 1.0),
    ServiceSpec("f3", "GC", "Geometric Correction", "L1A-L1B", {"type": "normal", "mu": 50.0, "sigma": 10.0}, 800, 300, 0.8),
    ServiceSpec("f4", "CM", "Cloud Masking", "L1B", {"type": "normal", "mu": 20.0, "sigma": 5.0}, 1000, 400, 0.9),
    ServiceSpec("f5", "IF", "Image Fusion", "L1C-L2", {"type": "normal", "mu": 160.0, "sigma": 30.0}, 1200, 400, 0.7),
    ServiceSpec("f6", "FE", "Feature Extraction", "L2", {"type": "normal", "mu": 80.0, "sigma": 15.0}, 1500, 500, 0.85),
    ServiceSpec("f7", "TD", "Target Detection", "L2-L3", {"type": "normal", "mu": 320.0, "sigma": 60.0}, 2500, 800, 0.6),
    ServiceSpec("f8", "RA", "Result Aggregation", "L3-L4", {"type": "uniform", "min": 1.0, "max": 10.0}, 300, 100, 1.0),
]

SERVICE_BY_SHORT = {service.short_name: service for service in SERVICE_LIBRARY}

EDGE_PROFILES: dict[tuple[str, str], tuple[float, float]] = {
    ("DA", "RC"): (15.0, 35.0),
    ("RC", "GC"): (15.0, 35.0),
    ("RC", "CM"): (50.0, 150.0),
    ("GC", "CM"): (150.0, 250.0),
    ("GC", "IF"): (300.0, 700.0),
    ("GC", "FE"): (300.0, 600.0),
    ("GC", "RA"): (80.0, 200.0),
    ("CM", "IF"): (50.0, 150.0),
    ("CM", "FE"): (50.0, 120.0),
    ("IF", "FE"): (80.0, 200.0),
    ("IF", "TD"): (80.0, 200.0),
    ("FE", "FE"): (10.0, 30.0),
    ("FE", "TD"): (20.0, 60.0),
    ("FE", "RA"): (5.0, 30.0),
    ("TD", "RA"): (0.05, 5.0),
    ("RA", "FE"): (5.0, 20.0),
}

INSTANCE_COUNTS = [
    ("chain_like", "1A", 12, "chain"),
    ("chain_like", "1B", 8, "chain"),
    ("wide_shallow", "2A", 12, "ws"),
    ("wide_shallow", "2B", 8, "ws"),
    ("general", "3A", 10, "general"),
    ("general", "3B", 6, "general"),
    ("general", "3C", 4, "general"),
]


def ensure_output_dirs() -> None:
    for path in (CHAIN_DIR, WIDE_DIR, GENERAL_DIR):
        path.mkdir(parents=True, exist_ok=True)


def sample_workload(service_short_name: str, rng: random.Random) -> float:
    distribution = SERVICE_BY_SHORT[service_short_name].workload_distribution
    if distribution["type"] == "uniform":
        value = rng.uniform(distribution["min"], distribution["max"])
        return round(value, 1)

    mu = float(distribution["mu"])
    sigma = float(distribution["sigma"])
    lower = max(0.5, mu - 3.0 * sigma)
    upper = mu + 3.0 * sigma
    while True:
        candidate = rng.gauss(mu, sigma)
        if lower <= candidate <= upper:
            return round(candidate, 1)


def sample_edge_size(src_service: str, dst_service: str, rng: random.Random) -> float:
    if (src_service, dst_service) not in EDGE_PROFILES:
        raise KeyError(f"Missing edge profile for {src_service} -> {dst_service}")
    low, high = EDGE_PROFILES[(src_service, dst_service)]
    return round(rng.uniform(low, high), 2)


def add_node(nodes: list[dict], service_short_name: str, tag: str) -> str:
    node_id = f"v{len(nodes) + 1}"
    spec = SERVICE_BY_SHORT[service_short_name]
    nodes.append(
        {
            "id": node_id,
            "tag": tag,
            "service_type": spec.short_name,
            "service_id": spec.service_id,
            "service_name": spec.name,
            "processing_level": spec.processing_level,
        }
    )
    return node_id


def add_edge(edges: set[tuple[str, str]], src: str, dst: str) -> None:
    if src == dst:
        raise ValueError("Self loops are not allowed in DAG instances")
    edges.add((src, dst))


def service_of(node_id: str, node_map: dict[str, dict]) -> str:
    return node_map[node_id]["service_type"]


def topo_levels(nodes: list[dict], edges: list[tuple[str, str]]) -> tuple[list[str], dict[str, int]]:
    adjacency: dict[str, list[str]] = defaultdict(list)
    indegree: dict[str, int] = {node["id"]: 0 for node in nodes}
    for src, dst in edges:
        adjacency[src].append(dst)
        indegree[dst] += 1

    queue = deque(sorted(node_id for node_id, degree in indegree.items() if degree == 0))
    order: list[str] = []
    level: dict[str, int] = {}
    while queue:
        node_id = queue.popleft()
        order.append(node_id)
        if node_id not in level:
            level[node_id] = 0
        for neighbor in adjacency[node_id]:
            level[neighbor] = max(level.get(neighbor, 0), level[node_id] + 1)
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)

    if len(order) != len(nodes):
        raise ValueError("Generated graph contains a cycle")
    return order, level


def build_metadata(
    instance_id: str,
    archetype: str,
    subarchetype: str,
    nodes: list[dict],
    edge_payload: list[dict],
) -> dict:
    edge_pairs = [(edge["src"], edge["dst"]) for edge in edge_payload]
    order, level = topo_levels(nodes, edge_pairs)
    indegree = Counter()
    outdegree = Counter()
    for src, dst in edge_pairs:
        indegree[dst] += 1
        outdegree[src] += 1

    level_width = Counter(level[node_id] for node_id in order)
    width = max(level_width.values())
    depth = max(level.values()) + 1
    sources = sorted(node["id"] for node in nodes if indegree[node["id"]] == 0)
    sinks = sorted(node["id"] for node in nodes if outdegree[node["id"]] == 0)
    service_distribution = Counter(node["service_type"] for node in nodes)
    total_workload = round(sum(node["workload_GFLOPs"] for node in nodes), 2)
    total_data = round(sum(edge["data_MB"] for edge in edge_payload), 2)

    return {
        "instance_id": instance_id,
        "archetype": archetype,
        "subarchetype": subarchetype,
        "num_nodes": len(nodes),
        "num_edges": len(edge_pairs),
        "num_sources": len(sources),
        "num_sinks": len(sinks),
        "source_nodes": sources,
        "sink_nodes": sinks,
        "width": width,
        "depth": depth,
        "structure_type": archetype,
        "service_distribution": dict(sorted(service_distribution.items())),
        "total_workload_GFLOPs": total_workload,
        "total_data_MB": total_data,
    }


def attach_sampled_parameters(nodes: list[dict], edges: set[tuple[str, str]], rng: random.Random) -> tuple[list[dict], list[dict]]:
    node_map = {node["id"]: node for node in nodes}
    for node in nodes:
        spec = SERVICE_BY_SHORT[node["service_type"]]
        node["workload_GFLOPs"] = sample_workload(spec.short_name, rng)
        node["memory_MB"] = spec.memory_mb
        node["storage_MB"] = spec.storage_mb
        node["eta_gflops_per_ghz_s"] = spec.eta_gflops_per_ghz_s

    edge_payload: list[dict] = []
    for src, dst in sorted(edges):
        edge_payload.append(
            {
                "src": src,
                "dst": dst,
                "src_service_type": node_map[src]["service_type"],
                "dst_service_type": node_map[dst]["service_type"],
                "data_MB": sample_edge_size(
                    node_map[src]["service_type"],
                    node_map[dst]["service_type"],
                    rng,
                ),
            }
        )
    return nodes, edge_payload


def build_chain_1a(rng: random.Random) -> tuple[list[dict], set[tuple[str, str]]]:
    nodes: list[dict] = []
    edges: set[tuple[str, str]] = set()
    sequence = ["DA", "RC", "GC", "CM", "IF", "FE", "TD", "RA"]
    sequence_ids = [add_node(nodes, service, f"main_{idx}") for idx, service in enumerate(sequence, start=1)]
    for src, dst in zip(sequence_ids, sequence_ids[1:]):
        add_edge(edges, src, dst)

    if rng.random() < 0.55:
        extra_fe = add_node(nodes, "FE", "deep_feature_refinement")
        add_edge(edges, sequence_ids[5], extra_fe)
        add_edge(edges, extra_fe, sequence_ids[6])

    if rng.random() < P_BRANCH:
        branch_cm = add_node(nodes, "CM", "auxiliary_masking")
        add_edge(edges, sequence_ids[2], branch_cm)
        if rng.random() < P_MERGE:
            add_edge(edges, branch_cm, sequence_ids[4])
        else:
            add_edge(edges, branch_cm, sequence_ids[5])

    return nodes, edges


def build_chain_1b(rng: random.Random) -> tuple[list[dict], set[tuple[str, str]]]:
    nodes: list[dict] = []
    edges: set[tuple[str, str]] = set()
    sequence = ["DA", "RC", "GC", "CM", "FE", "FE", "FE", "TD", "RA"]
    sequence_ids = [add_node(nodes, service, f"main_{idx}") for idx, service in enumerate(sequence, start=1)]
    for src, dst in zip(sequence_ids, sequence_ids[1:]):
        add_edge(edges, src, dst)

    if rng.random() < 0.6:
        temporal_refine = add_node(nodes, "FE", "temporal_refinement")
        add_edge(edges, sequence_ids[5], temporal_refine)
        add_edge(edges, temporal_refine, sequence_ids[6])

    if rng.random() < P_BRANCH:
        branch_fe = add_node(nodes, "FE", "auxiliary_temporal_branch")
        add_edge(edges, sequence_ids[3], branch_fe)
        if rng.random() < P_MERGE:
            add_edge(edges, branch_fe, sequence_ids[7])
        else:
            add_edge(edges, branch_fe, sequence_ids[6])

    return nodes, edges


def build_wide_2a(rng: random.Random) -> tuple[list[dict], set[tuple[str, str]]]:
    nodes: list[dict] = []
    edges: set[tuple[str, str]] = set()
    source_count = rng.randint(4, 5)
    gc_nodes: list[str] = []
    for source_index in range(source_count):
        da = add_node(nodes, "DA", f"source_{source_index + 1}_da")
        rc = add_node(nodes, "RC", f"source_{source_index + 1}_rc")
        gc = add_node(nodes, "GC", f"source_{source_index + 1}_gc")
        add_edge(edges, da, rc)
        add_edge(edges, rc, gc)
        gc_nodes.append(gc)

    fusion = add_node(nodes, "IF", "global_fusion")
    td = add_node(nodes, "TD", "detection_head")
    ra = add_node(nodes, "RA", "final_aggregation")
    for gc in gc_nodes:
        add_edge(edges, gc, fusion)
    add_edge(edges, fusion, td)
    add_edge(edges, td, ra)

    if rng.random() < P_BRANCH:
        extra_fe = add_node(nodes, "FE", "fusion_side_analysis")
        add_edge(edges, fusion, extra_fe)
        if rng.random() < P_MERGE:
            add_edge(edges, extra_fe, td)
        else:
            add_edge(edges, extra_fe, ra)

    if rng.random() < 0.25:
        optical_cm = add_node(nodes, "CM", "optical_cloud_mask")
        add_edge(edges, gc_nodes[0], optical_cm)
        add_edge(edges, optical_cm, fusion)

    return nodes, edges


def build_wide_2b(rng: random.Random) -> tuple[list[dict], set[tuple[str, str]]]:
    nodes: list[dict] = []
    edges: set[tuple[str, str]] = set()
    region_count = 5
    gc_nodes: list[str] = []
    for region_index in range(region_count):
        da = add_node(nodes, "DA", f"region_{region_index + 1}_da")
        rc = add_node(nodes, "RC", f"region_{region_index + 1}_rc")
        gc = add_node(nodes, "GC", f"region_{region_index + 1}_gc")
        add_edge(edges, da, rc)
        add_edge(edges, rc, gc)
        gc_nodes.append(gc)

    ra = add_node(nodes, "RA", "cross_region_aggregation")
    fe = add_node(nodes, "FE", "global_feature_extraction")
    td = add_node(nodes, "TD", "global_detection")
    for gc in gc_nodes:
        add_edge(edges, gc, ra)
    add_edge(edges, ra, fe)
    add_edge(edges, fe, td)

    if rng.random() < P_BRANCH:
        extra_fe = add_node(nodes, "FE", "regional_priority_analysis")
        add_edge(edges, ra, extra_fe)
        add_edge(edges, extra_fe, td)

    return nodes, edges


def build_general_3a(rng: random.Random) -> tuple[list[dict], set[tuple[str, str]]]:
    nodes: list[dict] = []
    edges: set[tuple[str, str]] = set()

    da_opt = add_node(nodes, "DA", "optical_da")
    da_sar = add_node(nodes, "DA", "sar_da")
    rc_opt = add_node(nodes, "RC", "optical_rc")
    rc_sar = add_node(nodes, "RC", "sar_rc")
    cm_opt = add_node(nodes, "CM", "optical_cm")
    gc_opt = add_node(nodes, "GC", "optical_gc")
    gc_sar = add_node(nodes, "GC", "sar_gc")
    fusion = add_node(nodes, "IF", "modality_fusion")
    fe_main = add_node(nodes, "FE", "main_feature_extraction")
    fe_aux = add_node(nodes, "FE", "aux_feature_extraction")
    td = add_node(nodes, "TD", "main_detection")
    ra = add_node(nodes, "RA", "final_aggregation")

    for src, dst in (
        (da_opt, rc_opt),
        (da_sar, rc_sar),
        (rc_opt, gc_opt),
        (rc_opt, cm_opt),
        (rc_sar, gc_sar),
        (gc_opt, fusion),
        (gc_sar, fusion),
        (cm_opt, fusion),
        (fusion, fe_main),
        (fe_main, fe_aux),
        (fe_aux, td),
        (td, ra),
    ):
        add_edge(edges, src, dst)

    if rng.random() < P_BRANCH:
        td_side = add_node(nodes, "TD", "parallel_detection_head")
        add_edge(edges, fe_main, td_side)
        add_edge(edges, td_side, ra)

    return nodes, edges


def build_general_3b(rng: random.Random) -> tuple[list[dict], set[tuple[str, str]]]:
    nodes: list[dict] = []
    edges: set[tuple[str, str]] = set()

    da_opt = add_node(nodes, "DA", "optical_da")
    da_ir = add_node(nodes, "DA", "infrared_da")
    rc_opt = add_node(nodes, "RC", "optical_rc")
    rc_ir = add_node(nodes, "RC", "infrared_rc")
    gc_opt = add_node(nodes, "GC", "optical_gc")
    gc_ir = add_node(nodes, "GC", "infrared_gc")
    fusion = add_node(nodes, "IF", "early_fusion")
    td_fast = add_node(nodes, "TD", "fast_warning_detection")
    ra_fast = add_node(nodes, "RA", "fast_warning_report")
    fe_deep = add_node(nodes, "FE", "deep_feature_analysis")
    td_full = add_node(nodes, "TD", "full_resolution_detection")
    ra_full = add_node(nodes, "RA", "detailed_report")

    for src, dst in (
        (da_opt, rc_opt),
        (da_ir, rc_ir),
        (rc_opt, gc_opt),
        (rc_ir, gc_ir),
        (gc_opt, fusion),
        (gc_ir, fusion),
        (fusion, td_fast),
        (td_fast, ra_fast),
        (fusion, fe_deep),
        (fe_deep, td_full),
        (td_full, ra_full),
    ):
        add_edge(edges, src, dst)

    if rng.random() < 0.45:
        da_aux = add_node(nodes, "DA", "auxiliary_da")
        rc_aux = add_node(nodes, "RC", "auxiliary_rc")
        gc_aux = add_node(nodes, "GC", "auxiliary_gc")
        add_edge(edges, da_aux, rc_aux)
        add_edge(edges, rc_aux, gc_aux)
        add_edge(edges, gc_aux, fusion)

    return nodes, edges


def build_general_3c(rng: random.Random) -> tuple[list[dict], set[tuple[str, str]]]:
    nodes: list[dict] = []
    edges: set[tuple[str, str]] = set()

    gc_nodes: list[str] = []
    for timestamp in ("t1", "t2", "t3"):
        da = add_node(nodes, "DA", f"{timestamp}_da")
        rc = add_node(nodes, "RC", f"{timestamp}_rc")
        gc = add_node(nodes, "GC", f"{timestamp}_gc")
        add_edge(edges, da, rc)
        add_edge(edges, rc, gc)
        gc_nodes.append(gc)

    fusion = add_node(nodes, "IF", "temporal_fusion")
    fe_a = add_node(nodes, "FE", "temporal_feature_a")
    fe_b = add_node(nodes, "FE", "temporal_feature_b")
    ra = add_node(nodes, "RA", "cross_time_aggregation")
    for gc in gc_nodes:
        add_edge(edges, gc, fusion)
    add_edge(edges, fusion, fe_a)
    add_edge(edges, fusion, fe_b)
    add_edge(edges, fe_a, ra)
    add_edge(edges, fe_b, ra)

    if rng.random() < P_BRANCH:
        fe_c = add_node(nodes, "FE", "temporal_feature_c")
        add_edge(edges, fusion, fe_c)
        add_edge(edges, fe_c, ra)

    return nodes, edges


BUILDERS: dict[str, Callable[[random.Random], tuple[list[dict], set[tuple[str, str]]]]] = {
    "1A": build_chain_1a,
    "1B": build_chain_1b,
    "2A": build_wide_2a,
    "2B": build_wide_2b,
    "3A": build_general_3a,
    "3B": build_general_3b,
    "3C": build_general_3c,
}


def output_dir_for(archetype: str) -> Path:
    return {"chain_like": CHAIN_DIR, "wide_shallow": WIDE_DIR, "general": GENERAL_DIR}[archetype]


def filename_for(prefix: str, subarchetype: str, index: int) -> str:
    return f"{prefix}_{subarchetype}_{index:03d}.json"


def generate_instance(
    archetype: str,
    subarchetype: str,
    prefix: str,
    index: int,
    seed_offset: int,
) -> dict:
    rng = random.Random(RNG_SEED + seed_offset)
    nodes, edge_pairs = BUILDERS[subarchetype](rng)
    nodes, edges = attach_sampled_parameters(nodes, edge_pairs, rng)
    instance_id = filename_for(prefix, subarchetype, index).removesuffix(".json")
    metadata = build_metadata(instance_id, archetype, subarchetype, nodes, edges)
    return {
        "instance_id": instance_id,
        "archetype": archetype,
        "subarchetype": subarchetype,
        "nodes": nodes,
        "edges": edges,
        "metadata": metadata,
    }


def write_service_library() -> None:
    payload = {
        "generator_seed": RNG_SEED,
        "probabilities": {"p_branch": P_BRANCH, "p_merge": P_MERGE},
        "services": [
            {
                "service_id": service.service_id,
                "short_name": service.short_name,
                "name": service.name,
                "processing_level": service.processing_level,
                "workload_distribution": service.workload_distribution,
                "memory_MB": service.memory_mb,
                "storage_MB": service.storage_mb,
                "eta_gflops_per_ghz_s": service.eta_gflops_per_ghz_s,
            }
            for service in SERVICE_LIBRARY
        ],
        "edge_profiles_MB": {
            f"{src}->{dst}": {"min": low, "max": high}
            for (src, dst), (low, high) in sorted(EDGE_PROFILES.items())
        },
    }
    SERVICE_LIBRARY_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_readme(summary: dict) -> None:
    lines = [
        "# Synthetic DAG Dataset",
        "",
        "This dataset is generated from `dag_dataset_generation_guide.md`.",
        "",
        "## Contents",
        "",
        "- `service_library.json`: service types, resource defaults, and workload distributions.",
        "- `dag_templates/`: 60 generated DAG template instances.",
        "- `dag_templates/dag_instances_summary.json`: aggregate statistics for all instances.",
        "- `arrival_traces/`: external reference arrival-process data and preprocessing outputs.",
        "",
        "## Generation Summary",
        "",
        f"- Total instances: {summary['total_instances']}",
        f"- Structural classes: {', '.join(sorted(summary['instances_by_archetype'].keys()))}",
        f"- Average nodes per instance: {summary['average_nodes']}",
        f"- Average edges per instance: {summary['average_edges']}",
        f"- Average depth: {summary['average_depth']}",
        f"- Average width: {summary['average_width']}",
        "",
        "## Template Directories",
        "",
        "- `dag_templates/chain_like`: subarchetypes 1A and 1B",
        "- `dag_templates/wide_shallow`: subarchetypes 2A and 2B",
        "- `dag_templates/general`: subarchetypes 3A, 3B, and 3C",
        "",
    ]
    README_PATH.write_text("\n".join(lines), encoding="utf-8")


def build_summary(instances: list[dict]) -> dict:
    archetype_counts = Counter(instance["archetype"] for instance in instances)
    subarchetype_counts = Counter(instance["subarchetype"] for instance in instances)
    node_counts = [instance["metadata"]["num_nodes"] for instance in instances]
    edge_counts = [instance["metadata"]["num_edges"] for instance in instances]
    depths = [instance["metadata"]["depth"] for instance in instances]
    widths = [instance["metadata"]["width"] for instance in instances]
    service_frequency = Counter()
    for instance in instances:
        service_frequency.update(instance["metadata"]["service_distribution"])

    return {
        "generator_seed": RNG_SEED,
        "total_instances": len(instances),
        "instances_by_archetype": dict(sorted(archetype_counts.items())),
        "instances_by_subarchetype": dict(sorted(subarchetype_counts.items())),
        "average_nodes": round(sum(node_counts) / len(node_counts), 2),
        "average_edges": round(sum(edge_counts) / len(edge_counts), 2),
        "average_depth": round(sum(depths) / len(depths), 2),
        "average_width": round(sum(widths) / len(widths), 2),
        "node_count_range": {"min": min(node_counts), "max": max(node_counts)},
        "edge_count_range": {"min": min(edge_counts), "max": max(edge_counts)},
        "depth_range": {"min": min(depths), "max": max(depths)},
        "width_range": {"min": min(widths), "max": max(widths)},
        "service_frequency": dict(sorted(service_frequency.items())),
        "instances": [
            {
                "instance_id": instance["instance_id"],
                "archetype": instance["archetype"],
                "subarchetype": instance["subarchetype"],
                "num_nodes": instance["metadata"]["num_nodes"],
                "num_edges": instance["metadata"]["num_edges"],
                "width": instance["metadata"]["width"],
                "depth": instance["metadata"]["depth"],
                "total_workload_GFLOPs": instance["metadata"]["total_workload_GFLOPs"],
                "total_data_MB": instance["metadata"]["total_data_MB"],
            }
            for instance in instances
        ],
    }


def main() -> None:
    ensure_output_dirs()
    write_service_library()

    instances: list[dict] = []
    seed_offset = 0
    for archetype, subarchetype, count, prefix in INSTANCE_COUNTS:
        output_dir = output_dir_for(archetype)
        for index in range(1, count + 1):
            instance = generate_instance(archetype, subarchetype, prefix, index, seed_offset)
            seed_offset += 1
            file_path = output_dir / f"{instance['instance_id']}.json"
            file_path.write_text(json.dumps(instance, indent=2), encoding="utf-8")
            instances.append(instance)

    summary = build_summary(instances)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_readme(summary)


if __name__ == "__main__":
    main()
