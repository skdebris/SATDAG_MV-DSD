from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]
ServiceTrafficMatrix = dict[tuple[str, str], float]


@dataclass(slots=True)
class ServiceSpec:
    service_id: str
    short_name: str
    name: str
    processing_level: str
    memory_mb: float
    storage_mb: float
    eta_gflops_per_ghz_s: float
    workload_distribution: JsonDict = field(default_factory=dict)


@dataclass(slots=True)
class TaskNode:
    task_id: str
    service_type: str
    service_id: str
    service_name: str
    processing_level: str
    workload_gflops: float
    memory_mb: float
    storage_mb: float
    eta_gflops_per_ghz_s: float
    input_size_mb: float = 0.0
    output_size_mb: float = 0.0
    metadata: JsonDict = field(default_factory=dict)


@dataclass(slots=True)
class DependencyEdge:
    src: str
    dst: str
    data_mb: float
    src_service_type: str
    dst_service_type: str


@dataclass(slots=True)
class RemoteSensingDAGRequest:
    request_id: str
    event_id: str
    region_id: str
    arrival_time_days: float
    arrival_timestamp_utc: str
    dag_type: str
    subarchetype: str
    dag_instance_id: str
    source_modalities: list[str]
    nodes: list[TaskNode]
    edges: list[DependencyEdge]
    mission_class: str | None = None
    num_nodes: int | None = None
    num_edges: int | None = None
    critical_path_len: int | None = None
    cp_workload_lb: float | None = None
    cp_data_lb: float | None = None
    lower_bound_time: float | None = None
    deadline_level: str | None = None
    deadline_slack_eta: float | None = None
    dag_slack_factor: float | None = None
    jitter_factor: float | None = None
    relative_deadline: float | None = None
    absolute_deadline: float | None = None
    cp_workload_lb_minutes: float | None = None
    cp_data_lb_minutes: float | None = None
    lower_bound_time_minutes: float | None = None
    relative_deadline_minutes: float | None = None
    absolute_deadline_days: float | None = None
    metadata: JsonDict = field(default_factory=dict)

    def node_map(self) -> dict[str, TaskNode]:
        return {node.task_id: node for node in self.nodes}

    def service_types(self) -> set[str]:
        return {node.service_type for node in self.nodes}


@dataclass(slots=True)
class DeploymentWindow:
    window_id: int
    start_time_days: float
    end_time_days: float
    requests: list[RemoteSensingDAGRequest]


@dataclass(slots=True)
class DeploymentWindowStats:
    window_id: int
    start_time_days: float
    end_time_days: float
    requests: list[RemoteSensingDAGRequest]
    dag_type_distribution: dict[str, float]
    subarchetype_distribution: dict[str, float]
    service_workload: dict[str, float]
    service_frequency: dict[str, int]
    service_traffic: ServiceTrafficMatrix
    critical_path_service_count: dict[str, float]
    critical_path_service_weight: dict[str, float]
    service_sync_weight: dict[str, float]
    arrival_rate_per_day: float
    request_count: int
    metadata: JsonDict = field(default_factory=dict)


@dataclass(slots=True)
class Satellite:
    satellite_id: str
    plane_index: int
    slot_index: int
    cpu_capacity_ghz: float
    memory_mb: float
    storage_mb: float
    container_slots: int
    metadata: JsonDict = field(default_factory=dict)


@dataclass(slots=True)
class LinkState:
    src: str
    dst: str
    capacity_mbps: float
    propagation_delay_ms: float
    available: bool = True


@dataclass(slots=True)
class GraphSnapshot:
    slot_index: int
    time_offset_minutes: float
    adjacency: dict[str, list[LinkState]]


@dataclass(slots=True)
class ContactPlanEdge:
    slot_index: int
    time_start_minutes: float
    time_end_minutes: float
    src: str
    dst: str
    capacity_mbps: float
    propagation_delay_ms: float
    contact_volume_mb: float


@dataclass(slots=True)
class SatelliteEnvironment:
    satellites: dict[str, Satellite]
    snapshots: list[GraphSnapshot]
    contact_plan: list[ContactPlanEdge]
    aggregate_graph: dict[str, dict[str, float]]
    aggregate_threshold: float
    slot_duration_minutes: float
    planning_horizon_minutes: float
    density_mode: str
    perturbation_mode: str
    metadata: JsonDict = field(default_factory=dict)


@dataclass(slots=True)
class DeploymentPlan:
    algorithm_name: str
    window_id: int
    service_placement: dict[str, list[str]]
    cpu_allocation: dict[tuple[str, str], float]
    replica_count: dict[str, int]
    metadata: JsonDict = field(default_factory=dict)


@dataclass(slots=True)
class RouteRecord:
    edge_key: str
    src_task_id: str
    dst_task_id: str
    src_satellite_id: str
    dst_satellite_id: str
    data_mb: float
    path: list[str]
    start_time_days: float
    finish_time_days: float
    waiting_delay_days: float
    transmission_delay_days: float
    propagation_delay_days: float
    success: bool
    failure_reason: str | None = None

    @property
    def total_delay_days(self) -> float:
        return self.waiting_delay_days + self.transmission_delay_days + self.propagation_delay_days


@dataclass(slots=True)
class TaskExecutionRecord:
    request_id: str
    task_id: str
    service_type: str
    satellite_id: str
    start_time_days: float
    finish_time_days: float
    compute_time_days: float
    predecessor_ready_time_days: float
    scheduler_priority: float
    selected_reason: str


@dataclass(slots=True)
class ExecutionResult:
    request_id: str
    window_id: int
    success: bool
    arrival_time_days: float
    start_time_days: float
    finish_time_days: float
    makespan_days: float
    task_records: list[TaskExecutionRecord]
    route_records: list[RouteRecord]
    cp_delay_breakdown_days: dict[str, float]
    cpu_utilization: dict[str, float]
    energy: float | None
    failure_reason: str | None
    metadata: JsonDict = field(default_factory=dict)


@dataclass(slots=True)
class SimulationArtifacts:
    output_dir: Path
    request_dataset_path: Path
    summary_path: Path
    per_request_path: Path
    deployment_plan_path: Path
