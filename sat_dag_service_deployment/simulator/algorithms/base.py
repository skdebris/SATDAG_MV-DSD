from __future__ import annotations

from ..models import DeploymentPlan, DeploymentWindowStats, SatelliteEnvironment, ServiceSpec


class BaseDeploymentAlgorithm:
    name = "base"

    def deploy(
        self,
        window_stats: DeploymentWindowStats,
        env: SatelliteEnvironment,
        service_catalog: dict[str, ServiceSpec],
        config: dict,
    ) -> DeploymentPlan:
        raise NotImplementedError

