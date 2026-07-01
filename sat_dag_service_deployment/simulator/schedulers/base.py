from __future__ import annotations

from ..models import DeploymentPlan, DeploymentWindow, ExecutionResult, RemoteSensingDAGRequest, SatelliteEnvironment


class BaseScheduler:
    name = "base"

    def reset_window(self, window: DeploymentWindow) -> None:
        raise NotImplementedError

    def schedule(
        self,
        request: RemoteSensingDAGRequest,
        deployment_plan: DeploymentPlan,
        env: SatelliteEnvironment,
        current_time: float,
        config: dict,
    ) -> ExecutionResult:
        raise NotImplementedError

