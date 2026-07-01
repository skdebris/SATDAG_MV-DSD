from __future__ import annotations

from .base import BaseDeploymentAlgorithm
from .baselines import (
    DRLDeployProxyAlgorithm,
    DependencyBlindAlgorithm,
    GreedyResourceAlgorithm,
    RandomDeployAlgorithm,
    SFCPathDecompAlgorithm,
)
from .cpmv_dsd import CPMVDSDAlgorithm
from .literature_baselines import FloodSFCPGreedyAlgorithm, JSDTSAOSAlgorithm, OnDocSatAlgorithm


def build_algorithm(name: str, seed: int = 0) -> BaseDeploymentAlgorithm:
    normalized = name.lower().replace("-", "_")
    if normalized == "cpmv_dsd":
        return CPMVDSDAlgorithm(seed=seed)
    if normalized == "dependency_blind":
        return DependencyBlindAlgorithm()
    if normalized == "sfc_path_decomp":
        return SFCPathDecompAlgorithm()
    if normalized == "greedy_resource":
        return GreedyResourceAlgorithm()
    if normalized in {"jsdts_aos_sat", "jsdts_aos"}:
        return JSDTSAOSAlgorithm(seed=seed)
    if normalized in {"ondoc_sat", "ondoc"}:
        return OnDocSatAlgorithm()
    if normalized in {"floodsfcp_greedy", "floodsfcp_greedy_sat"}:
        return FloodSFCPGreedyAlgorithm()
    if normalized == "random_deploy":
        return RandomDeployAlgorithm(seed=seed)
    if normalized == "drl_deploy":
        return DRLDeployProxyAlgorithm(seed=seed)
    raise ValueError(f"Unsupported deployment algorithm: {name}")


__all__ = ["BaseDeploymentAlgorithm", "build_algorithm"]
