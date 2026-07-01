from __future__ import annotations

from .base import BaseScheduler
from .ccp_mdag import CCPMDAGScheduler
from .etf import ETFScheduler
from .heft import HEFTScheduler
from .peft import PEFTScheduler
from .priority import PriorityScheduler


def build_scheduler(name: str) -> BaseScheduler:
    normalized = name.lower().replace("-", "_")
    if normalized == "heft":
        return HEFTScheduler()
    if normalized in {"peft", "peft_sat"}:
        return PEFTScheduler()
    if normalized in {"ccp_mdag", "ccp_mdag_sat"}:
        return CCPMDAGScheduler()
    if normalized == "etf":
        return ETFScheduler()
    if normalized in {"priority", "cpop_like"}:
        return PriorityScheduler()
    raise ValueError(f"Unsupported scheduler: {name}")


__all__ = ["BaseScheduler", "build_scheduler"]
