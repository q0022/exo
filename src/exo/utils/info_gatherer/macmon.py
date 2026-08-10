from typing import Self

from pydantic import BaseModel

from exo.shared.types.profiling import MemoryUsage, SystemPerformanceProfile
from exo.utils.pydantic_ext import TaggedModel


class _TempMetrics(BaseModel, extra="ignore"):
    """Temperature-related metrics returned by macmon."""

    cpu_temp_avg: float
    gpu_temp_avg: float


class _MemoryMetrics(BaseModel, extra="ignore"):
    """Memory-related metrics returned by macmon."""

    ram_total: int
    ram_usage: int
    swap_total: int
    swap_usage: int


class RawMacmonMetrics(BaseModel, extra="ignore"):
    """Complete set of metrics returned by macmon.

    Unknown fields are ignored for forward-compatibility.
    """

    timestamp: str  # ignored
    temp: _TempMetrics
    memory: _MemoryMetrics
    ecpu_usage: tuple[int, float]  # freq mhz, usage %
    pcpu_usage: tuple[int, float]  # freq mhz, usage %
    gpu_usage: tuple[int, float] = (0, 0.0)  # freq mhz, usage %
    gpu_active_ratio: float = 0.0
    cpu_usage_pct: float = 0.0
    all_power: float
    ane_power: float
    cpu_power: float
    gpu_power: float
    gpu_ram_power: float
    ram_power: float
    sys_power: float


class MacmonMetrics(TaggedModel):
    system_profile: SystemPerformanceProfile
    memory: MemoryUsage

    @classmethod
    def from_raw(cls, raw: RawMacmonMetrics) -> Self:
        gpu_pct = raw.gpu_active_ratio if raw.gpu_active_ratio > 0 else raw.gpu_usage[1]
        cpu_pct = raw.cpu_usage_pct if raw.cpu_usage_pct > 0 else raw.pcpu_usage[1]
        return cls(
            system_profile=SystemPerformanceProfile(
                gpu_usage=gpu_pct,
                temp=raw.temp.gpu_temp_avg,
                sys_power=raw.sys_power,
                pcpu_usage=cpu_pct,
                ecpu_usage=raw.ecpu_usage[1],
            ),
            memory=MemoryUsage.from_bytes(
                ram_total=raw.memory.ram_total,
                ram_available=(raw.memory.ram_total - raw.memory.ram_usage),
                swap_total=raw.memory.swap_total,
                swap_available=(raw.memory.swap_total - raw.memory.swap_usage),
            ),
        )

    @classmethod
    def from_raw_json(cls, json: str) -> Self:
        return cls.from_raw(RawMacmonMetrics.model_validate_json(json))
