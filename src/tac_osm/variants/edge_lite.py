"""Frozen-model edge export *audit* contracts; no quantized runtime is shipped.

INT8 byte reductions do NOT by themselves change neural parameter counts,
and proposed 10k-15k newly learned models violate current 95k-105k policy.
"""
from __future__ import annotations
from dataclasses import dataclass
from .contracts import ForecastEvidence

@dataclass(frozen=True)
class EdgeExportCandidate:
    evidence: ForecastEvidence
    trained_parameter_count: int
    deployed_learned_parameter_count: int
    format: str
    exported_bytes: int
    measured_device_latency_ms: float | None = None
    measured_hardware: str | None = None
    def __post_init__(self):
        if not 95_000 <= self.trained_parameter_count <= 105_000 or not (
            95_000 <= self.deployed_learned_parameter_count <= 105_000):
            raise ValueError("prospective 100k model-scale policy prohibits unapproved tiny learned deployment")
        if self.format not in ("INT8", "FP16", "FP32") or self.exported_bytes <= 0:
            raise ValueError("unknown export representation or byte count")
        if self.measured_device_latency_ms is not None and (
                self.measured_device_latency_ms < 0 or not self.measured_hardware):
            raise ValueError("measured latency requires real identified hardware")
    @property
    def qualified_under_50ms(self) -> bool:
        # A single measurement cannot demonstrate an SLA, even if fast.
        return False
