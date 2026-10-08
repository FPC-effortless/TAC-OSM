"""Bio-inspired TAC-OSM fusion: distributed local computation + multi-timescale memory.

The fusion package is a parallel experimental surface. It does not replace the
existing TAC-OSM loop; it supplies a compositional model in which every major
mechanism is an injected module with an inspectable trace and a switchable
implementation.

Research synthesis:
- fly nervous-system evidence -> local feedback modules + sparse long-range
  coordination;
- bacterial memory evidence -> multi-timescale internal state + experience
  dependent priming;
- TAC-OSM -> persistent state, selective computation, verification, repair,
  provenance, and strict temporal leakage boundary.
"""

from .interfaces import (
    CandidateScore,
    CoordinatorDecision,
    FusionStep,
    MemoryContext,
    ModuleExecution,
    RegimeContext,
)
from .memory import MemoryConfig, MultiScaleMemory
from .regime import RegimeConfig, RegimeState
from .coordinator import CoordinatorConfig, SparseCoordinator
from .operators import OperatorConfig, SpecialistPool
from .verifier import FusionVerifierConfig, FusionVerifier, RepairPolicy
from .model import FusionModelConfig, FusedTacOsmModel
from .ablation import FusionAblationConfig, all_fusion_matrices

__all__ = [
    "CandidateScore",
    "CoordinatorDecision",
    "FusionStep",
    "MemoryContext",
    "ModuleExecution",
    "RegimeContext",
    "MemoryConfig",
    "MultiScaleMemory",
    "RegimeConfig",
    "RegimeState",
    "CoordinatorConfig",
    "SparseCoordinator",
    "OperatorConfig",
    "SpecialistPool",
    "FusionVerifierConfig",
    "FusionVerifier",
    "RepairPolicy",
    "FusionModelConfig",
    "FusedTacOsmModel",
    "FusionAblationConfig",
    "all_fusion_matrices",
]
