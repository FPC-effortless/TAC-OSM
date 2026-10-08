"""Bio-inspired TAC-OSM fusion package.

This package is parallel to the existing TAC-OSM core. It provides one fused
execution loop with independently replaceable modules, explicit traces,
ablation configs, and benchmark tooling.
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
from .builder import BuiltFusionModel, build_fused_model
from .benchmark import FusionMetrics, run_arm, run_suite, run_multi_seed_suite, summarize_metrics, write_results
from .gates import GateResult, all_gates

__all__ = [
    "CandidateScore", "CoordinatorDecision", "FusionStep", "MemoryContext",
    "ModuleExecution", "RegimeContext", "MemoryConfig", "MultiScaleMemory",
    "RegimeConfig", "RegimeState", "CoordinatorConfig", "SparseCoordinator",
    "OperatorConfig", "SpecialistPool", "FusionVerifierConfig",
    "FusionVerifier", "RepairPolicy", "FusionModelConfig", "FusedTacOsmModel",
    "FusionAblationConfig", "all_fusion_matrices", "BuiltFusionModel",
    "build_fused_model", "FusionMetrics", "run_arm", "run_suite", "run_multi_seed_suite",
    "summarize_metrics", "write_results", "GateResult", "all_gates",
]
