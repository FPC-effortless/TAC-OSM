"""Build one fused TAC-OSM architecture from a frozen ablation config.

There is deliberately one constructor path. Changing an experiment arm changes
only configuration passed to the same modules; it never selects a special model
implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..environment import WorldConfig, WorldEnvironment
from ..state import PersistentStore, StateConfig
from .ablation import FusionAblationConfig
from .coordinator import CoordinatorConfig, SparseCoordinator
from .memory import MemoryConfig, MultiScaleMemory
from .model import FusedTacOsmModel, FusionModelConfig
from .operators import OperatorConfig, SpecialistPool
from .regime import RegimeConfig, RegimeState
from .verifier import FusionVerifier, FusionVerifierConfig, RepairPolicy


@dataclass(frozen=True)
class BuiltFusionModel:
    model: FusedTacOsmModel
    config: FusionAblationConfig
    manifest: dict[str, Any]


def build_fused_model(
    config: FusionAblationConfig,
    *,
    n_steps: int = 24,
    world_config: WorldConfig | None = None,
) -> BuiltFusionModel:
    """Construct a complete fused model with all components independently injected."""
    world_cfg = world_config if world_config is not None else WorldConfig(seed=config.seed)
    state = PersistentStore(
        StateConfig(
            enabled=config.fact_enabled,
            write=config.fact_write,
            intervention=config.fact_intervention,
            corruption_rate=config.fact_corruption_rate,
            seed=config.seed,
        )
    )
    environment = WorldEnvironment(world_cfg)

    decays = (0.20, 0.60, 0.90, 0.985)
    scales = config.memory_timescales
    if not 1 <= scales <= len(decays):
        raise ValueError(
            f"memory_timescales={scales} exceeds available registered "
            f"timescales 1..{len(decays)}"
        )

    memory_cfg = MemoryConfig(
        enabled=config.memory_enabled,
        n_timescales=scales,
        decays=decays[:scales],
        dim=world_cfg.dim,
        n_modules=8,
        local_enabled=config.local_memory,
        global_enabled=config.global_memory,
        priming_enabled=config.priming,
        intervention=config.memory_intervention,
        corruption_rate=config.memory_corruption_rate,
        seed=config.seed,
    )
    regime_cfg = RegimeConfig(enabled=True, feature_dim=19, seed=config.seed)
    coordinator_cfg = CoordinatorConfig(
        n_modules=8,
        top_k=min(config.top_k_modules, 8),
        mode=config.coordinator_mode,
        dim=world_cfg.dim,
        disabled_modules=config.disabled_modules,
        seed=config.seed,
    )
    operator_cfg = OperatorConfig(
        n_modules=8,
        dim=world_cfg.dim,
        shared_parameters=config.shared_operators,
        adaptive_halting=config.adaptive_halting,
        seed=config.seed + 17,
        computation_mode="nonlinear",
    )
    verifier_cfg = FusionVerifierConfig(
        mode=config.verifier_mode,
        require_positive_outcome_for_commit=config.verifier_commit_success_only,
        repair_attempts=2,
    )
    model_cfg = FusionModelConfig(
        n_steps=n_steps,
        seed=config.seed,
        learning=True,
        experience_write=True,
        dim=world_cfg.dim,
        n_modules=8,
        memory=memory_cfg,
        regime=regime_cfg,
        coordinator=coordinator_cfg,
        operators=operator_cfg,
        verifier=verifier_cfg,
        repair_enabled=config.repair,
    )

    model = FusedTacOsmModel(
        state=state,
        environment=environment,
        config=model_cfg,
        memory=MultiScaleMemory(memory_cfg),
        regime=RegimeState(regime_cfg),
        coordinator=SparseCoordinator(coordinator_cfg),
        operators=SpecialistPool(operator_cfg),
        verifier=FusionVerifier(verifier_cfg),
        repair=RepairPolicy(enabled=config.repair, max_attempts=2),
    )
    manifest = {
        "model": "tacosm_fused_bioinspired_2026-10-08",
        "configuration": config.describe(),
        "world": {
            "dim": world_cfg.dim,
            "n_candidates": world_cfg.n_candidates,
            "families": world_cfg.families,
            "seed": world_cfg.seed,
        },
        "modules": {
            "state": type(state).__name__,
            "memory": type(model.memory).__name__,
            "regime": type(model.regime).__name__,
            "coordinator": type(model.coordinator).__name__,
            "operators": type(model.operators).__name__,
            "verifier": type(model.verifier).__name__,
            "repair": type(model.repair).__name__,
        },
    }
    return BuiltFusionModel(model=model, config=config, manifest=manifest)
