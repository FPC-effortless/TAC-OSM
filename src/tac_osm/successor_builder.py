"""Single construction path for TAC-OSM Successor Architecture v1."""

from __future__ import annotations

from dataclasses import dataclass

from .energy_router import EnergyRouterConfig, RepresentationEnergyRouter
from .environment import WorldConfig, WorldEnvironment
from .explicit_executor import ExplicitExecutorConfig, ExplicitGraphExecutor
from .model import ModelConfig, TacOsmModel
from .state import PersistentStore, StateConfig
from .verifier import ThresholdVerifier, VerifierConfig


@dataclass(frozen=True)
class SuccessorConfig:
    dim: int = 8
    max_nodes: int = 16
    n_state_slots: int = 64
    n_steps: int = 12
    seed: int = 0
    learn: bool = True
    latent_dim: int = 16
    learning_rate: float = 0.01
    margin: float = 0.1
    top_k: int = 1
    executor_mode: str = "exact"
    verifier_type: str = "path"
    repair: bool = True
    write_on_success: bool = True

    def __post_init__(self) -> None:
        if self.dim < 4:
            raise ValueError("dim must be >= 4")
        if self.max_nodes < 4:
            raise ValueError("max_nodes must be >= 4")
        if self.n_state_slots < 1:
            raise ValueError("n_state_slots must be >= 1")
        if self.n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        if self.latent_dim < 1:
            raise ValueError("latent_dim must be positive")
        if self.top_k != 1:
            raise ValueError(
                "v1 model integration exposes one selected candidate; "
                "multi-candidate execution needs a separate budget contract"
            )


def build_successor(config: SuccessorConfig | None = None) -> TacOsmModel:
    """Build the successor loop with explicit addressing/routing/execution."""
    cfg = config if config is not None else SuccessorConfig()
    state = PersistentStore(
        StateConfig(
            enabled=True,
            write=True,
            intervention="persistent",
            seed=cfg.seed,
            n_slots=cfg.n_state_slots,
        )
    )
    router = RepresentationEnergyRouter(
        EnergyRouterConfig(
            input_dim=cfg.dim,
            latent_dim=cfg.latent_dim,
            learning_rate=cfg.learning_rate,
            margin=cfg.margin,
            top_k=cfg.top_k,
            seed=cfg.seed,
        )
    )
    executor = ExplicitGraphExecutor(
        ExplicitExecutorConfig(
            mode=cfg.executor_mode,
            max_nodes=cfg.max_nodes,
        )
    )
    verifier_config = VerifierConfig(type=cfg.verifier_type, repair=cfg.repair)
    verifier = ThresholdVerifier(verifier_config)
    from .verifier import BoundedRepairController
    repair = BoundedRepairController(verifier_config) if cfg.repair else None
    environment = WorldEnvironment(WorldConfig(seed=cfg.seed))

    return TacOsmModel(
        state=state,
        router=router,
        executor=executor,
        verifier=verifier,
        repair=repair,
        environment=environment,
        config=ModelConfig(
            n_steps=cfg.n_steps,
            learn=cfg.learn,
            seed=cfg.seed,
            write_on_success=cfg.write_on_success,
        ),
    )


__all__ = ["SuccessorConfig", "build_successor"]
