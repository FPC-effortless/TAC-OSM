"""The ablation harness — one architecture, controlled switches.

This is the primary scientific instrument of TAC-OSM. Every variant in
``docs/ABLATION_PLAN.md`` is produced by instantiating this config and
flipping fields. There is deliberately **no per-experiment code path**: a
comparison is only interpretable if the arms differ only in the switch
values, never in the code that runs.

Config keys are the switch surface from §7 of the ablation plan:

    state.enabled:       true | false
    state.intervention:  persistent | shuffled | reset | corrupted | random | wrong_key
    state.write:         true | false
    router.type:         learned | static | random | oracle | full_context
    structure.type:      learned | random | oracle | flattened
    verifier.type:       none | final | path
    verifier.repair:     true | false
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

StateIntervention = Literal[
    "persistent", "shuffled", "reset", "corrupted", "random", "wrong_key"
]
RouterType = Literal["learned", "static", "random", "oracle", "full_context"]
StructureType = Literal["learned", "random", "oracle", "flattened"]
VerifierType = Literal["none", "final", "path"]


@dataclass
class StateSwitch:
    enabled: bool = True
    write: bool = True
    intervention: StateIntervention = "persistent"
    corruption_rate: float = 0.0
    seed: int = 0


@dataclass
class RouterSwitch:
    type: RouterType = "learned"
    temperature: float = 0.5


@dataclass
class StructureSwitch:
    type: StructureType = "learned"


@dataclass
class VerifierSwitch:
    type: VerifierType = "path"
    repair: bool = True
    max_attempts: int = 3


@dataclass
class HistorySwitch:
    """Irrelevant-history levels for the context-scaling test.

    Levels and parity margin must be frozen **before** the confirmatory run.
    Choosing either after seeing results invalidates the scaling claim.
    """

    levels: tuple[int, ...] = (1000, 4000, 16000, 32000, 64000, 100000)
    parity_margin: float | None = None


@dataclass
class AblationConfig:
    """One cell of the ablation matrix."""

    name: str = "tac_osm_v0"
    seed: int = 0
    state: StateSwitch = field(default_factory=StateSwitch)
    router: RouterSwitch = field(default_factory=RouterSwitch)
    structure: StructureSwitch = field(default_factory=StructureSwitch)
    verifier: VerifierSwitch = field(default_factory=VerifierSwitch)
    history: HistorySwitch = field(default_factory=HistorySwitch)

    # Provenance is part of the config, not an afterthought: every result must
    # identify repository -> branch -> commit -> benchmark version ->
    # experiment id -> configuration -> seed -> artifact -> metric.
    provenance: dict[str, Any] = field(default_factory=dict)

    def label(self) -> str:
        """Compact machine-friendly name for this ablation cell."""
        parts = [self.name]
        if not self.state.enabled:
            parts.append("no_state")
        elif self.state.intervention != "persistent":
            parts.append(f"state_{self.state.intervention}")
        if not self.state.write:
            parts.append("no_write")
        if self.router.type != "learned":
            parts.append(f"router_{self.router.type}")
        if self.structure.type != "learned":
            parts.append(f"structure_{self.structure.type}")
        if self.verifier.type != "path":
            parts.append(f"verifier_{self.verifier.type}")
        if not self.verifier.repair:
            parts.append("no_repair")
        return "-".join(parts) if len(parts) > 1 else self.name

    def describe(self) -> dict[str, Any]:
        """Full switch state, for the results ledger."""
        return {
            "name": self.name,
            "label": self.label(),
            "seed": self.seed,
            "state": vars(self.state),
            "router": vars(self.router),
            "structure": vars(self.structure),
            "verifier": vars(self.verifier),
            "history": vars(self.history),
            "provenance": self.provenance,
        }


# --------------------------------------------------------------------------- #
# The canonical matrix
# --------------------------------------------------------------------------- #


def component_matrix(seed: int = 0) -> list[AblationConfig]:
    """The component ablation matrix from §4 of the ablation plan.

    Each row removes exactly one mechanism from the full model, so the
    difference between adjacent rows isolates that mechanism's contribution.
    """
    full = AblationConfig(name="full", seed=seed)
    return [
        _with(full, name="baseline", state=StateSwitch(enabled=False, write=False)),
        _with(full, name="plus_state", router=RouterSwitch(type="random")),
        _with(full, name="plus_routing", structure=StructureSwitch(type="random")),
        _with(
            full,
            name="plus_structure",
            verifier=VerifierSwitch(type="none", repair=False),
        ),
        _with(full, name="plus_verifier", verifier=VerifierSwitch(repair=False)),
        full,
    ]


def persistence_interventions(seed: int = 0) -> list[AblationConfig]:
    """Mechanism-specific persistence interventions, §5.1."""
    return [
        _with(
            AblationConfig(seed=seed),
            name=f"state_{iv}",
            state=StateSwitch(intervention=iv, corruption_rate=rate),
        )
        for iv, rate in [
            ("persistent", 0.0),
            ("shuffled", 0.0),
            ("reset", 0.0),
            ("corrupted", 0.25),
            ("corrupted", 0.5),
            ("corrupted", 1.0),
            ("random", 0.0),
            ("wrong_key", 0.0),
        ]
    ]


def routing_arms(seed: int = 0) -> list[AblationConfig]:
    """Router variants, §5.2. The static arm is the one that matters.

    Stage 2b showed a fixed similarity scorer alone can capture most of the
    available headroom (~71% at 8 candidates) for free. A routing result that
    does not beat static is not a routing result.
    """
    return [
        _with(AblationConfig(seed=seed), name=f"router_{t}", router=RouterSwitch(type=t))
        for t in ("learned", "static", "random", "oracle", "full_context")
    ]


def structure_arms(seed: int = 0) -> list[AblationConfig]:
    """Structural-execution variants, §5.3."""
    return [
        _with(
            AblationConfig(seed=seed),
            name=f"structure_{t}",
            structure=StructureSwitch(type=t),
        )
        for t in ("learned", "random", "oracle", "flattened")
    ]


def verification_arms(seed: int = 0) -> list[AblationConfig]:
    """Verification variants, §5.4.

    The final-vs-path comparison is genuinely open: the master benchmark §19
    requires it and no source repository has run it.
    """
    return [
        _with(
            AblationConfig(seed=seed),
            name="verifier_none",
            verifier=VerifierSwitch(type="none", repair=False),
        ),
        _with(
            AblationConfig(seed=seed),
            name="verifier_final",
            verifier=VerifierSwitch(type="final", repair=False),
        ),
        _with(
            AblationConfig(seed=seed),
            name="verifier_path",
            verifier=VerifierSwitch(type="path", repair=False),
        ),
        _with(
            AblationConfig(seed=seed),
            name="verifier_path_repair",
            verifier=VerifierSwitch(type="path", repair=True),
        ),
    ]


def all_matrices(seed: int = 0) -> dict[str, list[AblationConfig]]:
    """Every pre-registered matrix, keyed by name."""
    return {
        "component": component_matrix(seed),
        "persistence": persistence_interventions(seed),
        "routing": routing_arms(seed),
        "structure": structure_arms(seed),
        "verification": verification_arms(seed),
    }


def _with(
    base: AblationConfig,
    *,
    name: str,
    state: StateSwitch | None = None,
    router: RouterSwitch | None = None,
    structure: StructureSwitch | None = None,
    verifier: VerifierSwitch | None = None,
) -> AblationConfig:
    return AblationConfig(
        name=name,
        seed=base.seed,
        state=state if state is not None else StateSwitch(**vars(base.state)),
        router=router if router is not None else RouterSwitch(**vars(base.router)),
        structure=(
            structure if structure is not None else StructureSwitch(**vars(base.structure))
        ),
        verifier=(
            verifier if verifier is not None else VerifierSwitch(**vars(base.verifier))
        ),
        history=HistorySwitch(**vars(base.history)),
        provenance=dict(base.provenance),
    )


def freeze(config: AblationConfig, *, commit: str, experiment_id: str) -> AblationConfig:
    """Attach immutable provenance before a run.

    Call this once, immediately before training, so the config in the results
    ledger is the config that produced the result.
    """
    config.provenance = {
        "repository": "TAC-OSM",
        "commit": commit,
        "experiment_id": experiment_id,
        "config": config.describe(),
    }
    return config


__all__ = [
    "AblationConfig",
    "StateSwitch",
    "RouterSwitch",
    "StructureSwitch",
    "VerifierSwitch",
    "HistorySwitch",
    "component_matrix",
    "persistence_interventions",
    "routing_arms",
    "structure_arms",
    "verification_arms",
    "all_matrices",
    "freeze",
]
