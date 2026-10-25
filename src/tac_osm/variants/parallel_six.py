"""Prospective six-family catalogue for the single parallel TAC information engine.

These family profiles are NOT six independently fitted models. The verified
shared computational graph currently implements a continuous next-observation
pilot, not every task-specific head. See parallel-six/active-variants.json.
Existing six domain adapters are independent non-authoritative interfaces.
"""
from __future__ import annotations

from dataclasses import dataclass
from .registry import Variant, VARIANTS


@dataclass(frozen=True)
class ParallelFamily:
    family: Variant
    shared_engine: str
    execution_modes: tuple[str, ...]
    target_interface: str
    qualified_head: bool = False
    qualified_real_skill: bool = False
    authorized_to_act: bool = False


_ENGINE = "ParallelTACWorld + causal ordered-affine scan + WorldStreamState"
_MODES = ("causal_parallel_batch", "fixed_state_stream")
PARALLEL_FAMILIES = {
    v: ParallelFamily(v, _ENGINE, _MODES, spec.target)
    for v, spec in VARIANTS.items()
}
if len(PARALLEL_FAMILIES) != 6:
    raise AssertionError("exactly six active TAC families required")


def parallel_family(name: str | Variant) -> ParallelFamily:
    try:
        return PARALLEL_FAMILIES[Variant(name)]
    except (ValueError, KeyError) as exc:
        raise ValueError("not one of the six prospective TAC families") from exc


def instantiate_untrained_core(
    name: str | Variant, *, state_dim: int, action_dim: int,
    width: int = 88, decoder: int = 132, candidate: str = "original",
):
    """Construct common neural *research core*, not a fitted family predictor.

    Domains other than genuine continuous next-observation forecasting still
    need distinct typed observation/target encoders, heads and evidence.
    No optimizer may start without separate compulsory real-data/parameter/
    training-compute preflight. This factory does NOT attest to any of them.
    """
    parallel_family(name)
    if any(type(n) is not int or n < 1 for n in
           (state_dim, action_dim, width, decoder)):
        raise ValueError("positive integer dimensions required")
    if candidate == "original":
        from .parallel_continuous_world import ParallelTACWorld as Core
    elif candidate == "evidence_retrieval":
        from .evidence_retrieval_world import EvidenceRetrievalWorld as Core
    elif candidate == "coherent_trajectory":
        from .coherent_trajectory_world import CoherentTrajectoryWorld as Core
    elif candidate == "regime_trajectory":
        from .regime_trajectory_world import RegimeTrajectoryWorld as Core
    else:
        raise ValueError("unknown shared-core candidate")
    return Core(
        state_dim=state_dim, action_dim=action_dim,
        width=width, decoder=decoder,
    )


def require_qualified_domain_head(name: str | Variant) -> None:
    """Fail closed: no consolidated domain-specific head is qualified."""
    p = parallel_family(name)
    if not p.qualified_head or not p.qualified_real_skill:
        raise RuntimeError(
            f"{p.family.value}: domain head/independent real-world skill OPEN"
        )


def instantiate_entity_sensor_core(name: str | Variant, *, candidate: str = "source"):

    """Unqualified typed sensor candidate, within the existing Temporal family."""
    if parallel_family(name).family != Variant.TEMPORAL:
        raise ValueError('entity-sensor candidate belongs to Temporal research lane')
    if candidate == "source":
        from .entity_sensor_world import EntitySensorWorld
        return EntitySensorWorld()
    if candidate == "belief":
        from .belief_sensor_world import BeliefSensorWorld
        return BeliefSensorWorld()
    if candidate == "reconstruction":
        from .reconstruction_sensor_world import ReconstructionSensorWorld
        return ReconstructionSensorWorld()
    if candidate == "protected":
        from .protected_sensor_world import ProtectedSensorWorld
        return ProtectedSensorWorld()
    if candidate == "median_split":
        from .median_split_sensor_world import MedianSplitSensorWorld
        return MedianSplitSensorWorld()
    raise ValueError("unknown typed sensor candidate")
