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
    width: int = 88, decoder: int = 132,
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
    from .parallel_continuous_world import ParallelTACWorld
    return ParallelTACWorld(
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


def instantiate_untrained_postscan_core(
    name: str | Variant, *, state_dim: int, action_dim: int,
    width: int = 88, decoder: int = 132, ffn_hidden: int = 12,
):
    """Opt-in shared parallel state core with bounded post-scan FFN.

    This is still one of the SIX prospective families, never a seventh
    architecture family or a qualified neural model. It has no fitted
    checkpoints or certified domain-specific head. Separate authentic
    source, equal-resource comparator and replay gates remain mandatory.
    """
    parallel_family(name)
    from .parallel_residual_world import ParallelResidualWorld
    return ParallelResidualWorld(
        state_dim=state_dim, action_dim=action_dim,
        width=width, decoder=decoder, ffn_hidden=ffn_hidden,
    )
