"""Leakage audit for router interfaces — the anti-leakage boundary.

Implements gate G1 of ``docs/ABLATION_PLAN.md`` and the anti-leakage
boundary of ``PNDS_MASTER_BENCHMARK.md`` §7.

An ablation result is only interpretable if the router provably cannot see
the information it is being asked to select for. This is checked
structurally, at construction time, rather than assumed.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any, Sequence

# The forbidden set. A router may use only permitted query, observation,
# state, and candidate properties.
FORBIDDEN_FIELDS = frozenset(
    {
        "target",
        "answer",
        "outcome",
        "reward",
        "true_edge",
        "true_edges",
        "gold",
        "gold_index",
        "gold_structure_id",
        "oracle",
        "oracle_action",
        "oracle_mask",
        "correct_action",
        "label",
        "labels",
        "solution",
        "expected",
        "expected_action",
    }
)

# Field names that *look* like they encode the answer even if a dataclass
# author chose a different name. Substring scan over the rendered attribute
# set, so a renamed leak is still caught.
FORBIDDEN_SUBSTRINGS = ("gold", "oracle", "answer", "true_edge", "correct_action")


def audit_object(obj: Any) -> list[str]:
    """Return the forbidden attributes present on ``obj``."""
    hits: list[str] = []
    names = {f.name for f in fields(obj)} if _is_dataclass(obj) else set(dir(obj))
    for name in names:
        if name.startswith("_"):
            continue
        if name in FORBIDDEN_FIELDS:
            hits.append(name)
            continue
        lowered = name.lower()
        if any(s in lowered for s in FORBIDDEN_SUBSTRINGS):
            hits.append(name)
    return hits


def audit_router_inputs(
    *,
    query: Any,
    candidates: Sequence[Any],
    state: Any | None = None,
) -> dict:
    """Assert the router's observable inputs exclude the forbidden set.

    Returns a report; raises on a violation so the failure is a test failure
    rather than a silent contamination.
    """
    report: dict[str, list[str]] = {}

    hits = audit_object(query)
    if hits:
        report["query"] = hits
    for index, candidate in enumerate(candidates):
        hits = audit_object(candidate)
        if hits:
            report[f"candidate[{index}]"] = hits
    if state is not None:
        hits = audit_object(state)
        if hits:
            report["state"] = hits

    if report:
        raise AssertionError(
            "router interface leakage detected: forbidden attributes "
            f"reachable from router inputs: {report}"
        )
    return {"pass": True, "objects": 2 + len(candidates) + (1 if state is not None else 0)}


def _is_dataclass(obj: Any) -> bool:
    return hasattr(obj, "__dataclass_fields__")


__all__ = ["audit_router_inputs", "audit_object", "FORBIDDEN_FIELDS"]
