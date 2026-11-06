"""Proposal-only tool action planning; it CANNOT execute anything.

Final execution authority belongs to an authenticated, separately implemented
host with its own permission, confirmation, audit and effect-reconciliation
checks. Model output, R3 error logits and schema validity grant no permission.
"""
from dataclasses import dataclass
from typing import Sequence

from tac_osm.csl125_tool_protocol import ToolContractError, ToolProposal, ToolSpec


@dataclass(frozen=True)
class PlannedCall:
    index: int
    name: str
    effect: str
    status: str
    reason: str


def review_proposal(proposal: ToolProposal,
                    tools: Sequence[ToolSpec],
                    *, permitted_tool_names: frozenset[str],
                    confirmed_call_indices: frozenset[int] = frozenset()
                    ) -> tuple[PlannedCall, ...]:
    if proposal.status != "proposed":
        return ()
    if type(permitted_tool_names) is not frozenset or type(confirmed_call_indices) is not frozenset:
        raise ToolContractError("permissions/confirmations must be explicit host-provided sets")
    known={s.name:s for s in tools}
    result=[]
    for idx, call in enumerate(proposal.function_calls):
        spec=known.get(call["name"])
        if spec is None:
            raise ToolContractError("unknown tool in reviewed proposal")
        if spec.name not in permitted_tool_names:
            status,reason="denied","host allowlist excludes tool"
        elif spec.effect != "read" and idx not in confirmed_call_indices:
            status,reason="confirmation_required","write/high-impact effects require host confirmation"
        else:
            status,reason="candidate_only","independent host must still authenticate and validate execution"
        result.append(PlannedCall(idx,spec.name,spec.effect,status,reason))
    return tuple(result)
