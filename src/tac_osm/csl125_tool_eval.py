"""Independent tool-call AST/abstention scorer. No training or model claims.

Malformed output is NOT silently treated as successful abstention. Exact
structured arguments are compared after JSON decoding, never string ordering.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from tac_osm.csl125_tool_protocol import (
    ToolContractError, ToolSpec, validate_tool_proposal
)


@dataclass(frozen=True)
class ToolCaseScore:
    gold_has_call: bool
    predicted_has_call: bool
    valid_prediction: bool
    sequence_exact: bool
    arguments_exact: bool
    full_exact: bool
    abstention_correct: bool
    false_call: bool
    missed_call: bool
    error: str


def score_tool_case(predicted: str, gold: str, tools: Sequence[ToolSpec]) -> ToolCaseScore:
    truth=validate_tool_proposal(gold,tools)
    if type(predicted) is not str:
        return ToolCaseScore(bool(truth.function_calls),False,False,False,False,
            False,False,False,bool(truth.function_calls),"non-string prediction")
    try:
        guess=validate_tool_proposal(predicted,tools)
    except ToolContractError as exc:
        return ToolCaseScore(bool(truth.function_calls),False,False,False,False,
            False,False,False,bool(truth.function_calls),str(exc))
    gold_seq=[c["name"] for c in truth.function_calls]
    guess_seq=[c["name"] for c in guess.function_calls]
    sequence_exact=(gold_seq==guess_seq)
    args_exact=(sequence_exact and
        [c["arguments"] for c in truth.function_calls] ==
        [c["arguments"] for c in guess.function_calls])
    exact=bool(sequence_exact and args_exact)
    abstain=not gold_seq and not guess_seq
    return ToolCaseScore(bool(gold_seq),bool(guess_seq),True,sequence_exact,
                         args_exact,exact,abstain,
                         not bool(gold_seq) and bool(guess_seq),
                         bool(gold_seq) and not bool(guess_seq),"")


def aggregate_tool_scores(scores: Sequence[ToolCaseScore]) -> dict:
    if not scores:
        raise ToolContractError("cannot report empty evaluation cohort")
    n=len(scores)
    eligible=[s for s in scores if s.valid_prediction]
    return {
        "count":n,
        "exact_ast":sum(s.full_exact for s in scores)/n,
        "tool_sequence_exact":sum(s.sequence_exact for s in scores)/n,
        "valid_generation_fraction":len(eligible)/n,
        "invalid_fraction":1-len(eligible)/n,
        "abstention_correct_count":sum(s.abstention_correct for s in scores),
        "false_call_count":sum(s.false_call for s in scores),
        "missed_call_count":sum(s.missed_call for s in scores),
        "gold_no_call_count":sum(not s.gold_has_call for s in scores),
        "gold_call_count":sum(s.gold_has_call for s in scores),
        "false_call_on_no_call_rate":(
            sum(s.false_call for s in scores)/max(1,sum(not s.gold_has_call for s in scores))),
        "invalid_not_counted_as_abstention": True,
    }
