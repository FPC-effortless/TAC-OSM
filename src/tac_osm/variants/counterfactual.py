"""Sandbox scenario proposals with explicit associational/simulation provenance.

No intervention identification, optimizer training, MCTS production policy,
authorization or real-world counterfactual skill is inferred from a rollout.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Mapping

@dataclass(frozen=True)
class ScenarioEstimate:
    action_name: str
    predicted: Mapping[str, Any]
    evidence_kind: str
    causal_effect_established: bool = False
    may_execute: bool = False
    def __post_init__(self):
        if not self.action_name or self.evidence_kind not in (
                "observational_conditioning", "simulated_diagnostic"):
            raise ValueError("unsupported causal scenario evidence kind")
        if self.causal_effect_established or self.may_execute:
            raise ValueError("unverified scenarios cannot authorize action or imply causal effect")

class ScenarioAdvisor:
    def __init__(self, evaluator: Callable[[Mapping[str, Any], str], Mapping[str, Any]],
                 *, evidence_kind: str = "observational_conditioning"):
        if not callable(evaluator) or evidence_kind not in (
                "observational_conditioning", "simulated_diagnostic"):
            raise ValueError("scenario evaluator and factual provenance required")
        self.evaluator, self.evidence_kind = evaluator, evidence_kind

    def assess(self, observed_state: Mapping[str, Any],
               available_actions: list[str] | tuple[str, ...]) -> tuple[ScenarioEstimate, ...]:
        if not isinstance(observed_state, Mapping):
            raise ValueError("observed state must be mapping")
        if (not available_actions or len(set(available_actions)) != len(available_actions)
                or any(not isinstance(a, str) or not a for a in available_actions)):
            raise ValueError("nonempty distinct descriptive actions required")
        result = []
        for action in available_actions:
            prediction = self.evaluator(dict(observed_state), action)
            if not isinstance(prediction, Mapping):
                raise ValueError("scenario estimates must be mappings")
            result.append(ScenarioEstimate(action, dict(prediction), self.evidence_kind))
        return tuple(result)
