"""Experimental concept-swarm recurrent CED variant of TAC-OSM.

ENGINEERING ONLY: untrained; no real-world or architectural superiority result.
Persistent observations and REAL, masked outcome verification can change belief.
Candidate-action queries and any number of internal decoder passes CANNOT.
No assumed causal intervention estimates, observed costs or learned policy.
"""
from __future__ import annotations
from typing import NamedTuple
import math
import torch
from torch import nn
from .predictive_state_agent import Belief, PredictiveStateAgent


class ConceptBelief(NamedTuple):
    core: Belief
    concepts: torch.Tensor              # B x N x D; caller routes individual entities


class ConceptPrediction(NamedTuple):
    outcome_logits: torch.Tensor        # B x A x O; factual prediction, NOT do(a)
    next_observation: torch.Tensor      # B x A x context_size
    future_concepts: torch.Tensor       # B x A x N x D; unsupervised until linked events
    policy_scores: torch.Tensor         # B x A; untrained action-ranking logits
    information_scores: torch.Tensor    # B x A; untrained gain predictor
    step_logits: torch.Tensor           # K x B x A x O (coarse -> refined)
    step_deltas: torch.Tensor           # K x B x A (change magnitude, not accuracy)


class ConceptSwarmCED(nn.Module):
    """Native TAC state + persistent concept slots + looped, weight-shared decoder.

    Slot identities are learned, not externally resolved entity identities.
    Decoder loops only modify temporary query tensors. Real records are
    mandatory before training/evaluating prediction or policy capabilities.
    """
    def __init__(self, context_size: int, action_size: int, outcome_size: int = 1,
                 *, concept_count: int = 6, concept_dim: int = 16, refinement_steps: int = 4):
        super().__init__()
        if concept_count < 2 or concept_dim < 2 or not 1 <= refinement_steps <= 16:
            raise ValueError('invalid concept count, dimension or refinement steps')
        self.core = PredictiveStateAgent(context_size, action_size, outcome_size)
        self.concept_count = concept_count
        self.concept_dim = concept_dim
        self.refinement_steps = refinement_steps
        d = concept_dim
        self.anchors = nn.Parameter(torch.randn(concept_count, d) * 0.02)
        self.global_map = nn.Linear(64 + 16, d)
        self.core_map = nn.Linear(64, d)
        self.action_map = nn.Linear(8, d)
        # Learned similarity routing between concept representations, not a Transformer.
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.rel_bias = nn.Parameter(torch.zeros(concept_count, concept_count))
        self.observe_propose = nn.Linear(3*d, d)
        self.observe_gate = nn.Linear(2*d, d)
        self.decode_propose = nn.Linear(4*d, d)
        self.decode_gate = nn.Linear(3*d, d)
        self.pool_score = nn.Linear(d, 1)
        self.outcome_delta = nn.Linear(d, outcome_size)
        self.next_delta = nn.Linear(d, context_size)
        self.future_map = nn.Linear(d, d)
        self.policy_head = nn.Linear(d, 1)
        self.information_head = nn.Linear(d, 1)
        self.verify_delta = nn.Linear(d + outcome_size, d)
        self.verify_gate = nn.Linear(d + outcome_size, d)

    def initial(self, batch_size: int) -> ConceptBelief:
        core = self.core.initial(batch_size)
        return ConceptBelief(core, self.anchors.new_zeros(batch_size, self.concept_count, self.concept_dim))

    def _validate(self, state: ConceptBelief) -> int:
        b = self.core._validate(state.core)
        if state.concepts.shape != (b, self.concept_count, self.concept_dim):
            raise ValueError('concept belief shape mismatch')
        return b

    def _exchange(self, x: torch.Tensor) -> torch.Tensor:
        # Last two axes are N,D; arbitrary leading batch/action axes.
        score = self.q(x) @ self.k(x).transpose(-1, -2) / math.sqrt(self.concept_dim)
        score = score + self.rel_bias
        attention = torch.softmax(score, dim=-1)
        return attention @ self.v(x)

    def observe(self, state: ConceptBelief, context: torch.Tensor) -> ConceptBelief:
        b = self._validate(state)
        next_core = self.core.observe(state.core, context)
        observation = torch.tanh(self.core.observation(context))
        global_context = torch.tanh(self.global_map(torch.cat((next_core.fast, next_core.slow, observation), -1)))
        g = global_context[:, None, :].expand(-1, self.concept_count, -1)
        c = state.concepts
        messages = self._exchange(c + self.anchors)
        proposals = torch.tanh(self.observe_propose(torch.cat((c, messages, g), -1)))
        gate = torch.sigmoid(self.observe_gate(torch.cat((c, g), -1)))
        next_concepts = c + 0.25 * gate * (proposals - c)
        return ConceptBelief(next_core, next_concepts)

    def predict(self, state: ConceptBelief, context: torch.Tensor,
                candidates: torch.Tensor) -> ConceptPrediction:
        b = self._validate(state)
        native_logits, native_next = self.core.predict(state.core, context, candidates)
        a_count = candidates.shape[1]
        action_codes = torch.tanh(self.core.action(candidates))
        action_latents = torch.tanh(self.action_map(action_codes))
        global_latents = torch.tanh(self.core_map(torch.cat((state.core.fast, state.core.slow), -1)))
        g = global_latents[:, None, None, :].expand(b, a_count, self.concept_count, -1)
        a = action_latents[:, :, None, :].expand_as(g)
        q = torch.tanh(state.concepts[:, None, :, :] + self.anchors + a + g)
        logits_by_step = []
        deltas = []
        # Same weights and cached persistent state on every internal pass.
        for _ in range(self.refinement_steps):
            messages = self._exchange(q)
            proposal = torch.tanh(self.decode_propose(torch.cat((q, messages, a, g), -1)))
            gate = torch.sigmoid(self.decode_gate(torch.cat((q, messages, g), -1)))
            change = (0.75 / self.refinement_steps) * gate * (proposal - q)
            q = q + change
            weights = torch.softmax(self.pool_score(q), dim=-2)
            summary = (weights * q).sum(dim=-2)
            logits_by_step.append(native_logits + self.outcome_delta(summary))
            deltas.append(change.detach().abs().mean(dim=(-1, -2)))
        return ConceptPrediction(
            logits_by_step[-1],
            native_next + self.next_delta(summary),
            torch.tanh(self.future_map(q)),
            self.policy_head(summary).squeeze(-1),
            self.information_head(summary).squeeze(-1),
            torch.stack(logits_by_step),
            torch.stack(deltas),
        )

    def verify(self, state: ConceptBelief, executed_action: torch.Tensor,
               outcome: torch.Tensor, predicted_outcome: torch.Tensor,
               available: torch.Tensor) -> ConceptBelief:
        self._validate(state)
        core = self.core.verify(state.core, executed_action, outcome,
                                predicted_outcome, available)
        # Reuse core's finite-value and temporal-mask validation. No operation
        # on unavailable NaNs, and missing outcomes produce bitwise identity.
        observed = torch.where(available, outcome, torch.zeros_like(outcome))
        preceding = torch.where(available, predicted_outcome.detach(), torch.zeros_like(outcome))
        residual = torch.tanh(observed - preceding)
        e = residual[:, None, :].expand(-1, self.concept_count, -1)
        pair = torch.cat((state.concepts, e), dim=-1)
        delta = 0.125 * torch.sigmoid(self.verify_gate(pair)) * torch.tanh(self.verify_delta(pair))
        concepts = state.concepts + torch.where(available.unsqueeze(-1), delta, torch.zeros_like(delta))
        return ConceptBelief(core, concepts)
