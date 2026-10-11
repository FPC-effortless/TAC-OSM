"""Development successor to frozen FULL-LEARNED-SWILA-001.

The reference source is copied byte-for-byte from PR #143. This successor does
not modify that experiment. Streaming state must be explicitly passed between
decisions; accepted writes can receive credit from a subsequent decision.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .conditional_compute import ConditionalExecutor
from .full_plm_reference import FullLearnedPLM, FullPLMConfig, LearnedCASM, SwiLAMTSK


class VerifiedMTSK(SwiLAMTSK):
    def commit_verified(self, current_state, context, action, outcome, verifier_probability):
        if not torch.isfinite(verifier_probability).all():
            raise ValueError("nonfinite verifier probabilities")
        if ((verifier_probability < 0) | (verifier_probability > 1)).any():
            raise ValueError("verifier probabilities outside [0,1]")
        if action.ndim == 1:
            action = torch.nn.functional.one_hot(action.long(), self.config.action_count)
        action = action.to(context.dtype)
        x = self.write_proj(torch.cat((context, action, outcome.view(-1, 1)), dim=-1))
        # Gate the complete transition, including decay. A rejected write is
        # an identity operation; it cannot silently erase existing contents.
        _, proposal, _, _ = self._step(x, current_state, None)
        gate = verifier_probability if self.training else (
            verifier_probability >= self.config.verifier_threshold).to(context.dtype)
        gate = gate.view(-1, 1, 1, 1, 1)
        return current_state + gate * (proposal - current_state)


class DevelopmentCASM(LearnedCASM):
    def __init__(self, config):
        super().__init__(config)
        self.compute = ConditionalExecutor(config.d_model, config.casm_operators,
                                           config.casm_top_k, config.casm_depth,
                                           config.halt_threshold)
        # Preserve the reference interface without duplicate trainable banks.
        del self.operators, self.selector, self.halt

    def forward(self, query, selected, weights):
        graph, graph_query, adjacency = self.workspace(query, selected, weights)
        current = (graph * weights.unsqueeze(-1)).sum(1) + graph_query
        current, activity = self.compute(current, query)
        # Compatibility diagnostics; these are not reported as skipped work.
        empty = current.new_empty((current.shape[0], 0))
        return current, empty, empty, current[:, None], adjacency, activity.expand(current.shape[0], 1)


@dataclass(frozen=True)
class StreamingState:
    memory: Tensor
    responsibility: Tensor | None


class DevelopmentPLM(FullLearnedPLM):
    def __init__(self, config: FullPLMConfig = FullPLMConfig()):
        super().__init__(config)
        self.mtsk = VerifiedMTSK(config)
        self.casm = DevelopmentCASM(config)

    def observe(self, text: Tensor, image: Tensor, audio: Tensor,
                state: StreamingState | None = None) -> tuple[StreamingState, Tensor, Tensor]:
        """Consume one fresh observation; never replay the historical prefix."""
        z = self.representation(text, image, audio)
        memory = self.mtsk.initial_state(z.shape[0], z.device, z.dtype) if state is None else state.memory
        previous = None if state is None else state.responsibility
        read, memory, experts, responsibility = self.mtsk._step(z, memory, previous)
        return StreamingState(memory, responsibility), read, experts

    def decide(self, read: Tensor, experts: Tensor) -> Tensor:
        selected, weights, _ = self.cdl(read, experts)
        computation, *_ = self.casm(read, selected, weights)
        # Development streaming action path; the frozen reference planner is
        # deliberately not attributed to this bounded streaming experiment.
        return self.action_head(torch.cat((computation, read), dim=-1))

    def commit(self, state: StreamingState, context: Tensor, action: Tensor,
               evidence: Tensor, probability: Tensor) -> StreamingState:
        memory = self.mtsk.commit_verified(state.memory, context, action, evidence, probability)
        return StreamingState(memory, state.responsibility)
