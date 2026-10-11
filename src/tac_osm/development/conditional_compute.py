"""Task-trainable sparse dispatch with actual inference early exit.

Training uses selected-branch straight-through weights. Unselected operators
are not executed, so this is a biased gradient estimator, not soft mixture
equivalence. Inference uses normalized hard weights and removes halted samples
before dispatch. Traces count executed work, never fractional gate activity.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class DispatchTrace:
    depth: int
    active_samples: tuple[int, ...]
    dispatched: tuple[tuple[int, tuple[int, ...]], ...]

    @property
    def operator_sample_calls(self) -> int:
        return sum(len(samples) for _, samples in self.dispatched)


def selected_weights(logits: Tensor, top_k: int, training: bool) -> tuple[Tensor, Tensor]:
    if not 1 <= top_k <= logits.shape[-1]:
        raise ValueError("top_k outside operator bank")
    probs = logits.softmax(-1)
    indices = probs.topk(top_k, dim=-1).indices
    hard = torch.zeros_like(probs).scatter(1, indices, probs.gather(1, indices))
    hard = hard / hard.sum(-1, keepdim=True).clamp_min(1e-8)
    # Do not renormalize this surrogate after adding it: top-1 p/p would
    # eliminate the selector's task-gradient signal again.
    weights = hard.detach() + probs - probs.detach() if training else hard
    return indices, weights


class ConditionalExecutor(nn.Module):
    def __init__(self, dim: int, operators: int = 4, top_k: int = 1,
                 depth: int = 3, halt_threshold: float = 0.8):
        super().__init__()
        if dim < 1 or operators < 1 or depth < 1:
            raise ValueError("dimensions and depth must be positive")
        if not 1 <= top_k <= operators or not 0 < halt_threshold < 1:
            raise ValueError("invalid routing/halting configuration")
        self.top_k, self.depth, self.halt_threshold = top_k, depth, halt_threshold
        self.operators = nn.ModuleList([
            nn.Sequential(nn.Linear(2*dim, 2*dim), nn.GELU(), nn.Linear(2*dim, dim))
            for _ in range(operators)
        ])
        self.selector = nn.Sequential(nn.Linear(2*dim, dim), nn.GELU(),
                                      nn.Linear(dim, operators))
        self.halt = nn.Sequential(nn.Linear(2*dim, dim), nn.GELU(), nn.Linear(dim, 1))
        self.last_trace: tuple[DispatchTrace, ...] = ()

    def forward(self, current: Tensor, query: Tensor) -> tuple[Tensor, Tensor]:
        if current.ndim != 2 or current.shape != query.shape or current.shape[0] < 1:
            raise ValueError("current/query must share nonempty [batch,dim] shape")
        active = torch.ones(current.shape[0], dtype=torch.bool, device=current.device)
        traces, halt_rows = [], []
        for depth in range(self.depth):
            sample_ids = active.nonzero(as_tuple=True)[0]
            if sample_ids.numel() == 0:
                break
            joint = torch.cat((current[sample_ids], query[sample_ids]), dim=-1)
            halt = self.halt(joint).sigmoid().squeeze(-1)
            halt_rows.append(halt)
            if not self.training:
                keep = halt < self.halt_threshold
                active[sample_ids[~keep]] = False
                sample_ids, joint, halt = sample_ids[keep], joint[keep], halt[keep]
            dispatched = []
            if sample_ids.numel():
                indices, weights = selected_weights(self.selector(joint), self.top_k, self.training)
                delta = torch.zeros_like(current[sample_ids])
                for op_id, operator in enumerate(self.operators):
                    local = (indices == op_id).any(dim=1).nonzero(as_tuple=True)[0]
                    if local.numel():
                        # Actual operator call is inside the trace-producing branch.
                        value = operator(joint[local])
                        delta = delta.index_add(0, local, weights[local, op_id, None] * value)
                        dispatched.append((op_id, tuple(sample_ids[local].detach().cpu().tolist())))
                scale = (1-halt[:, None]) if self.training else 1.0
                current = current.index_add(0, sample_ids, scale * delta)
            traces.append(DispatchTrace(depth, tuple(sample_ids.detach().cpu().tolist()),
                                        tuple(dispatched)))
        self.last_trace = tuple(traces)
        # Diagnostic/cost regularizer; actual call counts come from last_trace.
        activity = torch.cat([1-h for h in halt_rows]).mean()
        return current, activity
