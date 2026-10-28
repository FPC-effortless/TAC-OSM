"""LANGLAYER034: experimental layered, reset-window causal language TAC.

Untrained architectural candidate stacked on unchanged CONTEXT026–028.
Layer 1 is the complete original TAC block, including contextual native
memory, aligned memories, waves, six concept seeds and exact CED solve.
Layer 2+ are independent, causal recurrent/concept residual blocks inserted
between the existing LayerNorm and language SwiGLU. The original language
head is narrowed to maintain a ~500K ACTIVE parameter research budget.

This is not DeepSeek's attention-based Causal Encoder–Decoder / CSA2, nor a
streaming-state or performance-qualified model. All source windows reset.
"""
from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

_CONTEXT026 = Path(__file__).resolve().parents[1] / "context026"
if str(_CONTEXT026) not in sys.path:
    sys.path.insert(0, str(_CONTEXT026))
from context_transition import ContextTransition026, scalar_affine_scan  # noqa: E402


class CausalResidualTACBlock(nn.Module):
    """Independent affine-scan memory and six-slot graph refinement.

    h_t = a_t * h_(t-1) + (1-a_t)*v_t with 0 < a_t < .995.
    All schedules depend on current input features only; finite windows
    have inclusive associative scans and no future-label reads. The
    concept posterior solves an SPD 6x6 system in exact-real arithmetic.
    """

    def __init__(self, width: int = 280, state_width: int = 48):
        super().__init__()
        if width <= 0 or state_width <= 0:
            raise ValueError("positive layer and state widths required")
        self.width = width
        self.state_width = state_width
        self.input_norm = nn.LayerNorm(width)
        self.input_proj = nn.Linear(width, state_width)
        self.forget = nn.Linear(state_width, state_width)
        self.offer = nn.Linear(state_width, state_width)
        self.read_gate = nn.Linear(state_width * 2, state_width)
        self.concept_gate = nn.Linear(state_width, 96)
        self.concept_offer = nn.Linear(state_width, 96)
        self.concept_q = nn.Linear(16, 16, bias=False)
        self.concept_k = nn.Linear(16, 16, bias=False)
        self.concept_to_state = nn.Linear(16, state_width)
        self.output = nn.Linear(state_width, width)
        self.residual_gain = nn.Parameter(torch.tensor(-2.0))

    def forward(self, z: torch.Tensor, *, return_trace: bool = False):
        if z.ndim != 3 or z.shape[-1] != self.width or z.shape[1] == 0:
            raise ValueError("expected [batch, nonzero_time, width]")
        b, t, _ = z.shape
        u = torch.tanh(self.input_proj(self.input_norm(z)))
        alpha = 0.995 * self.forget(u).sigmoid()
        offer = self.offer(u).tanh()
        # Causal, inclusive input-only scan. The start state is exactly zero.
        state = scalar_affine_scan(alpha, (1.0 - alpha) * offer)
        prior_state = F.pad(state[:, :-1], (0, 0, 1, 0))
        gate = self.read_gate(torch.cat((u, prior_state), dim=-1)).sigmoid()

        # Each layer has its OWN persistent concept seed/graph parameters.
        cg = 0.05 * self.concept_gate(u).sigmoid()
        cp = scalar_affine_scan(1 - cg, cg * self.concept_offer(u).tanh())
        previous_concepts = F.pad(cp[:, :-1], (0, 0, 1, 0)).reshape(b, t, 6, 16)
        G = self.concept_offer(u).tanh().reshape(b, t, 6, 16) + previous_concepts
        q = self.concept_q(G)
        k = self.concept_k(G)
        sm = (q @ k.transpose(-1, -2) / 4).softmax(dim=-1)
        A = 0.5 * (sm + sm.transpose(-1, -2))
        eye6 = torch.eye(6, dtype=z.dtype, device=z.device)
        A = A * (1 - eye6)
        L = torch.diag_embed(A.sum(dim=-1)) - A
        # H is strictly positive definite over real arithmetic: I*1.25
        # plus .25 times a symmetric nonnegative graph Laplacian.
        C = torch.linalg.solve(1.25 * eye6 + 0.25 * L, G + 0.25 * previous_concepts)
        pooled = (C * G.mean(dim=-1).softmax(dim=-1).unsqueeze(-1)).sum(dim=-2)
        combined = (u + gate * prior_state + self.concept_to_state(pooled)).tanh()
        gain = self.residual_gain.sigmoid()
        out = z + gain * self.output(combined)
        if return_trace:
            return out, {
                "state": state, "prior_state": prior_state,
                "concept_prewrite": previous_concepts, "concept_posterior": C,
                "concept_laplacian": L, "gain": gain,
            }
        return out


class LayeredContextTAC034(ContextTransition026):
    """2- or 3-layer TAC with one original source-processing block.

    The base forward, including source-admission masks and the unchanged
    contextual native read/write semantics, is inherited without copying.
    The parent calls self.layernorm(z) BEFORE its learned SwiGLU, which
    now composes LayerNorm and independent recurrent residual blocks.
    Active weights in the narrowed final SwiGLU retain real gradient paths.
    """

    def __init__(self, vocab: int = 256, depth: int = 2):
        if depth not in (2, 3):
            raise ValueError("only preregistered depths 2 or 3 are supported")
        # One narrow *shared* language head across the full stack.
        ffn_width = {2: 442, 3: 446}[depth]
        super().__init__(vocab=vocab, ffn_width=ffn_width)
        hidden = 48 if depth == 2 else 24
        self.layernorm = nn.Sequential(
            nn.LayerNorm(280),
            *(CausalResidualTACBlock(280, hidden) for _ in range(depth - 1)),
        )
        self.depth = depth
        self.layer_state_width = hidden
        self.ffn_width = ffn_width

    def architecture_record(self) -> dict:
        return {
            "family": "LANGLAYER034", "depth": self.depth,
            "per_residual_state_width": self.layer_state_width,
            "ffn_width": self.ffn_width,
            "trainable_parameters": self.parameter_count(),
            "deployed_learned_parameters": sum(p.numel() for p in self.parameters()),
            "streaming": False, "trained": False,
            "protected_test_evaluated": False,
        }
