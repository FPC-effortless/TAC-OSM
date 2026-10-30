"""LANG10M001: no-fit, 4,096-token TAC architecture-scale candidate.

Preserves the original CONTEXT026 associative/wave/concept/fast/slow first
stage. Replaces the one upper recurrence of LANGSCALE070 with four residual
recurrent + SwiGLU blocks. Retains the original causal source boundary.

This candidate is WINDOW RESET ONLY. It makes NO streaming, device-speed,
language-quality, trained-model, or mathematical-certification claim.
"""
from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

MATH_ROOT = Path(__file__).resolve().parents[2] / "mathematical-engineering"
for part in ("context026", "langlayer038"):
    location = str(MATH_ROOT / part)
    if location not in sys.path:
        sys.path.insert(0, location)

from context_transition import ContextTransition026
from recurrence_only import RecurrenceOnlyUpperBlock

VOCAB = 4096
WIDTH = 280
UPPER_BLOCKS = 4
UPPER_STATE = 128
UPPER_FFN = 1024
OUTPUT_FFN = 1214
EXPECTED_PARAMETERS = 10_001_188


class Float32EvidenceProjection(nn.Module):
    """Keep the tiny concept-solve RHS in FP32 under autocast.

    Reuse the original Linear parameters directly: no new weights, changed
    state_dict keys or RNG consumption. In FP32 this is the exact same F.linear
    operation. Under AMP it promotes only this projection output before tanh,
    so the original Float H and its evidence RHS share a supported solve dtype.
    """
    def __init__(self, original):
        super().__init__()
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.weight = original.weight
        self.bias = original.bias

    def forward(self, value):
        return F.linear(value, self.weight, self.bias).float()


class RecurrentFFNBlock(nn.Module):
    """Causal prewrite recurrent read + residual gated feed-forward layer."""

    def __init__(self):
        super().__init__()
        self.recurrence = RecurrenceOnlyUpperBlock(WIDTH, UPPER_STATE)
        self.norm = nn.LayerNorm(WIDTH)
        self.up = nn.Linear(WIDTH, UPPER_FFN)
        self.gate = nn.Linear(WIDTH, UPPER_FFN)
        self.down = nn.Linear(UPPER_FFN, WIDTH)
        self.ffn_gain = nn.Parameter(torch.tensor(-2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.recurrence(x)
        u = self.norm(z)
        return z + self.ffn_gain.sigmoid() * self.down(
            self.up(u) * F.silu(self.gate(u))
        )


class TACLanguage10M001(ContextTransition026):
    """Original TAC first-stage geometry plus four additional causal blocks."""

    def __init__(self):
        super().__init__(vocab=VOCAB, ffn_width=OUTPUT_FFN)
        self.evidence = Float32EvidenceProjection(self.evidence)
        self.layernorm = nn.Sequential(
            nn.LayerNorm(WIDTH),
            *[RecurrentFFNBlock() for _ in range(UPPER_BLOCKS)],
        )
        self.depth = UPPER_BLOCKS + 1
        self.upper_state_width = UPPER_STATE
        self.ffn_width = OUTPUT_FFN
        self.upper_concept_graph = False
        actual = self.parameter_count()
        deployed = sum(p.numel() for p in self.parameters())
        if actual != EXPECTED_PARAMETERS or deployed != EXPECTED_PARAMETERS:
            raise ValueError(
                f"LANG10M001 count mismatch: trainable={actual} deployed={deployed}"
            )

    def architecture_record(self) -> dict:
        return {
            "variant": "LANG10M001",
            "params": self.parameter_count(),
            "vocab_size": VOCAB,
            "embedding_width": 48,
            "native_memory": "8x8",
            "aligned_memory": "2x16x8",
            "wave_modes": 8,
            "base_concept_slots": "6x16",
            "base_fast_slow_width": 88,
            "upper_blocks": UPPER_BLOCKS,
            "upper_state_width": UPPER_STATE,
            "upper_ffn_width": UPPER_FFN,
            "output_ffn_width": OUTPUT_FFN,
            "output_head_tied": False,
            "concept_evidence_solve_dtype": "FP32 including AMP",
            "window_reset_only": True,
            "trained": False,
            "full_model_streaming": False,
            "composed_pnds_certification": "OPEN",
            "protected_test": "NOT_ACCESSED",
        }
