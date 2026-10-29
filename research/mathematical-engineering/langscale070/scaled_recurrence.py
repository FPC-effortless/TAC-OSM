"""LANGSCALE070 — capacity scaling with unchanged native TAC state architecture.

Single-axis FFN expansion/contraction only, not a complete all-axis scaling
law. Inherit original CONTEXT026 first stage and original LANGLAYER038
48-dimensional graph-free recurrent upper block. Original 500K source remains
unchanged. Three exact registered widths: 150,455,1067 hidden coordinates.
Memory geometry is intentionally FIXED at each scale to isolate head capacity.
Untrained constructor; fit permission belongs to preregistered manifest.
"""
from __future__ import annotations
from pathlib import Path
import sys
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
for n in ("context026","langlayer038"):
    path=str(ROOT/n)
    if path not in sys.path:sys.path.insert(0,path)
from context_transition import ContextTransition026
from recurrence_only import RecurrenceOnlyUpperBlock

WIDTHS={"p250k":150,"p500k":455,"p1000k":1067}
COUNTS={"p250k":250133,"p500k":499623,"p1000k":1000239}
RECURRENT=48


class ScaledRecurrenceTAC070(ContextTransition026):
    def __init__(self,scale="p500k",vocab=256):
        if vocab!=256 or scale not in WIDTHS:
            raise ValueError("only registered 256-byte vocabulary and three scales")
        width=WIDTHS[scale]
        super().__init__(vocab=vocab,ffn_width=width)
        self.layernorm=nn.Sequential(nn.LayerNorm(280),RecurrenceOnlyUpperBlock(280,RECURRENT))
        self.depth=2
        self.upper_state_width=RECURRENT
        self.upper_concept_graph=False
        self.ffn_width=width
        self.scale=scale
        if self.parameter_count()!=COUNTS[scale]:
            raise ValueError("original source parameter budget changed")

    def architecture_record(self):
        return {"family":"LANGSCALE070",
                "scale":self.scale,"ffn_width":self.ffn_width,
                "upper_state_width":RECURRENT,"upper_concept_graph":False,
                "trainable_parameters":self.parameter_count(),
                "deployed_parameters":sum(p.numel() for p in self.parameters()),
                "training":"NOT_RUN","protected_TEST":"NOT_ACCESSED",
                "original_recurrence_preserved":True,
                "scale_axis":"FFN_WIDTH_ONLY",
                "whole_model_streaming":False}
