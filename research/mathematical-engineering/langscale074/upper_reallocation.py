"""LANGSCALE074 — parameter-matched upper recurrent state reallocation.

Keeps original base native fast/slow/8x8+aligned+wave+concept state FIXED,
and reallocates ~500K trained parameters between recurrence-only upper block
and final 256-byte SwiGLU head. This is NOT native associative scaling, nor a
fixed-head pure state-width ablation. All original 070 weights stay intact.
"""
from __future__ import annotations
from pathlib import Path
import sys
from torch import nn
P=Path(__file__).resolve().parents[1]
for n in ("context026","langlayer038"):
    path=str(P/n)
    if path not in sys.path:sys.path.insert(0,path)
from context_transition import ContextTransition026
from recurrence_only import RecurrenceOnlyUpperBlock

WIDTHS={"h24":(24,480),"h48":(48,455),"h96":(96,388),"h128":(128,331)}
COUNTS={"h24":499625,"h48":499623,"h96":499537,"h128":499631}

def expected_parameters(state_width,ffn_width):
    if (type(state_width) is not int or type(ffn_width) is not int
            or state_width<=0 or ffn_width<=0):
        raise ValueError("positive integer widths only")
    # Original source: base 91145, h-upper 4*h^2+564*h, FFN 818*w.
    return 91145+4*state_width*state_width+564*state_width+818*ffn_width

class ReallocatedUpperTAC074(ContextTransition026):
    def __init__(self,tier="h48",vocab=256):
        if vocab!=256 or tier not in WIDTHS:
            raise ValueError("only four source-registered upper width controls")
        h,w=WIDTHS[tier]
        super().__init__(vocab=256,ffn_width=w)
        self.layernorm=nn.Sequential(nn.LayerNorm(280),RecurrenceOnlyUpperBlock(280,h))
        self.depth=2
        self.upper_state_width=h
        self.upper_concept_graph=False
        self.ffn_width=w
        self.tier=tier
        if expected_parameters(h,w)!=COUNTS[tier] or self.parameter_count()!=COUNTS[tier]:
            raise ValueError("source math / live active parameter count mismatch")

    def architecture_record(self):
        return {"family":"LANGSCALE074","tier":self.tier,
                "upper_state_width":self.upper_state_width,
                "ffn_width":self.ffn_width,
                "trainable_parameters":self.parameter_count(),
                "deployed_parameters":sum(p.numel() for p in self.parameters()),
                "base_native_memory":"UNCHANGED",
                "upper_concept_graph":False,"trained":False,
                "whole_model_streaming":False,
                "protected_TEST":"NOT_ACCESSED"}
