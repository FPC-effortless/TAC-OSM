"""LANGLAYER040: missing factorial controls for upper graph presence x state width.

This is a matched ~500K *reallocation* study, not a fixed-decoder
pure lesion: FFN widths are adjusted to pay for true active upper
parameters. First-stage CONTEXT026 native/aligned/wave/BASE concepts
are exactly preserved. Additional upper graph is optional.
"""
from __future__ import annotations
from pathlib import Path
import sys
from torch import nn

_P=Path(__file__).resolve().parents[1]
for n in ("context026","langlayer034","langlayer038"):
    if str(_P/n) not in sys.path:sys.path.insert(0,str(_P/n))
from context_transition import ContextTransition026
from layered_transition import CausalResidualTACBlock
from recurrence_only import RecurrenceOnlyUpperBlock

class FactorialDepthControl040(ContextTransition026):
    """Train missing two 2x2 upper factorial corners at ~500K params.

    Existing corners: h48/full-graph/FFN442=499725 (LANG035)
                      h59/no-graph/FFN442=499901 (LANG039).
    New corners:      h48/no-graph/FFN455=499623
                      h59/full-graph/FFN426=499848.
    """
    def __init__(self,vocab=256,*,state_width=48,upper_graph=False):
        if vocab!=256 or (state_width,upper_graph) not in ((48,False),(59,True)):
            raise ValueError("only preregistered factorial corners allowed")
        ffn_width=455 if state_width==48 else 426
        super().__init__(vocab=vocab,ffn_width=ffn_width)
        block=(CausalResidualTACBlock(280,state_width) if upper_graph
               else RecurrenceOnlyUpperBlock(280,state_width))
        self.layernorm=nn.Sequential(nn.LayerNorm(280),block)
        self.depth=2
        self.upper_state_width=state_width
        self.upper_concept_graph=upper_graph
        self.ffn_width=ffn_width

    def architecture_record(self):
        return {"family":"LANGLAYER040","upper_state_width":self.upper_state_width,
                "upper_concept_graph":self.upper_concept_graph,
                "ffn_width":self.ffn_width,
                "trainable_parameters":self.parameter_count(),
                "deployed_parameters":sum(p.numel() for p in self.parameters()),
                "trained":False,"protected_TEST":"NOT_ACCESSED",
                "whole_model_streaming":False}
