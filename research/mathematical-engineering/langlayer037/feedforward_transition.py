"""LANGLAYER037: tokenwise residual FFN-only depth control at ~500K parameters.

A controlled negative architecture: inherits the exact CONTEXT026 first TAC
native/aligned/wave/concept memory stage, but added depth receives NO newly
learned recurrent state, concept graph, or temporal token mixing. All
additional blocks operate on the same-token hidden representation only.
Source runs use reset windows and retain all original causality restrictions.
"""
from __future__ import annotations

from pathlib import Path
import sys
import torch
from torch import nn

_CONTEXT026=Path(__file__).resolve().parents[1]/"context026"
if str(_CONTEXT026) not in sys.path:
    sys.path.insert(0,str(_CONTEXT026))
from context_transition import ContextTransition026


class TokenwiseResidualFFNBlock(nn.Module):
    """SwiGLU-like tokenwise controlled nonlinearity; no temporal state."""
    def __init__(self,width:int=280,hidden:int=48):
        super().__init__()
        if width<=0 or hidden<=0:
            raise ValueError("positive block widths required")
        self.input_norm=nn.LayerNorm(width)
        self.up=nn.Linear(width,hidden)
        self.gate=nn.Linear(width,hidden)
        self.down=nn.Linear(hidden,width)
        self.residual_gain=nn.Parameter(torch.tensor(-2.0))

    def forward(self,z:torch.Tensor):
        if z.ndim!=3 or z.shape[1]<1 or z.shape[-1]!=self.input_norm.normalized_shape[0]:
            raise ValueError("expected nonempty [B,T,model_width]")
        p=self.input_norm(z)
        nonlinear=self.up(p).tanh()*self.gate(p).sigmoid()
        return z+self.residual_gain.sigmoid()*self.down(nonlinear)


class FeedforwardDepthControl037(ContextTransition026):
    """First complete native TAC stage + 1/2 independent tokenwise FFN blocks.

    Hidden coordinates deliberately match those of the corresponding
    recurrent/concept LANGLAYER034 (48/24). Parameters are kept within 0.1%
    of the native, recurrent-layer and three-stage experimental controls.
    """

    def __init__(self,vocab:int=256,depth:int=2):
        if depth not in (2,3):
            raise ValueError("only preregistered depths 2 and 3 allowed")
        hidden=48 if depth==2 else 24
        ffn_width=450 if depth==2 else 449
        super().__init__(vocab=vocab,ffn_width=ffn_width)
        self.layernorm=nn.Sequential(
            nn.LayerNorm(280),
            *(TokenwiseResidualFFNBlock(280,hidden) for _ in range(depth-1)),
        )
        self.depth=depth
        self.residual_width=hidden
        self.ffn_width=ffn_width

    def architecture_record(self):
        return {"family":"LANGLAYER037_FFN_CONTROL",
                "depth":self.depth,
                "state_width":0,
                "residual_hidden":self.residual_width,
                "ffn_width":self.ffn_width,
                "trainable_parameters":self.parameter_count(),
                "deployed_parameters":sum(p.numel() for p in self.parameters()),
                "additional_temporal_state":False,
                "trained":False,
                "protected_TEST":"NOT_ACCESSED"}
