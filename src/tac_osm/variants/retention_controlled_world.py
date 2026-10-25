"""TAC v1.7: bounded selective retention/write admission without extra weights.

Actual source-shaped new continuous model candidate: reuse EVERY trainable
weight of the original 93,324-param ParallelTACWorld, but reinterpret the
source memory-gate logits as a *joint controlled write-rate and forgetting*
law before applying the same exact-right-affine 8x8 update.

Raw a=sigmoid(A(e,f,s)), b=sigmoid(B(e,f,s)).
Controlled gates:
  b_eff = b * write_scale (default 1/4),
  a_keep = 1 - fast_forget_scale * (1-a) (default 0.02),
  a_eff = a_keep * (1-b_eff/4).
Then source-shaped update M'=a_eff*M+(b_eff*v-a_eff*b_eff*M@k)k^T.

This is a mathematically bounded memory operator on radius-4 matrices for
normalized keys and values, and has a parallel affine-scan representation.
New a,b remain causal and independent of the previous memory, so exactly
the SAME batch vs step graph works. Critically this makes address-protected
memory *last longer*, but it trades off write rate and may hurt learning
of changed facts. That performance MUST be empirically evaluated.

No state/parameter shape change: inherits the 93,324-param continuous TAC
world graph. The wrapper adds only deterministic scalar gate operations.
No new learned parameters. Initial weights can be remapped exactly from
the native continuous model via the "inner_linear" key mapping.
"""
from __future__ import annotations
import torch
from torch import nn
from .parallel_continuous_world import ParallelTACWorld


class ControlledGateLogits(nn.Module):
    def __init__(self, original_linear:nn.Linear,
                 write_scale:float=.25,fast_forget_scale:float=.02):
        super().__init__()
        if not (0<write_scale<=1 and 0<fast_forget_scale<=1):
            raise ValueError("fixed gate scalars must be in (0,1]")
        self.inner_linear=original_linear
        self.write_scale=float(write_scale)
        self.fast_forget_scale=float(fast_forget_scale)

    def forward(self,x):
        raw=self.inner_linear(x)
        a,b=raw.chunk(2,dim=-1)
        # Reinterpreting the existing learned gates, without adding params.
        a=(1-self.fast_forget_scale*(1-a.sigmoid())).clamp(max=1-1e-5)
        b=self.write_scale*b.sigmoid()
        # Base TAC applies sigmoid to the returned logit pair. This
        # mathematically realizes bounded controlled gates as intended.
        return torch.cat((torch.logit(a),torch.logit(b)),dim=-1)


class RetentionControlledTACWorld(ParallelTACWorld):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.memory_gates=ControlledGateLogits(self.memory_gates)

    def load_original_world_weights(self,original:ParallelTACWorld):
        if sum(p.numel() for p in self.parameters())!=sum(
                p.numel() for p in original.parameters()):
            raise AssertionError("unexpected parameter difference")
        ours=self.state_dict()
        theirs=original.state_dict()
        for k in ours:
            old_key=k.replace("memory_gates.inner_linear.","memory_gates.")
            if old_key not in theirs or ours[k].shape!=theirs[old_key].shape:
                raise ValueError("original weight mapping incompatible at "+k)
            ours[k]=theirs[old_key].clone()
        self.load_state_dict(ours,strict=True)
