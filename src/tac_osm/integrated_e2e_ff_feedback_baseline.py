"""Strong feed-forward control for the clean E2E-008 task.

Unlike FF-001, this control receives the same information available across
the PLM q1->q2 boundary: q1 entity representation, q2 entity representation,
q2 query indices/operator, and the detached q1 environment outcome.
There is no persistent state, CASM, verifier, or state update.
"""
from __future__ import annotations
import torch
from torch import Tensor, nn
import torch.nn.functional as F

class FeedForwardFeedbackMultimodal(nn.Module):
    def __init__(self,vocab_size:int=40,hidden_dim:int=64):
        super().__init__()
        from tac_osm.integrated_e2e_005 import TextEncoder,ImageEncoder,AudioEncoder,SharedRepresentation
        self.text=TextEncoder(vocab_size,hidden_dim)
        self.image=ImageEncoder(hidden_dim)
        self.audio=AudioEncoder(hidden_dim)
        self.rep=SharedRepresentation(hidden_dim)
        self.main=nn.Sequential(nn.Linear(157,87),nn.GELU(),nn.Linear(87,87),nn.GELU(),nn.Linear(87,2))
        self.query_residual=nn.Sequential(nn.Linear(157,2),nn.GELU(),nn.Linear(2,2))
    def encode(self,text:Tensor,image:Tensor,audio:Tensor)->Tensor:
        return self.rep(self.text(text),self.image(image),self.audio(audio))
    def query(self,q1_repr:Tensor,q2_repr:Tensor,i:Tensor,j:Tensor,op:Tensor,outcome1:Tensor)->Tensor:
        one=lambda x,n:F.one_hot(x.long(),n).to(q2_repr.dtype)
        q=torch.cat([q1_repr,q2_repr,one(i,12),one(j,12),one(op,4),outcome1.float().view(-1,1)],dim=-1)
        return self.main(q)+self.query_residual(q)
    def parameter_count(self)->int:
        return sum(p.numel() for p in self.parameters())
