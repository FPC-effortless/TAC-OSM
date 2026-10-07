"""Parameter-matched feed-forward control for clean E2E-008."""
from __future__ import annotations
import torch
from torch import Tensor, nn
import torch.nn.functional as F

class FeedForwardMultimodal(nn.Module):
    """No-persistence control: direct addressed representation -> query head."""
    def __init__(self, vocab_size: int = 40, hidden_dim: int = 64):
        super().__init__()
        from tac_osm.integrated_e2e_005 import TextEncoder, ImageEncoder, AudioEncoder, SharedRepresentation
        self.text=TextEncoder(vocab_size,hidden_dim)
        self.image=ImageEncoder(hidden_dim)
        self.audio=AudioEncoder(hidden_dim)
        self.rep=SharedRepresentation(hidden_dim)
        self.main=nn.Sequential(nn.Linear(92,105),nn.GELU(),nn.Linear(105,105),nn.GELU(),nn.Linear(105,2))
        self.query_residual=nn.Sequential(nn.Linear(92,9),nn.GELU(),nn.Linear(9,2))
    def encode(self,text:Tensor,image:Tensor,audio:Tensor)->Tensor:
        return self.rep(self.text(text),self.image(image),self.audio(audio))
    def query(self,entity_repr:Tensor,i:Tensor,j:Tensor,op:Tensor)->Tensor:
        q=torch.cat([entity_repr,F.one_hot(i.long(),12).to(entity_repr.dtype),F.one_hot(j.long(),12).to(entity_repr.dtype),F.one_hot(op.long(),4).to(entity_repr.dtype)],dim=-1)
        return self.main(q)+self.query_residual(q)
    def parameter_count(self)->int:
        return sum(p.numel() for p in self.parameters())
