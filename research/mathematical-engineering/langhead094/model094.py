"""LANGHEAD094 genuine CPython FIT -> separate Rust Book publisher DEV.

Four actual 100k learned language models. GRU and selective diagonal SSM are
real separately trained recurrence controls, not a renamed TAC ablation.
All source/eval results exploratory UNTIL complete matched compute is verified.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'research/mathematical-engineering/langhead093'))
from model093 import TiedHeadTAC093

ARMS=('original','tied_dot','gru2','selective_ssm')
SEEDS=(1701,1702,1703,1704,1705)
PARAMS={'original':100521,'tied_dot':100522,'gru2':100661,'selective_ssm':100404}

@dataclass(frozen=True)
class Recurrent094State:
    owners:tuple[str,...]
    clock:int
    value:torch.Tensor

class GRULanguage094(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding=nn.Embedding(256,32)
        self.recurrent=nn.GRU(32,83,num_layers=2,batch_first=True)
        self.readout=nn.Linear(83,256)
    def start(self,owners):
        if not owners or len(owners)!=len(set(owners)) or any(not isinstance(o,str) or not o for o in owners):
            raise ValueError('unique nonempty owners required')
        return Recurrent094State(tuple(owners),0,self.embedding.weight.new_zeros((2,len(owners),83)))
    def forward(self,tokens,*,state=None,owners=None):
        if tokens.dtype!=torch.long or tokens.ndim!=2 or tokens.shape[1]<1:raise ValueError('observed bytes [B,T]')
        if owners is None: owners=tuple(f'train-{i}' for i in range(tokens.shape[0]))
        if state is None:state=self.start(owners)
        if tuple(owners)!=state.owners or state.value.shape!=(2,len(owners),83) or tokens.shape[0]!=len(owners):
            raise ValueError('owner/GRU state mismatch')
        if not torch.isfinite(state.value).all():raise ValueError('nonfinite state')
        h,v=self.recurrent(self.embedding(tokens),state.value)
        return self.readout(h),Recurrent094State(tuple(owners),state.clock+tokens.shape[1],v)

class SelectiveSSMLanguage094(nn.Module):
    """Diagonal input-selective contractive state, with a gated output FFN.

    h_t = a_t h_(t-1)+(1-a_t) u_t
    a_t = exp(-softplus(raw_decay) sigmoid(G e_t)), u_t=tanh(W e_t).
    State transition is pointwise nonexpansive holding input fixed.
    This is not Mamba-3, Gated DeltaNet-2, or an equivalence claim.
    """
    def __init__(self):
        super().__init__()
        self.embedding=nn.Embedding(256,32)
        self.input=nn.Linear(32,128)
        self.gate=nn.Linear(32,128)
        self.offer=nn.Linear(32,128)
        self.raw_decay=nn.Parameter(torch.zeros(128))
        self.ffn_up=nn.Linear(128,154)
        self.ffn_gate=nn.Linear(128,154)
        self.ffn_out=nn.Linear(154,256)
    def start(self,owners):
        if not owners or len(owners)!=len(set(owners)) or any(not isinstance(o,str) or not o for o in owners):
            raise ValueError('unique nonempty owners required')
        return Recurrent094State(tuple(owners),0,self.embedding.weight.new_zeros((len(owners),128)))
    def forward(self,tokens,*,state=None,owners=None):
        if tokens.dtype!=torch.long or tokens.ndim!=2 or tokens.shape[1]<1:raise ValueError('observed bytes [B,T]')
        if owners is None:owners=tuple(f'train-{i}' for i in range(tokens.shape[0]))
        if state is None:state=self.start(owners)
        if tuple(owners)!=state.owners or state.value.shape!=(len(owners),128) or tokens.shape[0]!=len(owners):
            raise ValueError('owner/SSM state mismatch')
        if not torch.isfinite(state.value).all():raise ValueError('nonfinite SSM state')
        h=state.value;out=[]
        e=self.embedding(tokens)
        decay=F.softplus(self.raw_decay)
        for t in range(tokens.shape[1]):
            z=e[:,t]
            a=torch.exp(-decay*torch.sigmoid(self.gate(z)))
            u=torch.tanh(self.offer(z)+self.input(z))
            h=a*h+(1-a)*u
            f=self.ffn_up(h)*F.silu(self.ffn_gate(h))
            out.append(self.ffn_out(f))
        return torch.stack(out,dim=1),Recurrent094State(tuple(owners),state.clock+tokens.shape[1],h)

def make_model(arm):
    if arm=='original':m=TiedHeadTAC093('original')
    elif arm=='tied_dot':m=TiedHeadTAC093('tied_dot')
    elif arm=='gru2':m=GRULanguage094()
    elif arm=='selective_ssm':m=SelectiveSSMLanguage094()
    else:raise ValueError('unregistered architecture')
    n=sum(p.numel() for p in m.parameters() if p.requires_grad)
    d=sum(p.numel() for p in m.parameters())
    if n!=PARAMS[arm] or d!=n:raise ValueError(f'live learned parameter mismatch: {arm} {n} != {PARAMS[arm]}')
    return m
