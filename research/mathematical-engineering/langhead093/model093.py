"""LANGHEAD093: direct tied-output causal alternatives on unchanged 100521p TAC.

Only the observed byte x_t and trained embedding matrix enter the auxiliary score.
No hidden retrieval keys, next-byte label, elapsed-time claims, or evaluation inputs.
The permuted control tests token-to-candidate alignment without extra capacity.
"""
from __future__ import annotations
import math
import sys
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'research/mathematical-engineering/langmem086'))
from compact_tac import CompactDualClippedTAC086

ARMS=('original','tied_dot','tied_dot_norm','norm_only','shuffled_dot')
SEEDS=(1701,1702,1703,1704,1705)
COUNTS={a:100521+(a!='original') for a in ARMS}

class TiedHeadTAC093(nn.Module):
    def __init__(self,arm):
        super().__init__()
        if arm not in ARMS:raise ValueError('unregistered head')
        self.arm=arm
        self.core=CompactDualClippedTAC086()
        if arm!='original':
            self.raw_strength=nn.Parameter(torch.tensor(0.0))
        self.register_buffer('shuffled', (torch.arange(256)*73+41)%256,persistent=True)
        if self.counts()['trainable']!=COUNTS[arm]:raise ValueError('model count')
    def counts(self):
        n=sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'trainable':n,'deployed':sum(p.numel() for p in self.parameters())}
    def start(self,owners):return self.core.start(owners)
    def forward(self,tokens,*,state=None,owners=None):
        logits,state=self.core(tokens,state=state,owners=owners)
        if self.arm=='original':return logits,state
        E=self.core.embedding.weight
        observed=E[tokens]
        candidates=E[self.shuffled] if self.arm=='shuffled_dot' else E
        s=2.*torch.matmul(observed,candidates.transpose(0,1))
        if self.arm in ('tied_dot_norm','norm_only'):
            norm2=candidates.square().sum(-1)
            if self.arm=='norm_only':s=-norm2.view(1,1,-1).expand_as(s)
            else:s=s-norm2.view(1,1,-1)
        g=.02*torch.sigmoid(self.raw_strength)
        return logits+g*(s-s.mean(-1,keepdim=True)),state
