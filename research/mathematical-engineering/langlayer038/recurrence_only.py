"""LANGLAYER038: UNTRAINED recurrence-only depth2 candidate.

The complete original CONTEXT026 first stage is inherited unchanged. Only the
extra layer omits concept parameters, concept history and equilibrium solve.
Its 59-coordinate recurrence/readout uses the saved budget; final FFN442 is
identical in width to recurrent depth2. Reset windows, no streaming interface.
"""
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F

_CONTEXT=Path(__file__).resolve().parents[1]/'context026'
if str(_CONTEXT) not in sys.path: sys.path.insert(0,str(_CONTEXT))
from context_transition import ContextTransition026, scalar_affine_scan


class RecurrenceOnlyUpperBlock(nn.Module):
    def __init__(self,width=280,state_width=59):
        super().__init__()
        if type(width) is not int or type(state_width) is not int or width<=0 or state_width<=0:
            raise ValueError('positive integer widths required')
        self.width=width;self.state_width=state_width
        self.input_norm=nn.LayerNorm(width)
        self.input_proj=nn.Linear(width,state_width)
        self.forget=nn.Linear(state_width,state_width)
        self.offer=nn.Linear(state_width,state_width)
        self.read_gate=nn.Linear(2*state_width,state_width)
        self.output=nn.Linear(state_width,width)
        self.residual_gain=nn.Parameter(torch.tensor(-2.0))

    def observe(self,z,prior):
        """Actual upper-only one-step observation; no whole-model streaming API."""
        u=self.input_proj(self.input_norm(z)).tanh()
        alpha=.995*self.forget(u).sigmoid()
        return alpha*prior+(1-alpha)*self.offer(u).tanh()

    def read_from_prior(self,z,prior):
        """Pure prewrite read for the upper diagnostic adapter only."""
        u=self.input_proj(self.input_norm(z)).tanh()
        gate=self.read_gate(torch.cat((u,prior),-1)).sigmoid()
        return z+self.residual_gain.sigmoid()*self.output((u+gate*prior).tanh())

    def forward(self,z,*,return_trace=False):
        if z.ndim!=3 or z.shape[-1]!=self.width or z.shape[1]==0 or not z.is_floating_point():
            raise ValueError('expected floating nonempty [B,T,width]')
        u=self.input_proj(self.input_norm(z)).tanh()
        alpha=.995*self.forget(u).sigmoid()
        offer=self.offer(u).tanh()
        state=scalar_affine_scan(alpha,(1-alpha)*offer)
        prior=F.pad(state[:,:-1],(0,0,1,0))
        read_gate=self.read_gate(torch.cat((u,prior),-1)).sigmoid()
        combined=(u+read_gate*prior).tanh()
        gain=self.residual_gain.sigmoid()
        out=z+gain*self.output(combined)
        if return_trace:
            return out,{'state':state,'prior_state':prior,'read_gate':read_gate,
                        'alpha':alpha,'offer':offer,'gain':gain}
        return out


class RecurrenceOnlyDepth038(ContextTransition026):
    def __init__(self,vocab=256):
        if vocab!=256: raise ValueError('registered byte vocabulary must be256')
        super().__init__(vocab=vocab,ffn_width=442)
        self.layernorm=nn.Sequential(nn.LayerNorm(280),RecurrenceOnlyUpperBlock(280,59))
        self.depth=2;self.upper_state_width=59;self.ffn_width=442

    def architecture_record(self):
        return {'family':'LANGLAYER038','depth':2,'upper_state_width':59,'ffn_width':442,
                'trainable_parameters':self.parameter_count(),
                'deployed_parameters':sum(p.numel() for p in self.parameters()),
                'upper_concept_graph':False,'original_base_concepts_preserved':True,
                'streaming':False,'trained':False,'candidate_DEV_NLL':'NOT_RUN'}
