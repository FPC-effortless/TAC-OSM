"""Conventional byte controls, approximately 100K genuinely used parameters.

No optimizer/data access. Native source-matched math certification, complete
IEEE/AD proofs and architecture superiority are OPEN. Transformer stores raw
observations and recomputes its prefix; no KV-cache speed claim is made.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F

TARGET=100370
CONTROL_NAMES=('gru','selective_ssm','transformer')

@dataclass(frozen=True)
class State:
    owners: tuple[str,...]
    clock: int
    tensors: dict[str,torch.Tensor]
    arm: str

    def tensor_bytes(self):
        return sum(t.numel()*t.element_size() for t in self.tensors.values())


def _owners(owners,batch):
    owners=tuple(f'train-{i}' for i in range(batch)) if owners is None else tuple(owners)
    if batch<1 or len(owners)!=batch or len(set(owners))!=len(owners) or any(type(x) is not str or not x for x in owners):
        raise ValueError('unique nonempty owner strings must align with batch')
    return owners


def scan(a,b,initial,parallel=True):
    if not parallel:
        s=initial;out=[]
        for t in range(b.shape[1]):s=a[:,t]*s+b[:,t];out.append(s)
        return torch.stack(out,1)
    n=b.shape[1]
    if n==1:return a*initial[:,None]+b
    p=n//2
    odds=scan(a[:,1:2*p:2]*a[:,:2*p:2],a[:,1:2*p:2]*b[:,:2*p:2]+b[:,1:2*p:2],initial,True)
    prev=torch.cat((initial[:,None],odds),1)[:,:((n+1)//2)]
    evens=a[:,::2]*prev+b[:,::2]
    paired=torch.stack((evens[:,:p],odds),2).flatten(1,2)
    return torch.cat((paired,evens[:,-1:]),1) if n%2 else paired


class Head(nn.Module):
    def __init__(self,d,width):
        super().__init__();self.norm=nn.LayerNorm(d)
        self.up=nn.Linear(d,width);self.gate=nn.Linear(d,width);self.out=nn.Linear(width,256)
    def forward(self,x):
        x=self.norm(x);return self.out(self.up(x)*F.silu(self.gate(x)))


class Base(nn.Module):
    def counts(self):
        return {'trainable':sum(p.numel() for p in self.parameters() if p.requires_grad),
                'deployed':sum(p.numel() for p in self.parameters()),
                'fixed_buffer_bytes':sum(b.numel()*b.element_size() for b in self.buffers())}
    def _input(self,tokens,state,owners):
        if (not isinstance(tokens,torch.Tensor) or tokens.dtype!=torch.long or tokens.ndim!=2
            or min(tokens.shape)<1 or bool((tokens<0).any()) or bool((tokens>255).any())):
            raise ValueError('observed Long[B,T] bytes in0..255 required')
        if tokens.device!=self.embedding.weight.device:raise ValueError('device mismatch')
        if self.embedding.weight.dtype not in (torch.float32,torch.float64):raise ValueError('FP32/FP64 only')
        owners=_owners(owners,tokens.shape[0]);state=self.start(owners) if state is None else state
        if (not isinstance(state,State) or state.owners!=owners or state.arm!=self.arm
            or type(state.clock) is not int or state.clock<0):raise ValueError('owner/clock/arm mismatch')
        for v in state.tensors.values():
            if v.device!=tokens.device or v.shape[0]!=tokens.shape[0] or not bool(torch.isfinite(v).all()):raise ValueError('state shape/device/nonfinite')
        return state,owners


class GRUControl(Base):
    """e=tanh(E[x]); h=GRU(e,h_prev); logits=gated_head(h)."""
    arm='gru'
    def __init__(self):
        super().__init__();self.embedding=nn.Embedding(256,32);self.recurrent=nn.GRU(32,64,batch_first=True)
        self.head=Head(64,189)
    def start(self,owners):
        owners=_owners(owners,len(owners));return State(owners,0,{'hidden':self.embedding.weight.new_zeros(len(owners),64)},self.arm)
    def forward(self,tokens,*,state=None,owners=None,parallel=True,**kwargs):
        if kwargs:raise ValueError('unsupported input fields: '+str(sorted(kwargs)))
        state,owners=self._input(tokens,state,owners)
        if set(state.tensors)!={'hidden'} or state.tensors['hidden'].shape!=(len(owners),64) or state.tensors['hidden'].dtype!=self.embedding.weight.dtype:raise ValueError('complete GRU hidden state required')
        y,h=self.recurrent(self.embedding(tokens).tanh(),state.tensors['hidden'][None])
        return self.head(y),State(owners,state.clock+tokens.shape[1],{'hidden':h[0]},self.arm)


class SelectiveSSMControl(Base):
    """Input-dependent diagonal affine recurrence, not a Mamba reproduction.

    a_t=.999*sigmoid(W_a e_t+b_a), b_t=(1-a_t)*tanh(W_b e_t+b_b);
    s_t=a_t*s_(t-1)+b_t; h_t=tanh(W_s s_t+W_e e_t+b); logits=head(h).
    """
    arm='selective_ssm'
    def __init__(self):
        super().__init__();self.embedding=nn.Embedding(256,32)
        self.offer=nn.Linear(32,64);self.retain=nn.Linear(32,64)
        self.readout=nn.Linear(64,64);self.current=nn.Linear(32,64);self.head=Head(64,211)
    def start(self,owners):
        owners=_owners(owners,len(owners));return State(owners,0,{'hidden':self.embedding.weight.new_zeros(len(owners),64)},self.arm)
    def forward(self,tokens,*,state=None,owners=None,parallel=True,**kwargs):
        if kwargs:raise ValueError('unsupported input fields: '+str(sorted(kwargs)))
        state,owners=self._input(tokens,state,owners)
        if set(state.tensors)!={'hidden'} or state.tensors['hidden'].shape!=(len(owners),64) or state.tensors['hidden'].dtype!=self.embedding.weight.dtype:raise ValueError('complete diagonal state required')
        e=self.embedding(tokens).tanh();a=.999*self.retain(e).sigmoid();b=(1-a)*self.offer(e).tanh()
        s=scan(a,b,state.tensors['hidden'],parallel);h=(self.readout(s)+self.current(e)).tanh()
        return self.head(h),State(owners,state.clock+tokens.shape[1],{'hidden':s[:,-1]},self.arm)


class TransformerControl(Base):
    """One causal pre-norm self-attention block, sinusoidal positions, no dropout.

    All previously observed bytes are explicit in state. Prefix recomputation
    supplies exact accessible history; this is not a KV-cached implementation.
    """
    arm='transformer'
    def __init__(self):
        super().__init__();self.embedding=nn.Embedding(256,32)
        self.attention=nn.MultiheadAttention(32,4,dropout=0.,batch_first=True)
        self.attention_norm=nn.LayerNorm(32);self.ffn_norm=nn.LayerNorm(32)
        self.ffn_up=nn.Linear(32,64);self.ffn_out=nn.Linear(64,32);self.head=Head(32,259)
    def start(self,owners):
        owners=_owners(owners,len(owners));return State(owners,0,{'tokens':torch.empty(len(owners),0,dtype=torch.long,device=self.embedding.weight.device)},self.arm)
    def forward(self,tokens,*,state=None,owners=None,parallel=True,**kwargs):
        if kwargs:raise ValueError('unsupported input fields: '+str(sorted(kwargs)))
        state,owners=self._input(tokens,state,owners)
        if set(state.tensors)!={'tokens'} or state.tensors['tokens'].dtype!=torch.long or state.tensors['tokens'].shape!=(len(owners),state.clock):raise ValueError('complete observed token history required')
        previous=state.tensors['tokens']
        if bool((previous<0).any()) or bool((previous>255).any()):raise ValueError('invalid historical byte')
        joined=torch.cat((previous,tokens),1);b,n=joined.shape
        positions=torch.arange(n,dtype=self.embedding.weight.dtype,device=joined.device)[:,None]
        frequencies=torch.exp(torch.arange(0,32,2,dtype=self.embedding.weight.dtype,device=joined.device)*(-math.log(10000.)/32))
        phase=positions*frequencies;pe=torch.stack((phase.sin(),phase.cos()),-1).flatten(-2)
        h=self.embedding(joined)*math.sqrt(32)+pe[None];z=self.attention_norm(h)
        mask=torch.ones(n,n,dtype=torch.bool,device=joined.device).triu(1)
        attended,_=self.attention(z,z,z,attn_mask=mask,need_weights=False)
        h=h+attended;h=h+self.ffn_out(F.gelu(self.ffn_up(self.ffn_norm(h))))
        return self.head(h[:,state.clock:]),State(owners,n,{'tokens':joined},self.arm)


def make_control(name):
    return {'gru':GRUControl,'selective_ssm':SelectiveSSMControl,'transformer':TransformerControl}[name]()


if __name__=='__main__':
    for name in CONTROL_NAMES:
        m=make_control(name);print(name,m.counts())
