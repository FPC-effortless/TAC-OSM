"""Actual TAC110 LM integration, replacing its local token-attention lane.

Fast/slow associative banks, rotating field, perspective mixer, FFN and mid-stack
SSA are retained. mHC is outside this study; it is neither removed from a fitted
model nor claimed tested. Fresh native attention control uses identical API.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F

VENDOR=Path(__file__).resolve().parents[1]/"vendor"
for package in ("tacround110","tacopt109"):
    loaded=sys.modules.get(package)
    if loaded is not None and (not getattr(loaded,"__file__",None) or not Path(loaded.__file__).resolve().is_relative_to(VENDOR)):
        raise RuntimeError("A different pinned TAC source already owns the "+package+" namespace")
sys.path.insert(0,str(VENDOR))
from tacround110.model import ResearchTAC110, RoundConfig, RoundState
from tacopt109.optimized import OptimizedBlock104, OptimizationOptions
from tacopt109.reference.model import LayerState104, State104
from tacopt109.reference.kernels import rotary_scan
from tacopt109.paired_banks import read_bank_pair
from .memory import PhaseMemory, PhaseState

FFN_DIM={"attention":64,"delta":85,"phase_p":85,"hybrid_p":83,"hybrid_hard_p":83,"phase_p_support":85}

def named_state_tensors(state):
    out={f"native.{i}.{name}":v for i,l in enumerate(state.native.layers) for name,v in l.tensors().items()}
    for n in ("keys","values","survival"):
        v=getattr(state.native.ssa,n)
        if v is not None:out["ssa."+n]=v
    for i,s in enumerate(state.phase):
        out[f"phase.{i}.memory"]=s.memory
        if s.precision is not None:out[f"phase.{i}.precision"]=s.precision
    return out


@dataclass(frozen=True)
class CollectState:
    native: RoundState
    phase: tuple[PhaseState, ...]
    signature: str

    @property
    def owners(self): return self.native.owners
    @property
    def clock(self): return self.native.clock
    def detach(self):
        return CollectState(self.native.detach(),tuple(s.detach() for s in self.phase),self.signature)
    def state_bytes(self):
        return self.native.state_bytes()+sum(s.state_bytes() for s in self.phase)
    def payload(self):
        return dict(schema="collect011.tac.v1",native=self.native.payload(),
                    phase=[s.payload() for s in self.phase],signature=self.signature)
    @classmethod
    def from_payload(cls,p):
        if set(p) != {"schema","native","phase","signature"} or p["schema"] != "collect011.tac.v1":
            raise ValueError("complete COLLECT011 state required")
        return cls(RoundState.from_payload(p["native"]),tuple(PhaseState.from_payload(s) for s in p["phase"]),p["signature"])


class PhaseBlock(OptimizedBlock104):
    def __init__(self,config,*,kind,hybrid,support_gate,hard_routing=False):
        super().__init__(config,OptimizationOptions(pack_projections=True,fuse_banks=True))
        del self.local_query,self.local_key,self.local_value
        if config.local_rope or config.window!=1:raise ValueError("phase lane requires no native KV cache")
        h,k,v=config.n_heads,config.key_dim,config.value_dim
        self.phase_memory=PhaseMemory(config.d_model,key_dim=k if hybrid else h*k,
            value_dim=v if hybrid else h*v,heads=h if hybrid else 1,
            kind=kind,hybrid=hybrid,support_gate=support_gate,hard_routing=hard_routing)

    def _input_projections(self,x):
        modules=[(bank+"."+name,getattr(getattr(self,bank),name)) for bank in ("fast","slow")
                 for name in ("query","key","value","retention","eligibility")]
        modules += [(name,getattr(self,name)) for name in ("rotary_retention","rotary_angle","rotary_offer")]
        weight=torch.cat([m.weight for _,m in modules])
        bias=torch.cat([m.bias if m.bias is not None else x.new_zeros(m.out_features) for _,m in modules])
        return dict(zip((n for n,_ in modules),F.linear(x,weight,bias).split([m.out_features for _,m in modules],-1)))

    def forward(self,x,state,phase,write_mask,mode,clock,*,owners):
        c=self.config;b,t,_=x.shape
        normalized=self.memory_norm(x)
        projected=self._input_projections(normalized)
        inputs=[self._bank_inputs(getattr(self,bank),bank,normalized,
                getattr(state,bank+"_memory"),getattr(state,bank+"_mass"),write_mask,projected)
                for bank in ("fast","slow")]
        fast_result,slow_result=read_bank_pair(*inputs,chunk_size=c.chunk_size,mode=mode)
        fast,fm,fc,fe=fast_result;slow,sm,sc,se=slow_result
        recent,phase=self.phase_memory(normalized,state=phase,write_mask=write_mask,owners=owners)
        recent=recent.reshape(b,t,c.n_heads,c.value_dim)
        pieces=[fast,slow,fast-slow,fast*slow]
        alpha=projected["rotary_retention"].sigmoid()[...,None]
        a=torch.where(write_mask[...,None,None],alpha,torch.ones_like(alpha))
        theta=math.pi*projected["rotary_angle"].tanh().reshape(b,t,c.n_heads,c.value_dim//2)
        theta=torch.where(write_mask[...,None,None],theta,torch.zeros_like(theta))
        offer=projected["rotary_offer"].tanh().reshape(b,t,c.n_heads,c.value_dim)
        trajectory=rotary_scan(a,theta,offer,state.rotary,mode="parallel" if mode=="chunk" else "recurrent")
        pieces.extend((trajectory,recent,fe[...,None],se[...,None]))
        x=x+self.residual_scale*self.perspective_out(torch.cat(pieces,-1).flatten(-2))
        normalized=self.ffn_norm(x)
        value,gate=F.linear(normalized,torch.cat((self.ffn_value.weight,self.ffn_gate.weight))).split(c.ffn_dim,-1)
        x=x+self.residual_scale*self.ffn_out(value*F.silu(gate))
        # Empty native KV fields preserve the native schema without retaining tokens.
        return x,LayerState104(fm,fc,sm,sc,trajectory[:,-1].clone(),state.recent_key,state.recent_value),phase


class CollectiveTAC(ResearchTAC110):
    """Native attention, delta, wide phase/P, soft/hard hybrid and support arms."""
    ARMS=("attention","delta","phase_p","hybrid_p","hybrid_hard_p","phase_p_support")

    def __init__(self,arm="phase_p",*,width=32,ffn_dim=64,depth=4):
        if arm not in self.ARMS: raise ValueError("unknown registered arm")
        config=RoundConfig(depth=depth,d_model=width,ffn_dim=ffn_dim,ssa=True,
            n_heads=4,key_dim=16,value_dim=16,window=320 if arm=="attention" else 1,
            local_rope=arm=="attention")
        super().__init__(config)
        self.arm=arm
        if arm != "attention":
            self.core.blocks=nn.ModuleList(PhaseBlock(config.native_config(),
                kind="delta" if arm=="delta" else "preconditioned",hybrid=arm in ("hybrid_p","hybrid_hard_p"),
                support_gate=arm=="phase_p_support",hard_routing=arm=="hybrid_hard_p") for _ in range(depth))
        self.collect_signature=hashlib.sha256(json.dumps(dict(arm=arm,config=self.config_signature),sort_keys=True).encode()).hexdigest()

    def start(self,owners):
        native=super().start(owners)
        phases=() if self.arm=="attention" else tuple(b.phase_memory.start(native.owners) for b in self.core.blocks)
        return CollectState(native,phases,self.collect_signature)

    def forward(self,tokens,*,state=None,owners=None,write_mask=None,mode="chunk"):
        if mode not in ("chunk","recurrent"): raise ValueError("legal mode required")
        if owners is None:
            if state is not None: raise ValueError("explicit continuing owners required")
            if not isinstance(tokens,torch.Tensor) or tokens.ndim!=2: raise ValueError("Long[B,T] required")
            owners=tuple(f"fresh-{i}" for i in range(tokens.shape[0]))
        owners=self.core._owners(owners)
        if state is None: state=self.start(owners)
        if not isinstance(state,CollectState) or state.signature!=self.collect_signature or state.owners!=owners:
            raise ValueError("same-variant/owner complete CollectState required")
        expected=0 if self.arm=="attention" else self.config.depth
        if len(state.phase)!=expected or any(s.clock!=state.clock or s.owners!=owners for s in state.phase):
            raise ValueError("complete synchronized phase state required")
        native=state.native
        # Do not delegate attention to ResearchTAC110.forward: its virtual start()
        # expects a RoundState, whereas our public start() returns CollectState.
        self.core._validate(tokens,native.native,owners,fresh=False)
        if native.config_signature!=self.config_signature or native.ssa is None or native.ssa.clock!=native.clock:
            raise ValueError("complete synchronized native/SSA state required")
        if write_mask is None: write_mask=torch.ones_like(tokens,dtype=torch.bool)
        if not isinstance(write_mask,torch.Tensor) or write_mask.dtype!=torch.bool or write_mask.shape!=tokens.shape or write_mask.device!=tokens.device:
            raise ValueError("Bool[B,T] write mask required")
        x=self.core.embedding(tokens);layers=[];phases=[];sparse=native.ssa
        ropes=self.core._shared_rope(state.clock,tokens.shape[1],x.dtype)
        for i,(block,layer,rope) in enumerate(zip(self.core.blocks,native.layers,ropes)):
            if self.arm=="attention":
                x,final=block(x,layer,write_mask,mode,state.clock,rope=rope)
            else:
                x,final,phase=block(x,layer,state.phase[i],write_mask,mode,state.clock,owners=owners)
                phases.append(phase)
            layers.append(final)
            if i+1==self.config.depth//2:
                residual,sparse=self.sparse_memory(x,state=sparse,write_mask=write_mask,owners=owners)
                x=x+residual/math.sqrt(2*self.config.depth)
        logits=self.core.output(self.core.output_norm(x))
        ns=State104(owners,state.clock+tokens.shape[1],tuple(layers),self.core.config_signature)
        rs=RoundState(ns,sparse,self.config_signature)
        return logits,CollectState(rs,tuple(phases),self.collect_signature)
