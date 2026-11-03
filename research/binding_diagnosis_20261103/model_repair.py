"""FLOCK013 prospective AD-only repair; original variants remain immutable.

Only the two write-step calls use the repaired zero-gain adjoint. No parameters,
losses, initializers, stored facts, gates, read order or projections are added.
Instance methods are replaced; module globals are never patched.
"""
import hashlib
import math
import types
import torch
from torch.nn import functional as F
from collect018.model import ContextPhase, context_block
from collect016.model import CarriedTAC
from collect011.memory import PhaseState
from tacopt109.reference.model import LayerState104
from tacopt109.reference.kernels import rotary_scan
from tacopt109.paired_banks import read_bank_pair
from write_repair import write_step


def separated(memory, value_input, address_input, state, mask, owners):
    memory.validate(value_input, state, owners, mask)
    if address_input.shape != value_input.shape:
        raise ValueError("address/value geometry differs")
    shape = (*value_input.shape[:2], memory.heads, memory.key_dim//2)
    theta = math.pi*(memory.content(address_input).reshape(shape).tanh()+memory.role(address_input).reshape(shape).tanh().roll(1,-1))
    key = torch.stack((theta.cos(), theta.sin()), -1).flatten(-2)/math.sqrt(memory.key_dim//2)
    value = memory.value(value_input).reshape(*value_input.shape[:2], memory.heads, memory.value_dim).tanh()
    probability = memory.router(address_input).softmax(-1)
    hard = F.one_hot(probability.argmax(-1), memory.heads).to(probability.dtype)
    route = hard+(probability-probability.detach())
    gain = memory.admission(address_input).sigmoid()*mask[...,None]*route
    matrix, precision = state.memory, state.precision
    reads = []
    for key_t, value_t, gain_t, route_t in zip(key.unbind(1), value.unbind(1), gain.unbind(1), route.unbind(1)):
        matrix, precision = write_step(matrix, precision, key_t, value_t, gain_t, kind=memory.kind)
        reads.append((matrix@key_t.unsqueeze(-1)).squeeze(-1)*route_t[...,None])
    return torch.stack(reads, 1), PhaseState(owners, state.clock+value_input.shape[1], matrix.contiguous(), precision.contiguous(), memory.signature)


def phase_forward(self, x, state=None, write_mask=None, *, owners=None):
    if owners is None:
        if state is not None:
            raise ValueError("explicit continuing owners required")
        owners = tuple(f"fresh-{i}" for i in range(x.shape[0]))
    owners = self.owners(owners)
    state = self.start(owners) if state is None else state
    mask = torch.ones(x.shape[:2], device=x.device, dtype=torch.bool) if write_mask is None else write_mask
    return separated(self, x, x, state, mask, owners)


def repaired_context_block(self, x, state, phase, mask, mode, clock, *, owners):
    # Exact source arithmetic/order from collect018.context_block.
    c=self.config; b,t,_=x.shape; z=self.memory_norm(x); projected=self._input_projections(z)
    inputs=[self._bank_inputs(getattr(self,n),n,z,getattr(state,n+'_memory'),getattr(state,n+'_mass'),mask,projected) for n in ('fast','slow')]
    fr,sr=read_bank_pair(*inputs,chunk_size=c.chunk_size,mode=mode)
    fast,fm,fc,fe=fr; slow,sm,sc,se=sr
    alpha=projected['rotary_retention'].sigmoid()[...,None]
    a=torch.where(mask[...,None,None],alpha,torch.ones_like(alpha))
    theta=math.pi*projected['rotary_angle'].tanh().reshape(b,t,c.n_heads,c.value_dim//2)
    theta=torch.where(mask[...,None,None],theta,torch.zeros_like(theta))
    offer=projected['rotary_offer'].tanh().reshape(b,t,c.n_heads,c.value_dim)
    field=rotary_scan(a,theta,offer,state.rotary,mode='parallel' if mode=='chunk' else 'recurrent')
    components=torch.cat((fast,slow,fast-slow,fast*slow,field,torch.zeros_like(field),fe[...,None],se[...,None]),-1).flatten(-2)
    context=x+self.residual_scale*self.perspective_out(components)
    address=self.memory_norm(context)
    recent,phase=separated(self.phase_memory,z,address,phase,mask,owners)
    recent=recent.reshape(b,t,c.n_heads,c.value_dim)
    pieces=torch.cat((fast,slow,fast-slow,fast*slow,field,recent,fe[...,None],se[...,None]),-1).flatten(-2)
    x=x+self.residual_scale*self.perspective_out(pieces)
    zz=self.ffn_norm(x); v,g=F.linear(zz,torch.cat((self.ffn_value.weight,self.ffn_gate.weight))).split(c.ffn_dim,-1)
    x=x+self.residual_scale*self.ffn_out(v*F.silu(g))
    return x,LayerState104(fm,fc,sm,sc,field[:,-1].clone(),state.recent_key,state.recent_value),phase


class RepairedPhase(CarriedTAC):
    def __init__(self):
        super().__init__("full_bptt")
        self.variant = "FLOCK013_PHASE_ZERO_GAIN_ADJOINT"
        self.collect_signature = hashlib.sha256((self.collect_signature+self.variant).encode()).hexdigest()
        for block in self.core.blocks:
            block.phase_memory.forward = types.MethodType(phase_forward, block.phase_memory)


class RepairedContext(ContextPhase):
    def __init__(self):
        super().__init__()
        self.variant = "FLOCK013_CONTEXT_ZERO_GAIN_ADJOINT"
        self.collect_signature = hashlib.sha256((self.collect_signature+self.variant).encode()).hexdigest()
        for block in self.core.blocks:
            block.forward = types.MethodType(repaired_context_block, block)


def build(family, seed):
    torch.manual_seed(seed)
    if family == "phase":
        return RepairedPhase()
    if family == "context":
        return RepairedContext()
    raise ValueError("Only the two memory variants use this repair")
