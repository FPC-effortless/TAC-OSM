"""COLLECT016 inherited order-aware full graph time-loop reference: no source model/block/memory forward calls.

Original TAC104/TAC110 reference equations are reused with explicit lineage;
the new phase frontend, routing, write and whole-stack composition are below.
This is a numerical oracle on declared fixtures, not an IEEE or AD proof.
"""
import math
import torch
from torch.nn import functional as F
from .common import build
from collect012.reference104 import _linear,_rms,_fresh_state,rotary_sequential
from collect012.reference110 import _bank_reference,ssa_reference,fresh_ssa_reference,ffn_delta_reference,round_reference
from collect011.memory import PhaseState
from collect011.model import CollectState
from tacround110.model import RoundState
from tacopt109.reference.model import State104,LayerState104

def phase_reference(module,x,state,mask):
    b,t,_=x.shape;h,k,v=module.heads,module.key_dim,module.value_dim
    if not hasattr(module,'content'):
        values=_linear(x,module.value).reshape(b,t,h,v).tanh()
        admission=_linear(x,module.admission).sigmoid()*mask[...,None]
        probability=_linear(x,module.router).softmax(-1)
        hard=torch.zeros_like(probability).scatter(-1,probability.argmax(-1,keepdim=True),1.)
        route=hard+(probability-probability.detach())
        result=route[...,None]*admission[...,None]*route[...,None]*values
        return result,PhaseState(state.owners,state.clock+t,state.memory,state.precision,state.signature)
    content=_linear(x,module.content).reshape(b,t,h,k//2).tanh()
    role=_linear(x,module.role).reshape(b,t,h,k//2).tanh().roll(1,-1)
    theta=math.pi*(content+role)
    key=torch.stack((theta.cos(),theta.sin()),-1).flatten(-2)/math.sqrt(k//2)
    values=_linear(x,module.value).reshape(b,t,h,v).tanh()
    admission=_linear(x,module.admission).sigmoid()*mask[...,None]
    route=_linear(x,module.router).softmax(-1) if module.hybrid else torch.ones_like(admission)
    if module.hard_routing:
        hard=torch.zeros_like(route).scatter(-1,route.argmax(-1,keepdim=True),1.)
        route=hard+(route-route.detach())
    memory,p=state.memory,state.precision;reads=[]
    for j in range(t):
        q=key[:,j];rr=route[:,j]
        previous_memory,previous_p=memory,p
        g=admission[:,j]*rr
        residual=values[:,j]-torch.einsum('bhvk,bhk->bhv',memory,q)
        if module.kind=='delta':u=q;candidate_p=None
        else:
            z=torch.einsum('bhkl,bhl->bhk',p,q)
            leverage=(q*z).sum(-1);u=z/leverage.clamp_min(1e-8)[...,None]
            candidate_p=p-torch.einsum('bh,bhk,bhl->bhkl',g/(1+g*leverage),z,z)
            candidate_p=(candidate_p+candidate_p.transpose(-1,-2))/2
        proposed=memory+torch.einsum('bh,bhv,bhk->bhvk',g,residual,u)
        memory=torch.where((g!=0)[...,None,None],proposed,memory)
        if p is not None:p=torch.where((g!=0)[...,None,None],candidate_p,p)
        read_memory=memory if module.read_after else previous_memory
        read=torch.einsum('bhvk,bhk->bhv',read_memory,q)
        reads.append(read*rr[...,None])
    return torch.stack(reads,1),PhaseState(state.owners,state.clock+t,memory,p,state.signature)

def fresh_reference(model,owners):
    native=_fresh_state(model.core,owners);ssa=fresh_ssa_reference(model.sparse_memory,owners)
    phases=[]
    if model.arm!='attention':
        for block in model.core.blocks:
            m=block.phase_memory;w=m.value.weight
            s=w.new_zeros(len(owners),m.heads,m.value_dim,m.key_dim)
            p=None if m.kind=='delta' else torch.eye(m.key_dim,device=w.device,dtype=w.dtype).repeat(len(owners),m.heads,1,1)/m.ridge
            phases.append(PhaseState(owners,0,s,p,m.signature))
    return CollectState(RoundState(native,ssa,model.config_signature),tuple(phases),model.collect_signature)

def whole_reference(model,tokens,state=None,owners=None,write_mask=None):
    owners=tuple(f'fresh-{i}' for i in range(tokens.shape[0])) if owners is None else tuple(owners)
    state=fresh_reference(model,owners) if state is None else state
    if model.arm=='attention':
        y,native=round_reference(model,tokens,state=state.native,owners=owners,write_mask=write_mask)
        return y,CollectState(native,(),model.collect_signature)
    mask=torch.ones_like(tokens,dtype=torch.bool) if write_mask is None else write_mask
    x=F.embedding(tokens,model.core.embedding.weight);b,t=tokens.shape
    layers=[];phases=[];ssa=state.native.ssa
    for i,(block,old,phase) in enumerate(zip(model.core.blocks,state.native.layers,state.phase)):
        c=block.config;z=_rms(x,block.memory_norm)
        fast,fm,fc,fe=_bank_reference(z,z,block.fast,old.fast_memory,old.fast_mass,mask)
        slow,sm,sc,se=_bank_reference(z,z,block.slow,old.slow_memory,old.slow_mass,mask)
        recent,phase=phase_reference(block.phase_memory,z,phase,mask)
        recent=recent.reshape(b,t,c.n_heads,c.value_dim)
        a=_linear(z,block.rotary_retention).sigmoid()[...,None]
        a=torch.where(mask[...,None,None],a,torch.ones_like(a))
        theta=math.pi*_linear(z,block.rotary_angle).tanh().reshape(b,t,c.n_heads,c.value_dim//2)
        theta=torch.where(mask[...,None,None],theta,torch.zeros_like(theta))
        offer=_linear(z,block.rotary_offer).tanh().reshape(b,t,c.n_heads,c.value_dim)
        field=rotary_sequential(a,theta,offer,old.rotary)
        pieces=torch.cat((fast,slow,fast-slow,fast*slow,field,recent,fe[...,None],se[...,None]),-1).flatten(-2)
        x=x+block.residual_scale*_linear(pieces,block.perspective_out)
        x=x+ffn_delta_reference(x,block)
        layers.append(LayerState104(fm,fc,sm,sc,field[:,-1].clone(),old.recent_key,old.recent_value));phases.append(phase)
        if i+1==model.config.depth//2:
            residual,ssa=ssa_reference(model.sparse_memory,x,state=ssa,owners=owners,write_mask=mask)
            x=x+residual/math.sqrt(2*model.config.depth)
    logits=_linear(_rms(x,model.core.output_norm),model.core.output)
    native=State104(owners,state.clock+t,tuple(layers),model.core.config_signature)
    return logits,CollectState(RoundState(native,ssa,model.config_signature),tuple(phases),model.collect_signature)
