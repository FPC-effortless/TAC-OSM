"""Independent equations: no candidate model/block/phase/GRU forward calls."""
import math
import torch
from torch.nn import functional as F
from .model import GRUState
from collect016.reference import fresh_reference,whole_reference as old_reference
from collect012.reference104 import _linear,_rms,rotary_sequential
from collect012.reference110 import _bank_reference,ssa_reference,ffn_delta_reference
from collect011.memory import PhaseState
from collect011.model import CollectState
from tacround110.model import RoundState
from tacopt109.reference.model import State104,LayerState104
def phase_equations(module,x,state,mask,address_input=None):
    address_input=x if address_input is None else address_input
    b,t,_=x.shape;h,k,v=module.heads,module.key_dim,module.value_dim
    if not hasattr(module,'content'):
        values=_linear(x,module.value).reshape(b,t,h,v).tanh()
        admission=_linear(address_input,module.admission).sigmoid()*mask[...,None]
        probability=_linear(address_input,module.router).softmax(-1)
        hard=torch.zeros_like(probability).scatter(-1,probability.argmax(-1,keepdim=True),1.)
        route=hard+(probability-probability.detach())
        result=route[...,None]*admission[...,None]*route[...,None]*values
        return result,PhaseState(state.owners,state.clock+t,state.memory,state.precision,state.signature)
    content=_linear(address_input,module.content).reshape(b,t,h,k//2).tanh()
    role=_linear(address_input,module.role).reshape(b,t,h,k//2).tanh().roll(1,-1)
    theta=math.pi*(content+role)
    key=torch.stack((theta.cos(),theta.sin()),-1).flatten(-2)/math.sqrt(k//2)
    values=_linear(x,module.value).reshape(b,t,h,v).tanh()
    admission=_linear(address_input,module.admission).sigmoid()*mask[...,None]
    route=_linear(address_input,module.router).softmax(-1) if module.hybrid else torch.ones_like(admission)
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

def gru_reference(model,tokens,state=None,owners=None,write_mask=None):
    owners=tuple(f'fresh-{i}' for i in range(tokens.shape[0])) if owners is None else tuple(owners)
    c=model.core;h=c.embedding.weight.new_zeros(tokens.shape[0],125) if state is None else state.hidden[0]
    x=F.embedding(tokens,c.embedding.weight);rows=[]
    for j in range(tokens.shape[1]):
        a=F.linear(x[:,j],c.gru.weight_ih_l0,c.gru.bias_ih_l0);b=F.linear(h,c.gru.weight_hh_l0,c.gru.bias_hh_l0)
        ir,iz,inn=a.chunk(3,-1);hr,hz,hn=b.chunk(3,-1)
        r=(ir+hr).sigmoid();z=(iz+hz).sigmoid();n=(inn+r*hn).tanh();h=(1-z)*n+z*h;rows.append(h)
    y=torch.stack(rows,1);mu=y.mean(-1,keepdim=True);var=(y-mu).square().mean(-1,keepdim=True)
    y=(y-mu)*torch.rsqrt(var+c.output_norm.eps)*c.output_norm.weight+c.output_norm.bias
    return _linear(y,c.output),GRUState(owners,(0 if state is None else state.clock)+tokens.shape[1],h[None])

def whole_reference(model,tokens,state=None,owners=None,write_mask=None):
    if hasattr(model.core,'gru'):return gru_reference(model,tokens,state,owners,write_mask)
    if getattr(model,'variant','')!='collect018_context':return old_reference(model,tokens,state,owners,write_mask)
    owners=tuple(f'fresh-{i}' for i in range(tokens.shape[0])) if owners is None else tuple(owners)
    state=fresh_reference(model,owners) if state is None else state
    mask=torch.ones_like(tokens,dtype=torch.bool) if write_mask is None else write_mask
    x=F.embedding(tokens,model.core.embedding.weight);b,t=tokens.shape;layers=[];phases=[];ssa=state.native.ssa
    for i,(block,old,phase) in enumerate(zip(model.core.blocks,state.native.layers,state.phase)):
        c=block.config;z=_rms(x,block.memory_norm)
        fast,fm,fc,fe=_bank_reference(z,z,block.fast,old.fast_memory,old.fast_mass,mask)
        slow,sm,sc,se=_bank_reference(z,z,block.slow,old.slow_memory,old.slow_mass,mask)
        a=_linear(z,block.rotary_retention).sigmoid()[...,None];a=torch.where(mask[...,None,None],a,torch.ones_like(a))
        theta=math.pi*_linear(z,block.rotary_angle).tanh().reshape(b,t,c.n_heads,c.value_dim//2)
        theta=torch.where(mask[...,None,None],theta,torch.zeros_like(theta));offer=_linear(z,block.rotary_offer).tanh().reshape(b,t,c.n_heads,c.value_dim)
        field=rotary_sequential(a,theta,offer,old.rotary)
        components=torch.cat((fast,slow,fast-slow,fast*slow,field,torch.zeros_like(field),fe[...,None],se[...,None]),-1).flatten(-2)
        address=_rms(x+block.residual_scale*_linear(components,block.perspective_out),block.memory_norm)
        recent,phase=phase_equations(block.phase_memory,z,phase,mask,address)
        recent=recent.reshape(b,t,c.n_heads,c.value_dim)
        pieces=torch.cat((fast,slow,fast-slow,fast*slow,field,recent,fe[...,None],se[...,None]),-1).flatten(-2)
        x=x+block.residual_scale*_linear(pieces,block.perspective_out);x=x+ffn_delta_reference(x,block)
        layers.append(LayerState104(fm,fc,sm,sc,field[:,-1].clone(),old.recent_key,old.recent_value));phases.append(phase)
        if i+1==model.config.depth//2:
            residual,ssa=ssa_reference(model.sparse_memory,x,state=ssa,owners=owners,write_mask=mask);x=x+residual/math.sqrt(2*model.config.depth)
    logits=_linear(_rms(x,model.core.output_norm),model.core.output)
    native=State104(owners,state.clock+t,tuple(layers),model.core.config_signature)
    return logits,CollectState(RoundState(native,ssa,model.config_signature),tuple(phases),model.collect_signature)
