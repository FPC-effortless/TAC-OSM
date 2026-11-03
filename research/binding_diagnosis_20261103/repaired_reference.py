"""Independent full forward equations and smooth-candidate surrogate adjoint.

No candidate forward/block/write function is called. Zero-gain candidate values
follow the exact-real continuation; IEEE signed-zero freeze is checked against
the candidate separately. This is a numerical reference, not a full Lean proof.
"""
import math
import torch
from torch.nn import functional as F
from collect016.reference import fresh_reference
from collect012.reference104 import _linear, _rms, rotary_sequential
from collect012.reference110 import _bank_reference, ssa_reference, ffn_delta_reference
from collect011.memory import PhaseState
from collect011.model import CollectState
from tacround110.model import RoundState
from tacopt109.reference.model import State104, LayerState104


def phase_equations(module, x, state, mask, address):
    b,t,_=x.shape; h,k,v=module.heads,module.key_dim,module.value_dim
    content=_linear(address,module.content).reshape(b,t,h,k//2).tanh()
    role=_linear(address,module.role).reshape(b,t,h,k//2).tanh().roll(1,-1)
    theta=math.pi*(content+role)
    key=torch.stack((theta.cos(),theta.sin()),-1).flatten(-2)/math.sqrt(k//2)
    values=_linear(x,module.value).reshape(b,t,h,v).tanh()
    admission=_linear(address,module.admission).sigmoid()*mask[...,None]
    probability=_linear(address,module.router).softmax(-1)
    hard=torch.zeros_like(probability).scatter(-1,probability.argmax(-1,keepdim=True),1.)
    route=hard+(probability-probability.detach())
    memory,p=state.memory,state.precision; reads=[]
    for j in range(t):
        q=key[:,j]; r=route[:,j]; gain=admission[:,j]*r
        residual=values[:,j]-torch.einsum('bhvk,bhk->bhv',memory,q)
        z=torch.einsum('bhkl,bhl->bhk',p,q)
        leverage=(q*z).sum(-1)
        direction=z/leverage.clamp_min(1e-8)[...,None]
        candidate_p=p-torch.einsum('bh,bhk,bhl->bhkl',gain/(1+gain*leverage),z,z)
        p=(candidate_p+candidate_p.transpose(-1,-2))/2
        memory=memory+torch.einsum('bh,bhv,bhk->bhvk',gain,residual,direction)
        reads.append(torch.einsum('bhvk,bhk->bhv',memory,q)*r[...,None])
    return torch.stack(reads,1), PhaseState(state.owners,state.clock+t,memory,p,state.signature)


def whole_reference(model,tokens,state=None,owners=None,write_mask=None):
    family=getattr(model,'variant','')
    if family not in ('FLOCK013_PHASE_ZERO_GAIN_ADJOINT','FLOCK013_CONTEXT_ZERO_GAIN_ADJOINT'):
        raise ValueError('registered repaired family required')
    context=family=='FLOCK013_CONTEXT_ZERO_GAIN_ADJOINT'
    owners=tuple(f'fresh-{i}' for i in range(tokens.shape[0])) if owners is None else tuple(owners)
    state=fresh_reference(model,owners) if state is None else state
    mask=torch.ones_like(tokens,dtype=torch.bool) if write_mask is None else write_mask
    x=F.embedding(tokens,model.core.embedding.weight); b,t=tokens.shape; layers=[]; phases=[]; ssa=state.native.ssa
    for i,(block,old,phase) in enumerate(zip(model.core.blocks,state.native.layers,state.phase)):
        c=block.config; z=_rms(x,block.memory_norm)
        fast,fm,fc,fe=_bank_reference(z,z,block.fast,old.fast_memory,old.fast_mass,mask)
        slow,sm,sc,se=_bank_reference(z,z,block.slow,old.slow_memory,old.slow_mass,mask)
        retention=_linear(z,block.rotary_retention).sigmoid()[...,None]
        retention=torch.where(mask[...,None,None],retention,torch.ones_like(retention))
        theta=math.pi*_linear(z,block.rotary_angle).tanh().reshape(b,t,c.n_heads,c.value_dim//2)
        theta=torch.where(mask[...,None,None],theta,torch.zeros_like(theta))
        offer=_linear(z,block.rotary_offer).tanh().reshape(b,t,c.n_heads,c.value_dim)
        field=rotary_sequential(retention,theta,offer,old.rotary)
        if context:
            components=torch.cat((fast,slow,fast-slow,fast*slow,field,torch.zeros_like(field),fe[...,None],se[...,None]),-1).flatten(-2)
            address=_rms(x+block.residual_scale*_linear(components,block.perspective_out),block.memory_norm)
        else:
            address=z
        recent,phase=phase_equations(block.phase_memory,z,phase,mask,address)
        recent=recent.reshape(b,t,c.n_heads,c.value_dim)
        pieces=torch.cat((fast,slow,fast-slow,fast*slow,field,recent,fe[...,None],se[...,None]),-1).flatten(-2)
        x=x+block.residual_scale*_linear(pieces,block.perspective_out)
        x=x+ffn_delta_reference(x,block)
        layers.append(LayerState104(fm,fc,sm,sc,field[:,-1].clone(),old.recent_key,old.recent_value))
        phases.append(phase)
        if i+1==model.config.depth//2:
            residual,ssa=ssa_reference(model.sparse_memory,x,state=ssa,owners=owners,write_mask=mask)
            x=x+residual/math.sqrt(2*model.config.depth)
    logits=_linear(_rms(x,model.core.output_norm),model.core.output)
    native=State104(owners,state.clock+t,tuple(layers),model.core.config_signature)
    return logits,CollectState(RoundState(native,ssa,model.config_signature),tuple(phases),model.collect_signature)
