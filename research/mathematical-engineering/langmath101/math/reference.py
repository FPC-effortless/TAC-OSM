"""Independent functional input-to-logit/state reference for LANGMATH101 cores.

This file calls no candidate forward method or native affine_scan. It evaluates
the declared equations using copied parameter tensors and explicit serial state
updates. Shared PyTorch primitives are not an independent IEEE/AD proof.
"""
import math
import torch
from torch.nn import functional as F


def linear(p,name,x):
    return F.linear(x,p[name+'.weight'],p.get(name+'.bias'))


def norm(p,name,x):
    return F.layer_norm(x,(x.shape[-1],),p[name+'.weight'],p[name+'.bias'],1e-5)


def serial(a,b,s):
    rows=[]
    for t in range(b.shape[1]):
        s=a[:,t]*s+b[:,t]
        rows.append(s)
    return torch.stack(rows,1)


def budget_head(p,x,prefix='head'):
    z=norm(p,prefix+'.norm',x)
    return linear(p,prefix+'.out',linear(p,prefix+'.up',z)*F.silu(linear(p,prefix+'.gate',z)))


def native098(p,tokens,s=None,clock=0):
    b,t=tokens.shape
    if s is None:
        shape={'context':(48,),'lexical':(3,32),'previous_key':(2,8),'memory':(2,16,8),'mass':(2,8)}
        s={n:p['embedding.weight'].new_zeros((b,*v)) for n,v in shape.items()}
    e=F.embedding(tokens,p['embedding.weight']).tanh()
    history=torch.cat((s['lexical'],e),1);local=e
    for lag in range(3):local=local+p['context_lag'][lag].tanh()*history[:,2-lag:2-lag+t]
    local=local.tanh();d=torch.sigmoid(-p['raw_context_decay'])
    offer=linear(p,'context_input',local).tanh()
    ctx=serial(d[None,None].expand(b,t,-1),(1-d)*offer,s['context'])
    past=torch.cat((s['context'][:,None],ctx[:,:-1]),1)
    raw=linear(p,'address_value',torch.cat((e,local,past),-1))
    key=raw[...,:16].reshape(b,t,2,8).softmax(-1)
    query=raw[...,16:32].reshape(b,t,2,8).softmax(-1)
    value=raw[...,32:64].reshape(b,t,2,16).tanh()
    # Original098 registers FP32 constants and later dtype conversion preserves
    # those rounded values; do not silently substitute exact FP64 literals.
    lo=p['bank_min'];hi=p['bank_max']
    a=lo+(hi-lo)*raw[...,64:].sigmoid()
    if clock==0:a=torch.cat((torch.ones_like(a[:,:1]),a[:,1:]),1)
    previous=torch.cat((s['previous_key'][:,None],key[:,:-1]),1)
    mat=s['memory'];mass=s['mass'];mats=[];masses=[]
    for i in range(t):
        keep=a[:,i,...,None];k=previous[:,i];v=value[:,i]
        mat=keep[...,None]*mat+(1-keep[...,None])*v[...,None]*k[...,None,:]
        mass=keep*mass+(1-keep)*k
        mats.append(mat);masses.append(mass)
    mat=torch.stack(mats,1);mass=torch.stack(masses,1)
    evidence=(mass*query).sum(-1)
    read=(mat*query[...,None,:]).sum(-1)/(evidence[...,None]+1e-4)
    fast,slow=read.unbind(-2)
    gate=linear(p,'compare',torch.cat((local,past,fast,slow,(fast-slow).abs(),evidence),-1)).sigmoid()
    mixed=gate*fast+(1-gate)*slow
    z=norm(p,'norm',torch.cat((e,local,past,fast,slow,mixed,evidence),-1))
    logits=linear(p,'ffn_out',linear(p,'ffn_up',z)*F.silu(linear(p,'ffn_gate',z)))
    return logits,{'context':ctx[:,-1],'lexical':history[:,-3:],'previous_key':key[:,-1],
                   'memory':mat[:,-1],'mass':mass[:,-1]}


def stateless(p,tokens,window):
    e=F.embedding(tokens,p['embedding.weight']).tanh();b,t,d=e.shape
    history=torch.cat((e.new_zeros((b,window-1,d)),e),1)
    x=torch.cat([history[:,window-1-j:window-1-j+t] for j in range(window)],-1)
    return budget_head(p,x),{}


def block(p,prefix,h,s,clock,associative,hierarchy):
    b,t,d=h.shape;c=p[prefix+'.raw_decay'].shape[0]
    rate=linear(p,prefix+'.update',h).sigmoid() if hierarchy else h.new_ones((b,t,1))
    decay=torch.sigmoid(-p[prefix+'.raw_decay'])[None,None]
    a=1-rate*(1-decay)
    context=serial(a,(1-a)*linear(p,prefix+'.context_input',h).tanh(),s['context'])
    past=torch.cat((s['context'][:,None],context[:,:-1]),1)
    if not associative:
        z=norm(p,prefix+'.norm',torch.cat((h,past),-1))
        output=h+linear(p,prefix+'.out',linear(p,prefix+'.up',z)*F.silu(linear(p,prefix+'.gate',z)))
        return output,{'context':context[:,-1]}
    k=s['key'].shape[-1];v=s['matrix'].shape[-2]
    raw=linear(p,prefix+'.address_value',torch.cat((h,past),-1))
    key=raw[...,:2*k].reshape(b,t,2,k).softmax(-1)
    query=raw[...,2*k:4*k].reshape(b,t,2,k).softmax(-1)
    value=raw[...,4*k:4*k+2*v].reshape(b,t,2,v).tanh()
    lo=h.new_tensor([.97,.995]);hi=h.new_tensor([.995,.99995])
    alpha=1-rate*(1-(lo+(hi-lo)*raw[...,-2:].sigmoid()))
    if clock==0:alpha=torch.cat((torch.ones_like(alpha[:,:1]),alpha[:,1:]),1)
    previous=torch.cat((s['key'][:,None],key[:,:-1]),1)
    mat=s['matrix'];mass=s['mass'];mats=[];masses=[]
    for i in range(t):
        keep=alpha[:,i,...,None];kk=previous[:,i];vv=value[:,i]
        mat=keep[...,None]*mat+(1-keep[...,None])*vv[...,None]*kk[...,None,:]
        mass=keep*mass+(1-keep)*kk
        mats.append(mat);masses.append(mass)
    mat=torch.stack(mats,1);mass=torch.stack(masses,1)
    evidence=(mass*query).sum(-1)
    read=(mat*query[...,None,:]).sum(-1)/(evidence[...,None]+1e-4)
    fast,slow=read.unbind(-2)
    gate=linear(p,prefix+'.compare',torch.cat((h,past,fast,slow,(fast-slow).abs(),evidence),-1)).sigmoid()
    mixed=gate*fast+(1-gate)*slow
    z=torch.cat((h,past,fast,slow,mixed,evidence),-1)
    output=h+linear(p,prefix+'.out',norm(p,prefix+'.norm',z)).tanh()
    return output,{'context':context[:,-1],'key':key[:,-1],'matrix':mat[:,-1],'mass':mass[:,-1]}


def stacked(p,tokens,s,clock,depth,associative,hierarchy):
    e=F.embedding(tokens,p['embedding.weight']).tanh();t=tokens.shape[1]
    history=torch.cat((s['lexical'],e),1);h=e
    for lag in range(3):h=h+p['local_lag'][lag].tanh()*history[:,2-lag:2-lag+t]
    h=h.tanh();final={'lexical':history[:,-3:]};levels=[]
    for i in range(depth):
        if hierarchy and i:h=linear(p,f'upward.{i-1}',h).tanh()
        names=('context','key','matrix','mass') if associative else ('context',)
        ss={k:s[f'{i}.{k}'] for k in names}
        h,ns=block(p,f'blocks.{i}',h,ss,clock,associative,hierarchy)
        levels.append(h);final.update({f'{i}.{k}':v for k,v in ns.items()})
    if hierarchy:
        for i in range(depth-2,-1,-1):h=levels[i]+linear(p,f'downward.{i}',h).tanh()
    return budget_head(p,h),final


def forward(arm,p,tokens,state=None,clock=0):
    if arm in ('native098','factorized098'):return native098(p,tokens,state,clock)
    if arm in ('stateless_current','stateless_window4'):return stateless(p,tokens,1 if arm=='stateless_current' else 4)
    if arm.startswith('stack_'):
        _,kind,depth=arm.split('_');return stacked(p,tokens,state,clock,int(depth),kind=='assoc',False)
    if arm.startswith('hierarchy_'):
        _,kind,depth=arm.split('_');return stacked(p,tokens,state,clock,int(depth),kind=='assoc',True)
    if arm=='gru':
        e=F.embedding(tokens,p['embedding.weight']).tanh();h=state['hidden'];values=[]
        for t in range(tokens.shape[1]):
            gi=F.linear(e[:,t],p['recurrent.weight_ih_l0'],p['recurrent.bias_ih_l0'])
            gh=F.linear(h,p['recurrent.weight_hh_l0'],p['recurrent.bias_hh_l0'])
            ir,iz,inn=gi.chunk(3,-1);hr,hz,hn=gh.chunk(3,-1)
            r=(ir+hr).sigmoid();z=(iz+hz).sigmoid();n=(inn+r*hn).tanh()
            h=(1-z)*n+z*h;values.append(h)
        return budget_head(p,torch.stack(values,1)),{'hidden':h}
    if arm=='selective_ssm':
        e=F.embedding(tokens,p['embedding.weight']).tanh();a=.999*linear(p,'retain',e).sigmoid()
        hidden=serial(a,(1-a)*linear(p,'offer',e).tanh(),state['hidden'])
        out=(linear(p,'readout',hidden)+linear(p,'current',e)).tanh()
        return budget_head(p,out),{'hidden':hidden[:,-1]}
    if arm=='transformer':
        joined=torch.cat((state['tokens'],tokens),1);b,n=joined.shape
        position=torch.arange(n,dtype=p['embedding.weight'].dtype,device=tokens.device)[:,None]
        frequency=torch.exp(torch.arange(0,32,2,dtype=position.dtype,device=tokens.device)*(-math.log(10000.)/32))
        phase=position*frequency;pe=torch.stack((phase.sin(),phase.cos()),-1).flatten(-2)
        h=F.embedding(joined,p['embedding.weight'])*math.sqrt(32)+pe[None]
        z=norm(p,'attention_norm',h)
        q,k,v=F.linear(z,p['attention.in_proj_weight'],p['attention.in_proj_bias']).chunk(3,-1)
        q,k,v=[x.reshape(b,n,4,8).transpose(1,2) for x in (q,k,v)]
        score=(q@k.transpose(-1,-2))/math.sqrt(8)
        mask=torch.ones(n,n,dtype=torch.bool,device=tokens.device).triu(1)
        score=score.masked_fill(mask,float('-inf'));attention=(score.softmax(-1)@v).transpose(1,2).reshape(b,n,32)
        h=h+linear(p,'attention.out_proj',attention)
        h=h+linear(p,'ffn_out',F.gelu(linear(p,'ffn_up',norm(p,'ffn_norm',h))))
        return budget_head(p,h[:,clock:]),{'tokens':joined}
    raise NotImplementedError('No mathematical reference registered for '+arm)
