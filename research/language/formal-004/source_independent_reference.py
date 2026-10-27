"""Independent exact-operation PyTorch forward reference for TAC-LANG-GATE-003.

Uses source PARAMETER TENSORS but never calls source forward/stream_step,
time_features, keys or memory reads. This is numerical correspondence, not
a formal compiler/IEEE proof or new trained model.
"""
import math
import torch
import torch.nn.functional as F

PERIODS = (7, 13, 29, 59)

def dense(layer, x):
    return x @ layer.weight.mT + (layer.bias if layer.bias is not None else 0.)

def phi(index, dtype, device, batch):
    i = torch.as_tensor(index, dtype=torch.int64, device=device)
    terms = []
    for period in PERIODS:
        angle = i.remainder(period).to(dtype) * (2 * math.pi / period)
        terms += [torch.cos(angle), torch.sin(angle)]
    return torch.stack(terms).view(1, 8).expand(batch, 8)

def norm(x):
    return x / torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(1e-6)

def silu(x):
    return x * torch.sigmoid(x)

def time_features(mode, count, batch, dtype, device):
    pos = torch.full((batch,), count, dtype=dtype, device=device)
    if mode == "no_time":
        return torch.zeros(batch, 7, dtype=dtype, device=device)
    if mode == "legacy":
        return torch.stack((torch.log1p(pos), torch.sin(pos/16), torch.cos(pos/16),
            torch.full_like(pos, math.log(2.)), torch.log1p(pos+1),
            torch.full_like(pos, math.log(2.)), (pos>0).to(dtype)), dim=-1)
    return torch.stack((torch.log1p(pos), torch.sin(pos/16), torch.cos(pos/16),
        torch.sin(pos/64), torch.cos(pos/64), torch.sin(pos/256),
        torch.cos(pos/256)), dim=-1)

def read_aligned(model, old, key_native, t):
    b = key_native.shape[0]
    lags = (1,8,32,128) if model.position_mode=="multiscale_relative" else (1,)
    outputs, scores = [], []
    for lag in lags:
        address = phi(max(t-lag,0), old.dtype, old.device, b)
        key = norm(torch.einsum("bi,hij->bhj", address, model.rotation))
        head_read = torch.einsum("bhvk,bhk->bhv", old, key)
        gate = torch.softmax(dense(model.read_router, address), dim=-1)
        merged = (head_read * gate[...,None]).sum(1)
        valid = t>=lag
        outputs.append(merged if valid else merged*0)
        score = (key_native * norm(address)).sum(-1)
        scores.append(score if valid else torch.full_like(score,-1e4))
    if len(outputs)==1:
        return outputs[0]
    weights=torch.softmax(torch.stack(scores,-1),dim=-1)
    return (torch.stack(outputs,1)*weights[...,None]).sum(1)

def independent_step(model, token, state):
    """Source-shaped full token-to-logits and next-state equation."""
    b=token.numel();t=state["count"];device=token.device;dtype=model.embedding.weight.dtype
    e = torch.tanh(F.embedding(token,model.embedding.weight)+
        dense(model.time_proj,time_features(model.position_mode,t,b,dtype,device)))
    zeros_fast=torch.zeros_like(state["sum"])
    zeros_r=e.new_zeros(b,8)
    basis,_=dense(model.fast_map,torch.cat((e,zeros_fast,zeros_r),-1)).chunk(2,-1)
    context=state["sum"]/max(t,1)
    p,g=dense(model.fast_map,torch.cat((e,context,zeros_r),-1)).chunk(2,-1)
    fast0=context+torch.sigmoid(g)*(torch.tanh(p)-context)
    p,g=dense(model.slow_map,torch.cat((fast0,context,zeros_r),-1)).chunk(2,-1)
    slow0=context+.25*torch.sigmoid(g)*(torch.tanh(context+torch.tanh(p))-context)
    nk=norm(dense(model.key_native,e))
    nv=norm(dense(model.value_native,e))
    a,beta=torch.sigmoid(dense(model.native_gates,torch.cat((e,fast0,slow0),-1))).chunk(2,-1)
    a=a*(1-beta/4)
    old_native=state["native"]
    nr=(old_native@nk[...,None]).squeeze(-1)
    next_native=a[...,None]*old_native+(beta*(nv-a*nr))[...,None]*nk[:,None,:]
    p,g=dense(model.fast_map,torch.cat((e,fast0,nr),-1)).chunk(2,-1)
    fast=fast0+torch.sigmoid(g)*(torch.tanh(p)-fast0)
    p,g=dense(model.slow_map,torch.cat((fast,slow0,nr),-1)).chunk(2,-1)
    slow=slow0+.25*torch.sigmoid(g)*(torch.tanh(slow0+torch.tanh(p))-slow0)
    address=phi(t,dtype,device,b)
    key=norm(torch.einsum("bi,hij->bhj",address,model.rotation))
    value=torch.tanh(dense(model.source_content,e))
    route=torch.softmax(dense(model.source_router,e),dim=-1)
    alpha=torch.sigmoid(dense(model.source_retention,e))
    modulation=dense(model.source_silu_mod,e)
    if model.gate_mode=="silu":
        modulation=silu(2*modulation)
    beta=torch.sigmoid(dense(model.source_strength,e)+modulation)*route
    old=state["aligned"]
    pre=torch.einsum("bhvk,bhk->bhv",old,key)
    next_aligned=alpha[:,:,None,None]*old+(
        beta[:,:,None,None]*(value[:,None]-alpha[:,:,None]*pre)[:,:,:,None]*key[:,:,None,:])
    read=read_aligned(model,old,nk,t)
    z=torch.cat((fast,slow,nr,e,read),-1)
    mean=z.mean(-1,keepdim=True); variance=(z-mean).square().mean(-1,keepdim=True)
    z=(z-mean)*torch.rsqrt(variance+model.norm.eps)*model.norm.weight+model.norm.bias
    logits=dense(model.read_out,silu(dense(model.read_in,z)))+(
        torch.sigmoid(model.res_logit)*dense(model.res_out,silu(dense(model.res_in,z))))
    next_state={"sum":state["sum"]+torch.tanh(basis),
        "native":next_native,"aligned":next_aligned,
        "count":t+1,"owner":state["owner"]}
    return logits,next_state

def run(model,tokens,trace=False):
    b,t=tokens.shape
    state={"sum":model.embedding.weight.new_zeros(b,88),
        "native":model.embedding.weight.new_zeros(b,8,8),
        "aligned":model.embedding.weight.new_zeros(b,2,16,8),
        "count":0,"owner":torch.arange(b,dtype=torch.int64,device=tokens.device)}
    outputs=[]; old=[]
    for i in range(t):
        old.append(state["aligned"])
        y,state=independent_step(model,tokens[:,i],state)
        outputs.append(y)
    result=torch.stack(outputs,1)
    return (result,state,torch.stack(old,1)) if trace else result

def historical(model,tokens,prior,lag):
    b,t=tokens.shape
    if not isinstance(lag,int) or lag<1 or lag>=t:
        raise ValueError("invalid historical lag")
    scores=[]
    for i in range(lag,t):
        addr=phi(i-lag,model.embedding.weight.dtype,tokens.device,b)
        key=norm(torch.einsum("bi,hij->bhj",addr,model.rotation))
        values=torch.einsum("bhvk,bhk->bhv",prior[:,i],key)
        router=torch.softmax(dense(model.read_router,addr),-1)
        retrieved=(values*router[...,None]).sum(1)
        hidden=dense(model.hist_project,retrieved)
        scores.append(hidden @ model.embedding.weight.mT)
    return torch.stack(scores,1),tokens[:,:-lag]
