"""Independent chronological equation evaluator; does not call either scan kernel.

Uses fitted layer weights, independently traverses source/matrix equations and
final decoder. This is numerical refinement evidence, not full-network Lean.
"""
import torch
from torch.nn import functional as F

def serial(m,values,observed,ages,clock,owners):
    B,T,_=values.shape
    safe=torch.where(observed,values,torch.zeros_like(values))
    e=torch.tanh(m.encoder(torch.cat((safe,observed.to(values.dtype),ages,clock,m.station(owners)[:,None].expand(B,T,4)),-1)))
    S=values.new_zeros(B,13);C=S.clone();E=values.new_zeros(B,62);M=values.new_zeros(B,8,8);count=values.new_zeros(B,1)
    zero=values.new_zeros(B,8)
    for t in range(T):
        x=e[:,t];context=torch.cat((S/C.clamp_min(1e-6),C,E/count.clamp_min(1)),-1)
        p,g=m.fast_map(torch.cat((x,context,zero),-1)).chunk(2,-1)
        fast=(1-g.sigmoid())*context+g.sigmoid()*p.tanh()
        d,g=m.slow_map(torch.cat((fast,context,zero),-1)).chunk(2,-1)
        gate=.25*g.sigmoid();slow=(1-gate)*context+gate*(context+d.tanh()).tanh()
        k=F.normalize(m.address(x),dim=-1,eps=1e-6);v=F.normalize(m.content(x),dim=-1,eps=1e-6)
        if t==T-1:
            q=F.normalize(k+m.query(torch.cat((x,fast,slow),-1)),dim=-1,eps=1e-6)
            read=torch.einsum('bij,bj->bi',M,q)
            d,g=m.fast_map(torch.cat((x,fast,read),-1)).chunk(2,-1);f=(1-g.sigmoid())*fast+g.sigmoid()*d.tanh()
            d,g=m.slow_map(torch.cat((f,slow,read),-1)).chunk(2,-1);gate=.25*g.sigmoid();s=(1-gate)*slow+gate*(slow+d.tanh()).tanh()
            raw=m.readout(torch.cat((f,s,read,x),-1))+m.retrieved_delta(read)
        alpha,b=m.memory_gates(torch.cat((x,fast,slow),-1)).sigmoid().chunk(2,-1);alpha=alpha*(1-b/4)
        use=observed[:,t].any(-1)
        # Rank-one delta equation, not the production matrix-matrix scan.
        decayed=alpha[...,None]*M
        estimate=torch.einsum('bij,bj->bi',decayed,k)
        updated=decayed+b[...,None]*(v-estimate)[...,None]*k[:,None,:]
        M=torch.where(use[:,None,None],updated,M)
        beta=observed[:,t].to(values.dtype)*m.source_admission(x).sigmoid()
        S=(1-beta)*S+beta*safe[:,t]/8;C=(1-beta)*C+beta
        base=m.fast_map(torch.cat((x,values.new_zeros(B,88),zero),-1)).chunk(2,-1)[0].tanh()[:,26:]
        E=E+torch.where(use[:,None],base,torch.zeros_like(base));count=count+use[:,None]
    anchor=torch.where(observed[:,-1,:6],safe[:,-1,:6],8*S[:,:6]/C[:,:6].clamp_min(1e-6))
    d,scale=raw.chunk(2,-1)
    return anchor+2*d.tanh(),.05+F.softplus(scale),S,C,E,M
