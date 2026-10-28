#!/usr/bin/env python3
"""TAC-LANG-PAR-001: genuine scan-parallel experimental language adaptation.

Mathematical source: PNDS #50 `SourceAlignedModel.lean` and #49
`SourceAlignedMemory.lean`, specifically causal input-only prep, two 16x8
source-ordinal addressed gated delta heads, 8x8 native corrective memory,
4-block right-affine associative scan and read-before-current-write.
TAC implementation lineage: draft PRs #520/#521, source SHA
63ea0aa67904fa946b9baa69d2e4b0cc23dc1c9a.

This is a NEW language-specialized geometry, not the *same* Torch graph or a
Lean-verified implementation of PAR-MEM-010 (whose typed inputs and outputs
are R14). Novel language output requires separate formal refinement.
"""
from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F

V=128
PERIODS=(7,13,29,59)

def affine_scan(P,Q,initial=None):
    """Work-efficient chronological right-affine Blelloch scan, Torch/autograd.
    P/Q [B,T,8,8]; M_next=M@P+Q. Exact-real associative, FP approximate.
    """
    B,T=P.shape[:2]
    assert T>0 and P.shape==Q.shape==(B,T,8,8)
    size=1<<(T-1).bit_length()
    eye=torch.eye(8,device=P.device,dtype=P.dtype).reshape(1,1,8,8)
    if size>T:
        P=torch.cat([P,eye.expand(B,size-T,8,8)],1)
        Q=torch.cat([Q,Q.new_zeros(B,size-T,8,8)],1)
    levels=[(P,Q)]
    while levels[-1][0].shape[1]>1:
        pp,qq=levels[-1];left=pp[:,0::2];right=pp[:,1::2]
        levels.append((left@right,qq[:,0::2]@right+qq[:,1::2]))
    excp=eye.expand(B,1,8,8); excq=Q.new_zeros(B,1,8,8)
    for pp,qq in levels[-2::-1]:
        lp=pp[:,0::2];lq=qq[:,0::2]
        rp=excp@lp;rq=excq@lp+lq
        excp=torch.stack([excp,rp],2).reshape(B,-1,8,8)
        excq=torch.stack([excq,rq],2).reshape(B,-1,8,8)
    incp=excp@P; incq=excq@P+Q
    if initial is not None:incq=initial[:,None]@incp+incq
    return incq[:,:T]

def schedule(k,v,alpha,beta,mask=None):
    """Corrective PNDS law alpha*M + beta*(v-alpha*M*k) k^T."""
    *shape,T,d=k.shape
    eye=torch.eye(d,device=k.device,dtype=k.dtype)
    if mask is not None:
        alpha=torch.where(mask,alpha,torch.ones_like(alpha))
        beta=torch.where(mask,beta,torch.zeros_like(beta))
    P=alpha[...,None]*(eye-beta[...,None]*k.unsqueeze(-1)*k.unsqueeze(-2))
    Q=beta[...,None]*v.unsqueeze(-1)*k.unsqueeze(-2)
    return P,Q

def rectangular_affine_scan(P,Q):
    """Associative scan with shared [B,T,8,8] transform and [B,T,16,8] innovation.

    Same exact-real matrix update as scanning the two 8-row value blocks
    separately, but P is combined once per head. FP32 reassociation may differ.
    """
    B,T=P.shape[:2]
    assert P.shape==(B,T,8,8) and Q.shape==(B,T,16,8) and T>0
    length=1<<(T-1).bit_length()
    eye=torch.eye(8,device=P.device,dtype=P.dtype).view(1,1,8,8)
    if length>T:
        P=torch.cat((P,eye.expand(B,length-T,8,8)),1)
        Q=torch.cat((Q,Q.new_zeros(B,length-T,16,8)),1)
    levels=[(P,Q)]
    while levels[-1][0].shape[1]>1:
        pp,qq=levels[-1]
        rp=pp[:,1::2]
        levels.append((pp[:,0::2]@rp,qq[:,0::2]@rp+qq[:,1::2]))
    exP=eye.expand(B,1,8,8)
    exQ=Q.new_zeros(B,1,16,8)
    for pp,qq in levels[-2::-1]:
        lp,lq=pp[:,0::2],qq[:,0::2]
        rp=exP@lp
        rq=exQ@lp+lq
        exP=torch.stack((exP,rp),2).reshape(B,-1,8,8)
        exQ=torch.stack((exQ,rq),2).reshape(B,-1,16,8)
    return (exQ@P+Q)[:,:T]


def rectangular_schedule(k,v,alpha,beta):
    """P=alpha(I-beta k k^T), Q=beta v k^T; value width 16."""
    B,T,d=k.shape
    assert d==8 and v.shape==(B,T,16)
    eye=torch.eye(8,dtype=k.dtype,device=k.device)
    P=alpha[...,None]*(eye-beta[...,None]*k.unsqueeze(-1)*k.unsqueeze(-2))
    Q=beta[...,None]*v.unsqueeze(-1)*k.unsqueeze(-2)
    return P,Q

def one_scan(k,v,a,b):
    return affine_scan(*schedule(k,v,a,b))

def unit_test_scan():
    torch.manual_seed(4)
    for n in (1,2,7,16,33):
        k=F.normalize(torch.randn(2,n,8,dtype=torch.float64),dim=-1)
        v=torch.tanh(torch.randn(2,n,8,dtype=torch.float64))
        a=torch.rand(2,n,1,dtype=torch.float64)*.1+.9
        b=torch.rand(2,n,1,dtype=torch.float64)*.1
        P,Q=schedule(k,v,a,b)
        y=affine_scan(P,Q)
        m=torch.zeros(2,8,8,dtype=torch.float64);out=[]
        for t in range(n):
            m=m@P[:,t]+Q[:,t];out.append(m)
        d=(y-torch.stack(out,1)).abs().max().item()
        assert d<3e-12,(n,d)

class ParallelTACLanguage(nn.Module):
    """Language adapter to PNDS-parallel fast+slow / two-source-head graph.

    State [B,88] fast source basis sum + [B,8,8] native matrix +
    [B,2,16,8] aligned memory + two clocks = 410 FP32, with owner/ordinal.
    This experiment uses contiguous character ordinal within each training
    window, never actual protected historical keys, and all writes observed.
    The default old read retrieves source-known preceding ordinal (t-1), not
    an oracle content address. Scores are next-character autoregressive.
    """
    def __init__(self, width=88,decoder=92,emb=48,post=10,position_mode="multiscale_relative",optimized_scan=True,gate_mode="silu"):
        super().__init__()
        assert emb==48 and width==88
        self.width=width;self.decoder=decoder;self.post=post
        if gate_mode not in ("silu","tanh"):raise ValueError("unknown gate mode")
        self.gate_mode=gate_mode
        if position_mode not in ("legacy", "multiscale", "multiscale_relative", "no_time"):
            raise ValueError("unknown position_mode")
        self.position_mode=position_mode; self.optimized_scan=optimized_scan
        self.embedding=nn.Embedding(V,emb)
        self.time_proj=nn.Linear(7,emb,bias=False)
        self.key_native=nn.Linear(emb,8)
        self.value_native=nn.Linear(emb,8)
        self.fast_map=nn.Linear(emb+width+8,2*width)
        self.slow_map=nn.Linear(2*width+8,2*width)
        self.native_gates=nn.Linear(emb+2*width,2)
        self.rotation=nn.Parameter(torch.stack((torch.eye(8),torch.roll(torch.eye(8),1,1)),0))
        self.source_content=nn.Linear(emb,16)
        self.source_router=nn.Linear(emb,2)
        self.source_retention=nn.Linear(emb,2)
        self.source_strength=nn.Linear(emb,2)
        self.source_silu_mod=nn.Linear(emb,2,bias=False)
        self.hist_project=nn.Linear(16,emb,bias=False)
        self.read_router=nn.Linear(8,2)
        # both PNDS native 8-dimensional and ordinal-aligned 16D read are used
        read_dim=2*width+8+emb+16
        self.norm=nn.LayerNorm(read_dim)
        self.read_in=nn.Linear(read_dim,decoder)
        self.read_out=nn.Linear(decoder,V)
        self.res_in=nn.Linear(read_dim,post)
        self.res_out=nn.Linear(post,V)
        self.res_logit=nn.Parameter(torch.tensor(-2.0))
        nn.init.zeros_(self.res_out.weight);nn.init.zeros_(self.res_out.bias)
        nn.init.zeros_(self.source_silu_mod.weight)
        with torch.no_grad():
            self.source_retention.bias.fill_(4.5)
            self.source_strength.bias.fill_(-1.5)

    def time_features(self,B,T,dtype,device,offset=0):
        pos=torch.arange(offset,offset+T,device=device,dtype=dtype).view(1,T).expand(B,T)
        # Uninterrupted text has dt=1. Prior admitted write was at t-1.
        ones=torch.ones_like(pos)
        if self.position_mode=="no_time":
            return torch.zeros(B,T,7,dtype=dtype,device=device)
        if self.position_mode=="legacy":
            return torch.stack((torch.log1p(pos),torch.sin(pos/16),torch.cos(pos/16),
                                math.log(2.0)*ones,torch.log1p(pos+1),
                                math.log(2.0)*ones,(pos>0).to(dtype)),dim=-1)
        # Genuine multi-timescale position, no pseudo-physical time clock.
        return torch.stack((torch.log1p(pos),torch.sin(pos/16),torch.cos(pos/16),
                            torch.sin(pos/64),torch.cos(pos/64),
                            torch.sin(pos/256),torch.cos(pos/256)),dim=-1)

    def keys(self, B,T, device,dtype, offset):
        index=torch.arange(offset,offset+T,device=device,dtype=torch.int64)
        phi=torch.stack([f((index.remainder(p)).to(dtype)*(2*math.pi/p))
               for p in PERIODS for f in (torch.cos,torch.sin)],-1)
        phi=phi[None].expand(B,-1,-1)
        k=F.normalize(torch.einsum('bti,hij->bthj',phi,self.rotation),dim=-1,eps=1e-6)
        return phi,k

    def forward(self,x,disable_native=False,disable_aligned=False,return_trace=False,offset=0):
        if x.ndim!=2 or x.dtype!=torch.int64: raise ValueError('tokens [B,T] int64 required')
        B,T=x.shape
        if T<1:raise ValueError('nonempty input')
        e=torch.tanh(self.embedding(x)+self.time_proj(self.time_features(B,T,self.embedding.weight.dtype,x.device,offset)))
        zeros_f=e.new_zeros(B,T,self.width);zeros_r=e.new_zeros(B,T,8)
        basis=self.fast_map(torch.cat([e,zeros_f,zeros_r],-1)).chunk(2,-1)[0].tanh()
        prior=F.pad(basis.cumsum(1)[:,:-1],(0,0,1,0))
        ctx=prior/torch.arange(1+offset,T+offset+1,device=x.device,dtype=e.dtype).clamp_min(1).view(1,T,1)
        # For offset=0, PNDS denominator max(count,1) at position t (0-index)
        if offset==0:ctx=prior/torch.arange(T,device=x.device,dtype=e.dtype).clamp_min(1).view(1,T,1)
        p,g=self.fast_map(torch.cat([e,ctx,zeros_r],-1)).chunk(2,-1)
        pf=torch.lerp(ctx,p.tanh(),g.sigmoid())
        d,g=self.slow_map(torch.cat([pf,ctx,zeros_r],-1)).chunk(2,-1)
        ps=torch.lerp(ctx,(ctx+d.tanh()).tanh(),.25*g.sigmoid())
        nk=F.normalize(self.key_native(e),dim=-1,eps=1e-6)
        nv=F.normalize(self.value_native(e),dim=-1,eps=1e-6)
        a,b=self.native_gates(torch.cat([e,pf,ps],-1)).sigmoid().chunk(2,-1)
        a=a*(1-b/4.)
        NM=one_scan(nk,nv,a,b)
        Nprev=F.pad(NM[:,:-1],(0,0,0,0,1,0))
        nr=(Nprev@nk.unsqueeze(-1)).squeeze(-1)
        if disable_native:nr=nr*0
        f,g=self.fast_map(torch.cat([e,pf,nr],-1)).chunk(2,-1)
        f=torch.lerp(pf,f.tanh(),g.sigmoid())
        d,g=self.slow_map(torch.cat([f,ps,nr],-1)).chunk(2,-1)
        s=torch.lerp(ps,(ps+d.tanh()).tanh(),.25*g.sigmoid())
        phi,source_keys=self.keys(B,T,x.device,e.dtype,offset)
        val=self.source_content(e).tanh().reshape(B,T,1,2,8).expand(-1,-1,2,-1,-1)
        route=self.source_router(e).softmax(-1)
        alpha=self.source_retention(e).sigmoid()
        raw=self.source_silu_mod(e)
        # The differentiable SiLU feature is a MODULATION of a bounded sigmoid gate.
        # Never let raw SiLU values become retention or write strengths.
        modifier=F.silu(2*raw) if self.gate_mode=="silu" else raw
        beta=torch.sigmoid(self.source_strength(e)+modifier)*route
        if self.optimized_scan:
            # Factor duplicate 8x8 P schedules shared across 16 value coordinates.
            kk=source_keys.permute(0,2,1,3).reshape(B*2,T,8)
            vv=self.source_content(e).tanh().unsqueeze(1).expand(B,2,T,16).reshape(B*2,T,16)
            aa=alpha.permute(0,2,1).reshape(B*2,T,1)
            bb=beta.permute(0,2,1).reshape(B*2,T,1)
            PP,QQ=rectangular_schedule(kk,vv,aa,bb)
            MM=rectangular_affine_scan(PP,QQ).reshape(B,2,T,16,8).permute(0,2,1,3,4)
        else:
            # Original four 8x8 scans, retained for exact implementation ablation.
            kk=source_keys.unsqueeze(-2).expand(-1,-1,-1,2,-1)
            aa=alpha[...,None,None].expand(-1,-1,-1,2,1)
            bb=beta[...,None,None].expand(-1,-1,-1,2,1)
            def flatten(z):return z.permute(0,2,3,1,4).reshape(B*4,T,z.shape[-1])
            MM=one_scan(flatten(kk),flatten(val),flatten(aa),flatten(bb))
            MM=MM.reshape(B,2,2,T,8,8).permute(0,3,1,2,4,5).reshape(B,T,2,16,8)
        before=F.pad(MM[:,:-1],(0,0,0,0,0,0,1,0))
        reread=self._aligned_read(before,e,nk,offset)
        if disable_aligned:reread=reread*0
        stacked=self.norm(torch.cat([f,s,nr,e,reread],-1))
        logits=self.read_out(F.silu(self.read_in(stacked))) + (
            self.res_logit.sigmoid()*self.res_out(F.silu(self.res_in(stacked))))
        if return_trace:
            return logits,dict(native=NM,aligned=MM,native_read=nr,aligned_read=reread,
                               native_key=nk,ordinal_key=source_keys,
                               fast=f,slow=s)
        return logits

    def historical_logits(self, tokens, trace, lag):
        """Read past ordinal from the PRE-WRITE memory, then predict its observed
        character. Target is only supplied to the separate training loss;
        no historical target is supplied to the read itself. No KV cache.
        """
        if not isinstance(lag,int) or lag<1:raise ValueError('lag must be positive integer')
        B,T=tokens.shape
        if lag>=T:raise ValueError('lag must be smaller than observed context')
        mem=trace['aligned']
        prev=F.pad(mem[:,:-1],(0,0,0,0,0,0,1,0))
        oldidx=(torch.arange(T,device=tokens.device,dtype=torch.int64)-lag).clamp_min(0)
        qp=torch.stack([f((oldidx.remainder(p)).to(mem.dtype)*(2*math.pi/p))
              for p in PERIODS for f in (torch.cos,torch.sin)],-1)[None].expand(B,-1,-1)
        key=F.normalize(torch.einsum('bti,hij->bthj',qp,self.rotation),dim=-1,eps=1e-6)
        read=torch.einsum('bthvk,bthk->bthv',prev,key)
        route=self.read_router(qp).softmax(-1)
        merged=(read*route.unsqueeze(-1)).sum(-2)
        # shared input embeddings as the historical classifier: 768 extra learned coordinates
        logits=F.linear(self.hist_project(merged[:,lag:]),self.embedding.weight)
        return logits, tokens[:,:-lag]

    def _aligned_read(self,before,e,nk,offset):
        """Read PREWRITE state. Relative offsets are queries, never sidecar values."""
        B,T=e.shape[:2]
        pos=torch.arange(offset,offset+T,device=e.device,dtype=torch.int64)
        lags=(1,8,32,128) if self.position_mode=="multiscale_relative" else (1,)
        heads=[];scores=[]
        for lag in lags:
            qidx=(pos-lag).clamp_min(0)
            qp=torch.stack([fn(qidx.remainder(p).to(e.dtype)*(2*math.pi/p))
                 for p in PERIODS for fn in (torch.cos,torch.sin)],-1)[None].expand(B,-1,-1)
            qk=F.normalize(torch.einsum('bti,hij->bthj',qp,self.rotation),dim=-1,eps=1e-6)
            hread=torch.einsum('bthvk,bthk->bthv',before,qk)
            rgate=self.read_router(qp).softmax(-1)
            reread=(hread*rgate.unsqueeze(-1)).sum(-2)
            valid=(pos>=lag).view(1,T,1)
            heads.append(torch.where(valid,reread,torch.zeros_like(reread)))
            # Input-only learned relevance using existing native address.
            # Previous ordinals are known at prediction time; future labels never enter.
            qscore=(nk*F.normalize(qp,dim=-1,eps=1e-6)).sum(-1)
            scores.append(qscore.masked_fill(~valid.squeeze(-1),-1e4))
        if len(lags)==1:return heads[0]
        weights=torch.stack(scores,-1).softmax(-1)
        return (torch.stack(heads,-2)*weights[...,None]).sum(-2)

    def _aligned_read_step(self,old,e,nk,index):
        B=e.shape[0]
        lags=(1,8,32,128) if self.position_mode=="multiscale_relative" else (1,)
        heads=[];scores=[]
        for lag in lags:
            qp=self.keys_raw_phi(B,e.device,e.dtype,max(0,index-lag))
            qk=F.normalize(torch.einsum('bi,hij->bhj',qp,self.rotation),dim=-1,eps=1e-6)
            hh=torch.einsum('bhvk,bhk->bhv',old,qk)
            rr=(hh*self.read_router(qp).softmax(-1).unsqueeze(-1)).sum(1)
            heads.append(rr if index>=lag else rr*0)
            scores.append((nk*F.normalize(qp,dim=-1,eps=1e-6)).sum(-1) if index>=lag else nk.new_full((B,),-1e4))
        if len(lags)==1:return heads[0]
        weights=torch.stack(scores,-1).softmax(-1)
        return (torch.stack(heads,1)*weights.unsqueeze(-1)).sum(1)

    def init_stream(self,batch:int,owner=None,device=None):
        p=self.embedding.weight
        if device is None:device=p.device
        if owner is None:owner=torch.arange(batch,device=device,dtype=torch.int64)
        if owner.shape!=(batch,) or owner.dtype!=torch.int64:raise ValueError('owner int64 [B]')
        return {'sum':p.new_zeros(batch,self.width),'native':p.new_zeros(batch,8,8),
                'aligned':p.new_zeros(batch,2,16,8),'count':0,'owner':owner.clone()}

    def stream_step(self,token,state,owner=None,disable_native=False,disable_aligned=False):
        if token.ndim!=1 or token.dtype!=torch.int64:raise ValueError('token [B] int64')
        B=token.shape[0];t=state['count']
        if owner is not None and not torch.equal(state['owner'],owner):
            raise ValueError('cross-owner reuse prohibited')
        device=token.device;dtype=self.embedding.weight.dtype
        time=self.time_features(B,1,dtype,device,t)[:,0]
        e=torch.tanh(self.embedding(token)+self.time_proj(time))
        zf=e.new_zeros(B,self.width);zr=e.new_zeros(B,8)
        basis=self.fast_map(torch.cat([e,zf,zr],-1)).chunk(2,-1)[0].tanh()
        ctx=state['sum']/max(t,1)
        p,g=self.fast_map(torch.cat([e,ctx,zr],-1)).chunk(2,-1)
        pf=torch.lerp(ctx,p.tanh(),g.sigmoid())
        d,g=self.slow_map(torch.cat([pf,ctx,zr],-1)).chunk(2,-1)
        ps=torch.lerp(ctx,(ctx+d.tanh()).tanh(),.25*g.sigmoid())
        nk=F.normalize(self.key_native(e),dim=-1,eps=1e-6)
        nv=F.normalize(self.value_native(e),dim=-1,eps=1e-6)
        a,b=self.native_gates(torch.cat([e,pf,ps],-1)).sigmoid().chunk(2,-1)
        a=a*(1-b/4.)
        nr=(state['native']@nk.unsqueeze(-1)).squeeze(-1)
        new_native=a[...,None]*state['native']+ (b*(nv-a*nr)).unsqueeze(-1)*nk.unsqueeze(-2)
        if disable_native:nr=nr*0
        f,g=self.fast_map(torch.cat([e,pf,nr],-1)).chunk(2,-1)
        f=torch.lerp(pf,f.tanh(),g.sigmoid())
        d,g=self.slow_map(torch.cat([f,ps,nr],-1)).chunk(2,-1)
        s=torch.lerp(ps,(ps+d.tanh()).tanh(),.25*g.sigmoid())
        _,sk=self.keys(B,1,device,dtype,t)
        sk=sk[:,0]
        val=self.source_content(e).tanh()
        route=self.source_router(e).softmax(-1)
        aa=self.source_retention(e).sigmoid()
        raw=self.source_silu_mod(e)
        modifier=F.silu(2*raw) if self.gate_mode=="silu" else raw
        bb=torch.sigmoid(self.source_strength(e)+modifier)*route
        old=state['aligned']
        old_projection=torch.einsum('bhvk,bhk->bhv',old,sk)
        inn=val[:,None]-aa[...,None]*old_projection
        new_aligned=aa[...,None,None]*old + (bb[...,None,None]*inn.unsqueeze(-1)*sk.unsqueeze(-2))
        reread=self._aligned_read_step(old,e,nk,t)
        if disable_aligned:reread=reread*0
        stacked=self.norm(torch.cat([f,s,nr,e,reread],-1))
        logits=self.read_out(F.silu(self.read_in(stacked)))+(self.res_logit.sigmoid()*self.res_out(F.silu(self.res_in(stacked))))
        after={'sum':state['sum']+basis,'native':new_native,'aligned':new_aligned,
               'count':t+1,'owner':state['owner']}
        return logits,after

    @staticmethod
    def keys_raw_phi(B,device,dtype,index):
        ix=torch.tensor(index,dtype=torch.int64,device=device)
        return torch.stack([fun(ix.remainder(p).to(dtype)*(2*math.pi/p))
            for p in PERIODS for fun in (torch.cos,torch.sin)],-1).reshape(1,8).expand(B,8)
