"""CONTEXT026: UNTRAINED 500122p causal contextual-transition TAC candidate.

Separate from the original 100,680-parameter checkpoint. Mathematical prototype:
  - native 8x8, two 16x8 source-aligned heads, eight 2D damped waves,
    88-dim basis sum and six 16D persistent concept seeds = 520 floats.
  - independent causal input-only scan schedules and source admission masks.
  - postscan 6-concept quadratic equilibrium followed by a large live SwiGLU
    language head. Does NOT assert active inference or retrieval skill.
Inherited520-float summaries exclude any future context cache/temporary scans.
This candidate is window-reset only; no streaming/RAM/speed qualification.
All source index and target documents are caller-provenance responsibilities.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F

PERIODS=(7,13,29,59)

def affine_scan(p,q):
    """Inclusive chronological right-affine scan: S_next=S@P+Q.
    p: [B,T,D,D], q:[B,T,R,D], identity for nonwrites.
    Parallel span O(log T); here Hillis-Steele O(T log T) *work*.
    """
    if p.ndim!=4 or q.ndim!=4 or p.shape[:2]!=q.shape[:2] or p.shape[-1]!=p.shape[-2] or q.shape[-1]!=p.shape[-1]:
        raise ValueError('bad affine scan geometry')
    t=p.shape[1];gap=1
    while gap<t:
        mp=p[:,:-gap]@p[:,gap:]
        mq=q[:,:-gap]@p[:,gap:]+q[:,gap:]
        p=torch.cat((p[:,:gap],mp),dim=1)
        q=torch.cat((q[:,:gap],mq),dim=1)
        gap*=2
    return q

def scalar_affine_scan(p,q):
    """Componentwise causal affine stream C_next=P*C+Q."""
    t=p.shape[1];gap=1
    while gap<t:
        mp=p[:,:-gap]*p[:,gap:]
        mq=q[:,:-gap]*p[:,gap:]+q[:,gap:]
        p=torch.cat((p[:,:gap],mp),dim=1)
        q=torch.cat((q[:,:gap],mq),dim=1)
        gap*=2
    return q

def phi_position(t,dtype):
    x=t.to(dtype=dtype)
    return torch.stack((torch.log1p(x),torch.sin(x/16),torch.cos(x/16),
        torch.sin(x/64),torch.cos(x/64),torch.sin(x/256),torch.cos(x/256)),dim=-1)

def ordinal_code(indices,dtype):
    xs=[]
    for p in PERIODS:
        angle=indices.remainder(p).to(dtype)*(2*math.pi/p)
        xs.extend((angle.cos(),angle.sin()))
    return torch.stack(xs,dim=-1)

def last_write_memory(inclusive,mask):
    """Gathers actual last admitted prefix; preserves nonwrite bytes."""
    b,t=inclusive.shape[:2]
    idx=torch.arange(t,device=mask.device).view(1,t).expand(b,t)
    last=torch.where(mask,idx,torch.full_like(idx,-1)).cummax(1).values
    ind=last.clamp_min(0).view(b,t,*([1]*(inclusive.ndim-2))).expand_as(inclusive)
    found=(last>=0).view(b,t,*([1]*(inclusive.ndim-2)))
    return torch.where(found,inclusive.gather(1,ind),torch.zeros_like(inclusive))

class ContextTransition026(nn.Module):
    def __init__(self,vocab=256,ffn_width=501):
        super().__init__()
        self.vocab=vocab
        self.embedding=nn.Embedding(vocab,48)
        self.pos=nn.Linear(7,48,bias=False)
        self.basis=nn.Linear(48,88)
        self.context_lag_weights=nn.Parameter(torch.full((3,48),0.1))
        self.native_key=nn.Linear(48,8)
        self.native_value=nn.Linear(48,8)
        self.native_gate=nn.Linear(48,2)
        self.aligned_value=nn.Linear(48,16)
        self.aligned_gate=nn.Linear(48,6)
        self.rotation=nn.Parameter(torch.stack((torch.eye(8),torch.roll(torch.eye(8),1,1))))
        self.wave_decay=nn.Linear(48,8)
        self.wave_input=nn.Linear(48,16)
        self.wave_angle=nn.Parameter(torch.linspace(.02,.37,8))
        self.concept_value=nn.Linear(48,96)
        self.concept_gate=nn.Linear(48,96)
        self.fast=nn.Linear(48+88+16+8,88)
        self.slow=nn.Linear(88+88+8,88)
        self.evidence=nn.Linear(48+88+88+16+16,96)
        self.concept_q=nn.Linear(16,16,bias=False)
        self.concept_k=nn.Linear(16,16,bias=False)
        self.layernorm=nn.LayerNorm(280)
        self.ffn_up=nn.Linear(280,ffn_width)
        self.ffn_gate=nn.Linear(280,ffn_width)
        self.ffn_out=nn.Linear(ffn_width,vocab)
        self.concept_out=nn.Linear(16,vocab,bias=False)

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self,tokens,*,start=0,write_mask=None,return_trace=False):
        if tokens.ndim!=2 or tokens.dtype!=torch.long or tokens.size(1)==0:
            raise ValueError('token tensor must be [B,T] of int64')
        if type(start) is not int or start!=0:raise ValueError("context026 requires reset source windows; no carried-state interface")
        b,t=tokens.shape
        if tokens.min()<0 or tokens.max()>=self.vocab:raise ValueError('token out of range')
        if write_mask is None:write_mask=torch.ones(b,t,dtype=torch.bool,device=tokens.device)
        if write_mask.shape!=(b,t) or write_mask.dtype!=torch.bool:raise ValueError('invalid admission mask')
        tt=torch.arange(start,start+t,device=tokens.device).view(1,t).expand(b,t)
        e=torch.tanh(self.embedding(tokens)+self.pos(phi_position(tt,self.embedding.weight.dtype)))
        basis=torch.tanh(self.basis(e))
        prev_basis=F.pad(basis.cumsum(1)[:,:-1],(0,0,1,0))
        ctx=prev_basis / tt.clamp_min(1).unsqueeze(-1)
        # Keys depend on observed lexical context, never labels or future tokens.
        lexical=torch.tanh(self.embedding(tokens))
        local=lexical
        for lag in range(1,4):
            past=F.pad(lexical[:,:-lag],(0,0,lag,0))[:,:t] if lag<t else torch.zeros_like(lexical)
            local=local+torch.tanh(self.context_lag_weights[lag-1])*past
        key=F.normalize(self.native_key(torch.tanh(local)),dim=-1,eps=1e-6)
        write_key=F.pad(key[:,:-1],(0,0,1,0))
        value=torch.tanh(self.native_value(e))
        phi=ordinal_code(tt,e.dtype)
        aligned_key=F.normalize(torch.einsum('bti,hij->bthj',phi,self.rotation),dim=-1,eps=1e-6)
        alphabeta=self.native_gate(e).sigmoid()
        aligned_gates=self.aligned_gate(e).sigmoid().view(b,t,2,3)
        # 8x8 native + 2x16x8 aligned rectangular memories as B*3 scan.
        allkey=torch.cat((write_key.unsqueeze(2),aligned_key),dim=2)
        alpha=torch.cat((.995*alphabeta[...,0:1],aligned_gates[...,0]),dim=-1)
        erase=torch.cat((alphabeta[...,1:2],aligned_gates[...,1]),dim=-1)
        write=torch.cat((alphabeta[...,1:2],aligned_gates[...,2]),dim=-1)
        allvalue=torch.cat((F.pad(value,(0,8)).unsqueeze(2),
                            torch.tanh(self.aligned_value(e)).unsqueeze(2).expand(-1,-1,2,-1)),dim=2)
        eye=torch.eye(8,dtype=e.dtype,device=e.device)
        P=alpha[...,None,None]*(eye-erase[...,None,None]*allkey.unsqueeze(-1)*allkey.unsqueeze(-2))
        Q=write[...,None,None]*allvalue.unsqueeze(-1)*allkey.unsqueeze(-2)
        bank_mask=write_mask[:,:,None].expand(-1,-1,3).clone()
        bank_mask[:,0,0]=False  # no predecessor context for the first native pair
        P=torch.where(bank_mask[...,None,None],P,eye)
        Q=torch.where(bank_mask[...,None,None],Q,torch.zeros_like(Q))
        mem=affine_scan(P.permute(0,2,1,3,4).reshape(b*3,t,8,8),
                        Q.permute(0,2,1,3,4).reshape(b*3,t,16,8))
        mem=mem.reshape(b,3,t,16,8).permute(0,2,1,3,4)
        # Gather each bank's actual admitted prefix, preserving native t0/nonwrites.
        mem=last_write_memory(mem.permute(0,2,1,3,4).reshape(b*3,t,16,8),
            bank_mask.permute(0,2,1).reshape(b*3,t)).reshape(b,3,t,16,8).permute(0,2,1,3,4)
        old=F.pad(mem[:,:-1],(0,0,0,0,0,0,1,0))
        native_old=torch.einsum('btij,btj->bti',mem[:,:,0,:8],key)
        # Source ordinal read before current write; no truth labels or historic content passed.
        previous=ordinal_code((tt-1).clamp_min(0),e.dtype)
        previous_key=F.normalize(torch.einsum('bti,hij->bthj',previous,self.rotation),dim=-1,eps=1e-6)
        aligned_read=torch.einsum('bthvk,bthk->bthv',old[:,:,1:],previous_key).mean(2)
        aligned_read=torch.where((tt>0).unsqueeze(-1),aligned_read,torch.zeros_like(aligned_read))
        # Wave modes are independent 2x2 right-affine input-only matrices.
        angle=self.wave_angle.to(dtype=e.dtype)
        c=angle.cos().view(1,1,8);s=angle.sin().view(1,1,8)
        rho=(.97+.029*self.wave_decay(e).sigmoid())
        R=torch.stack((torch.stack((c.expand(b,t,-1),-s.expand(b,t,-1)),-1),
                        torch.stack((s.expand(b,t,-1),c.expand(b,t,-1)),-1)),-2)
        p_wave=rho[...,None,None]*R
        q_wave=.025*torch.tanh(self.wave_input(e)).view(b,t,8,1,2)
        wave=affine_scan(p_wave.permute(0,2,1,3,4).reshape(b*8,t,2,2),
                         q_wave.permute(0,2,1,3,4).reshape(b*8,t,1,2))
        wave=wave.view(b,8,t,2).permute(0,2,1,3)
        oldwave=F.pad(wave[:,:-1],(0,0,0,0,1,0)).flatten(-2)
        # Persistent six concept seeds have a scanable input-only coordinate recurrence.
        cg=.05*self.concept_gate(e).sigmoid()
        offer=torch.tanh(self.concept_value(e))
        cp=scalar_affine_scan(1-cg,cg*offer).view(b,t,6,16)
        oldcp=F.pad(cp[:,:-1],(0,0,0,0,1,0))
        f=torch.tanh(self.fast(torch.cat((e,ctx,oldwave,native_old),-1)))
        slow=torch.tanh(self.slow(torch.cat((f,ctx,native_old),-1)))
        G=torch.tanh(self.evidence(torch.cat((e,f,slow,aligned_read,oldwave),-1))).view(b,t,6,16)
        q=self.concept_q(G);k=self.concept_k(G)
        sm=torch.softmax(q@k.transpose(-1,-2)/4,-1)
        A=.5*(sm+sm.transpose(-1,-2))
        A=A*(1-torch.eye(6,dtype=A.dtype,device=A.device))
        L=torch.diag_embed(A.sum(-1))-A
        eye6=torch.eye(6,dtype=e.dtype,device=e.device)
        H=1.25*eye6+.25*L
        C=torch.linalg.solve(H,G+.25*oldcp)
        pooled=(C*torch.softmax(G.mean(-1),dim=-1).unsqueeze(-1)).sum(dim=-2)
        z=self.layernorm(torch.cat((f,slow,native_old,aligned_read,oldwave,pooled,e),-1))
        logits=self.ffn_out(self.ffn_up(z)*F.silu(self.ffn_gate(z)))+self.concept_out(pooled)
        if return_trace:return logits,dict(native_prewrite=old[:,:,0,:8],native_memory=mem[:,:,0,:8],native_read=native_old,native_query_key=key,native_write_key=write_key,native_value=value,native_alpha=alpha[...,0],native_beta=write[...,0],native_admission=bank_mask[...,0],aligned_prewrite=old[:,:,1:],wave_prewrite=oldwave,concept_prewrite=oldcp,concept_posterior=C,concept_laplacian=L,gate_beta=write,basis=ctx)
        return logits
