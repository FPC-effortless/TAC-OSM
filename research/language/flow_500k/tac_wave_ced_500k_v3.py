"""TAC-LANG-MEM-500K-007 v3: dual-timescale, owner-bound source retrieval candidate.

Separate from the original 100,680-parameter checkpoint. Mathematical prototype:
  - native 8x8, two 16x8 source-aligned heads, eight 2D damped waves,
    88-dim basis sum and six 16D persistent concept seeds = 520 floats.
  - independent causal input-only scan schedules and source admission masks.
  - postscan 6-concept quadratic equilibrium followed by a large live SwiGLU
    language head. Does NOT assert active inference or retrieval skill.
All source index and target documents are caller-provenance responsibilities.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F

PERIODS=(7,13,29,59)

def affine_scan(p,q,return_pairs=False):
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
    return (p,q) if return_pairs else q

def scalar_affine_scan(p,q,return_pairs=False):
    """Componentwise causal affine stream C_next=P*C+Q."""
    t=p.shape[1];gap=1
    while gap<t:
        mp=p[:,:-gap]*p[:,gap:]
        mq=q[:,:-gap]*p[:,gap:]+q[:,gap:]
        p=torch.cat((p[:,:gap],mp),dim=1)
        q=torch.cat((q[:,:gap],mq),dim=1)
        gap*=2
    return (p,q) if return_pairs else q

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

class WaveCED500KSlow(nn.Module):
    def __init__(self,vocab=128,ffn_width=605):
        super().__init__()
        self.vocab=vocab
        self.embedding=nn.Embedding(vocab,48)
        self.pos=nn.Linear(7,48,bias=False)
        self.basis=nn.Linear(48,88)
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

    def forward(self,tokens,*,start=0,write_mask=None,return_trace=False,
                initial_state=None,return_state=False,graph_scale=1.0,
                source_ids=None,query_ordinals=None):
        if tokens.ndim!=2 or tokens.dtype!=torch.long or tokens.size(1)==0:
            raise ValueError('token tensor must be [B,T] of int64')
        if type(start) is not int or start<0:
            raise ValueError('absolute source ordinal start must be nonnegative integer')
        if not isinstance(graph_scale,(float,int)) or type(graph_scale) is bool or not math.isfinite(graph_scale) or not (0<=graph_scale<=1):
            raise ValueError('graph_scale must be finite in [0,1] (diagnostic intervention)')
        b,t=tokens.shape
        if source_ids is not None:
            if type(source_ids) is not tuple or len(source_ids)!=b or not all(
               type(x) is str and 0<len(x)<=256 for x in source_ids):
                raise ValueError('source_ids must be a tuple of nonempty per-row owner identities')
        if initial_state is not None and (source_ids is None or
            not isinstance(initial_state,dict) or initial_state.get('source_ids')!=source_ids):
            raise ValueError('carried state requires matching nonempty source identities')
        if query_ordinals is not None:
            if source_ids is None or not isinstance(query_ordinals,torch.Tensor) or (
               query_ordinals.dtype != torch.long or query_ordinals.shape != (b,t) or
               query_ordinals.device != tokens.device):
                raise ValueError('owner-bound query ordinals must have int64 shape [B,T]')
        if tokens.min()<0 or tokens.max()>=self.vocab:raise ValueError('token out of range')
        if write_mask is None:write_mask=torch.ones(b,t,dtype=torch.bool,device=tokens.device)
        if write_mask.shape!=(b,t) or write_mask.dtype!=torch.bool:raise ValueError('invalid admission mask')
        if initial_state is None:
            if start != 0:
                raise ValueError('nonzero ordinal requires carried state')
            basis_initial=self.embedding.weight.new_zeros((b,88))
            native_initial=self.embedding.weight.new_zeros((b,8,8))
            aligned_initial=self.embedding.weight.new_zeros((b,2,16,8))
            wave_initial=self.embedding.weight.new_zeros((b,8,2))
            concept_initial=self.embedding.weight.new_zeros((b,6,16))
        else:
            if type(initial_state.get('count')) is not int or initial_state['count'] != start:
                raise ValueError('carried ordinal count mismatch')
            expected={'basis':(b,88),'native':(b,8,8),'aligned':(b,2,16,8),
                      'waves':(b,8,2),'concepts':(b,6,16)}
            for k,shape in expected.items():
                v=initial_state.get(k)
                if not isinstance(v,torch.Tensor) or tuple(v.shape)!=shape or v.device!=tokens.device or v.dtype!=self.embedding.weight.dtype:
                    raise ValueError('carried state mismatch for '+k)
            basis_initial=initial_state['basis'];native_initial=initial_state['native']
            aligned_initial=initial_state['aligned'];wave_initial=initial_state['waves']
            concept_initial=initial_state['concepts']
            # The bounded-write invariant assumes finite and initially
            # admissible state; reject corrupted/legacy overlarge carries.
            if not all(bool(torch.isfinite(initial_state[name]).all())
                       for name in expected):
                raise ValueError('nonfinite carried state')
            if bool((native_initial.norm(dim=(-2,-1)) > math.sqrt(8)+1e-3).any()) or bool(
                (aligned_initial.norm(dim=(-2,-1)) > 4+1e-3).any()):
                raise ValueError('carried associative memory exceeds v2 bound')
        tt=torch.arange(start,start+t,device=tokens.device).view(1,t).expand(b,t)
        e=torch.tanh(self.embedding(tokens)+self.pos(phi_position(tt,self.embedding.weight.dtype)))
        basis=torch.tanh(self.basis(e))
        prev_basis=F.pad(basis.cumsum(1)[:,:-1],(0,0,1,0))+basis_initial[:,None,:]
        ctx=prev_basis / tt.clamp_min(1).unsqueeze(-1)
        # Input-only source-known keys, normalized with epsilon.
        key=F.normalize(self.native_key(e),dim=-1,eps=1e-6)
        value=torch.tanh(self.native_value(e))
        phi=ordinal_code(tt,e.dtype)
        aligned_key=F.normalize(torch.einsum('bti,hij->bthj',phi,self.rotation),dim=-1,eps=1e-6)
        alphabeta=self.native_gate(e).sigmoid()
        aligned_gates=self.aligned_gate(e).sigmoid().view(b,t,2,3)
        # 8x8 native + 2x16x8 aligned rectangular memories as B*3 scan.
        allkey=torch.cat((key.unsqueeze(2),aligned_key),dim=2)
        alpha=torch.cat((alphabeta[...,0:1],aligned_gates[...,0]),dim=-1)
        erase=torch.cat((alphabeta[...,1:2],aligned_gates[...,1]),dim=-1)
        # Bank 1 stays fast. Bank 2 is a source-indexed slow associative
        # store: rho in [.995,.999] and low per-write key-directed erasure.
        # This is a mathematical memory-mechanics candidate, NOT established
        # long-term retrieval skill. Retains 499290 learned parameters.
        alpha=torch.cat((alpha[...,:2], .995+.004*alpha[...,2:3]),dim=-1)
        erase=torch.cat((erase[...,:2], .002*erase[...,2:3]),dim=-1)
        # v2 stability law: write<=1-alpha; Q cannot accrue unboundedly
        # when ||value|| and ||key|| are bounded. No new trainable weights.
        write_raw=torch.cat((alphabeta[...,1:2],aligned_gates[...,2]),dim=-1)
        write=(1-alpha)*write_raw
        allvalue=torch.cat((F.pad(value,(0,8)).unsqueeze(2),
                            torch.tanh(self.aligned_value(e)).unsqueeze(2).expand(-1,-1,2,-1)),dim=2)
        eye=torch.eye(8,dtype=e.dtype,device=e.device)
        P=alpha[...,None,None]*(eye-erase[...,None,None]*allkey.unsqueeze(-1)*allkey.unsqueeze(-2))
        Q=write[...,None,None]*allvalue.unsqueeze(-1)*allkey.unsqueeze(-2)
        P=torch.where(write_mask[...,None,None,None],P,eye)
        Q=torch.where(write_mask[...,None,None,None],Q,torch.zeros_like(Q))
        prefixP,prefixQ=affine_scan(P.permute(0,2,1,3,4).reshape(b*3,t,8,8),
                        Q.permute(0,2,1,3,4).reshape(b*3,t,16,8),return_pairs=True)
        memory_initial=torch.cat((F.pad(native_initial,(0,0,0,8)).unsqueeze(1),aligned_initial),dim=1)
        mem_flat=memory_initial.reshape(b*3,16,8)[:,None,:,:]@prefixP+prefixQ
        mem=mem_flat.reshape(b,3,t,16,8).permute(0,2,1,3,4)
        old=torch.cat((memory_initial[:,None],mem[:,:-1]),dim=1)
        native_old=torch.einsum('btij,btj->bti',old[:,:,0,:8],key)
        # Source ordinal read before current write; no truth labels or historic content passed.
        if query_ordinals is None:
            query_index=(tt-1).clamp_min(0)
        else:
            valid=(((query_ordinals == -1) & (tt == 0)) |
                   ((query_ordinals >= 0) & (query_ordinals < tt)))
            if not bool(valid.all()):
                raise ValueError('query must reference an earlier source ordinal in the same owner stream')
            query_index=query_ordinals.clamp_min(0)
        requested=ordinal_code(query_index,e.dtype)
        requested_key=F.normalize(torch.einsum('bti,hij->bthj',requested,self.rotation),dim=-1,eps=1e-6)
        per_bank=torch.einsum('bthvk,bthk->bthv',old[:,:,1:],requested_key)
        # Fixed signal gain compensates for bounded slow-bank writes: it
        # cannot restore information that never survived or was overwritten.
        aligned_read=.5*(per_bank[:,:,0]+32*per_bank[:,:,1])
        aligned_read=torch.where((tt>0).unsqueeze(-1),aligned_read,torch.zeros_like(aligned_read))
        # Wave modes are independent 2x2 right-affine input-only matrices.
        angle=self.wave_angle.to(dtype=e.dtype)
        c=angle.cos().view(1,1,8);s=angle.sin().view(1,1,8)
        rho=(.97+.029*self.wave_decay(e).sigmoid())
        R=torch.stack((torch.stack((c.expand(b,t,-1),-s.expand(b,t,-1)),-1),
                        torch.stack((s.expand(b,t,-1),c.expand(b,t,-1)),-1)),-2)
        p_wave=rho[...,None,None]*R
        q_wave=.025*torch.tanh(self.wave_input(e)).view(b,t,8,1,2)
        waveP,waveQ=affine_scan(p_wave.permute(0,2,1,3,4).reshape(b*8,t,2,2),
                         q_wave.permute(0,2,1,3,4).reshape(b*8,t,1,2),return_pairs=True)
        waveflat=wave_initial.reshape(b*8,1,2)[:,None,:,:]@waveP+waveQ
        wave=waveflat.view(b,8,t,2).permute(0,2,1,3)
        oldwave=torch.cat((wave_initial[:,None],wave[:,:-1]),dim=1).flatten(-2)
        # Persistent six concept seeds have a scanable input-only coordinate recurrence.
        cg=.05*self.concept_gate(e).sigmoid()
        offer=torch.tanh(self.concept_value(e))
        cpP,cpQ=scalar_affine_scan(1-cg,cg*offer,return_pairs=True)
        cp=(concept_initial.flatten(-2)[:,None,:]*cpP+cpQ).view(b,t,6,16)
        oldcp=torch.cat((concept_initial[:,None],cp[:,:-1]),dim=1)
        f=torch.tanh(self.fast(torch.cat((e,ctx,oldwave,native_old),-1)))
        slow=torch.tanh(self.slow(torch.cat((f,ctx,native_old),-1)))
        G=torch.tanh(self.evidence(torch.cat((e,f,slow,aligned_read,oldwave),-1))).view(b,t,6,16)
        q=self.concept_q(G);k=self.concept_k(G)
        sm=torch.softmax(q@k.transpose(-1,-2)/4,-1)
        A=.5*(sm+sm.transpose(-1,-2))
        A=A*(1-torch.eye(6,dtype=A.dtype,device=A.device))
        L=torch.diag_embed(A.sum(-1))-A
        eye6=torch.eye(6,dtype=e.dtype,device=e.device)
        H=1.25*eye6+(.25*graph_scale)*L
        C=torch.linalg.solve(H,G+.25*oldcp)
        # Old unweighted mean annihilated EVERY symmetric Laplacian:
        # 1^T L=0, so concept_q/k had zero functional path to logits.
        # This slot-specific, source-content-aware query is deliberately
        # non-uniform. It exposes cross-slot graph transfer to both logits
        # and input gradients without extra learned parameters.
        slot_prior=torch.linspace(-1.5,1.5,6,device=e.device,dtype=e.dtype)
        slot_scores=(G*oldcp).sum(-1)/4+slot_prior
        slot_weights=F.softmax(slot_scores,dim=-1)
        pooled=(slot_weights.unsqueeze(-1)*C).sum(dim=-2)
        z=self.layernorm(torch.cat((f,slow,native_old,aligned_read,oldwave,pooled,e),-1))
        logits=self.ffn_out(self.ffn_up(z)*F.silu(self.ffn_gate(z)))+self.concept_out(pooled)
        new_state=dict(count=start+t,source_ids=source_ids,basis=basis_initial+basis.sum(1),
            native=mem[:,-1,0,:8,:],aligned=mem[:,-1,1:,:,:],
            waves=wave[:,-1,:,:],concepts=cp[:,-1,:,:])
        if return_trace:
            trace=dict(native_prewrite=old[:,:,0,:8],aligned_prewrite=old[:,:,1:],
                wave_prewrite=oldwave,concept_prewrite=oldcp,concept_posterior=C,
                concept_laplacian=L,concept_read_weights=slot_weights,concept_evidence=G,gate_beta=write,gate_alpha=alpha,gate_erase=erase,basis=ctx)
            return (logits,trace,new_state) if return_state else (logits,trace)
        return (logits,new_state) if return_state else logits
