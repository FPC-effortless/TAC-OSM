"""LANGMEM078 dual-timescale native associative candidate, original SOURCE-derived.

Original model exactly LANGSCALE075 m8 (499623 learned weights) plus FOUR
active learnable scalars controlling a second k8x8 slow matrix:
 * slow_log_decay: rho=exp(-softplus(raw_rate) * Δbyte), Δbyte=1
 * slow_erase_logit: beta_s=.05 sigmoid(raw)
 * slow_write_logit: eps_s=.01 sigmoid(raw)
 * slow_read_logit: add sigmoid(raw)*slow_read to original 8D native read
Original fast native source, aligned/wave/concepts/upper48 and FFN455
weights have exact same names/shapes, same original seed initialization.
The first 8x8 fast source is unchanged; slow lane is a mathematically
bounded input-only affine scan. This code supports reset contexts, NOT yet
qualified true carried new-lane streaming. Neither proof of old-content
retrieval nor an experimentally tested trained improvement.
"""
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F

P=Path(__file__).resolve().parents[1]
for folder in ("langscale075","context026","langscale070","langlayer038"):
    path=str(P/folder)
    if path not in sys.path: sys.path.insert(0,path)
from native_matrix import NativeMatrixTAC075
from context_transition import affine_scan,last_write_memory,scalar_affine_scan,ordinal_code,phi_position

BASE_PARAMETERS=499623
DUAL_PARAMETERS=499627
class DualNativeTimescaleTAC078(NativeMatrixTAC075):
    def __init__(self,vocab=256):
        super().__init__("m8",vocab=vocab)
        self.slow_log_decay=nn.Parameter(torch.tensor(-4.5,dtype=torch.float32))
        self.slow_erase_logit=nn.Parameter(torch.tensor(-1.0,dtype=torch.float32))
        self.slow_write_logit=nn.Parameter(torch.tensor(-1.0,dtype=torch.float32))
        self.slow_read_logit=nn.Parameter(torch.tensor(-2.0,dtype=torch.float32))
        if self.parameter_count()!=DUAL_PARAMETERS:
            raise ValueError("unregistered actual dual-lane active parameter count")

    def registered_geometry(self):
        return {"study":"LANGMEM078","fast_native_shape":[8,8],
          "slow_native_shape":[8,8],"original_aligned_shapes":[2,16,8],
          "fast_slow_mlp_outputs":"88/88_UNCHANGED",
          "learned_parameters":self.parameter_count(),"original_base_parameters":BASE_PARAMETERS,
          "upper_recurrent_width":48,"ffn_width":455,
          "slow_one_source_byte_dt":1,
          "whole_model_true_carried_streaming":"NOT_QUALIFIED",
          "protected_TEST":"NOT_ACCESSED"}

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
        allvalue=torch.cat((F.pad(value,(0,16-self.native_dim)).unsqueeze(2),
                            torch.tanh(self.aligned_value(e)).unsqueeze(2).expand(-1,-1,2,-1)),dim=2)
        eye=torch.eye(self.native_dim,dtype=e.dtype,device=e.device)
        P=alpha[...,None,None]*(eye-erase[...,None,None]*allkey.unsqueeze(-1)*allkey.unsqueeze(-2))
        Q=write[...,None,None]*allvalue.unsqueeze(-1)*allkey.unsqueeze(-2)
        bank_mask=write_mask[:,:,None].expand(-1,-1,3).clone()
        bank_mask[:,0,0]=False  # no predecessor context for the first native pair
        P=torch.where(bank_mask[...,None,None],P,eye)
        Q=torch.where(bank_mask[...,None,None],Q,torch.zeros_like(Q))
        mem=affine_scan(P.permute(0,2,1,3,4).reshape(b*3,t,self.native_dim,self.native_dim),
                        Q.permute(0,2,1,3,4).reshape(b*3,t,16,self.native_dim))
        mem=mem.reshape(b,3,t,16,self.native_dim).permute(0,2,1,3,4)
        # Gather each bank's actual admitted prefix, preserving native t0/nonwrites.
        mem=last_write_memory(mem.permute(0,2,1,3,4).reshape(b*3,t,16,self.native_dim),
            bank_mask.permute(0,2,1).reshape(b*3,t)).reshape(b,3,t,16,self.native_dim).permute(0,2,1,3,4)
        # New independently persisted, bounded 8x8 *slow* native-associative lane.
        # Per observed source byte Δt=1 (this is NOT wall-clock elapsed time).
        slow_rho=torch.exp(-F.softplus(self.slow_log_decay))*torch.ones_like(alpha[...,0])
        slow_beta=(.05*torch.sigmoid(self.slow_erase_logit))*torch.ones_like(slow_rho)
        slow_write=(.01*torch.sigmoid(self.slow_write_logit))*torch.ones_like(slow_rho)
        slow_p=slow_rho[...,None,None]*(eye-slow_beta[...,None,None]*
                    write_key[...,None]*write_key[...,None,:])
        slow_q=slow_write[...,None,None]*value[...,None]*write_key[...,None,:]
        slow_admit=write_mask.clone()
        slow_admit[:,0]=False  # exactly original native first-byte pairing convention
        slow_p=torch.where(slow_admit[...,None,None],slow_p,eye)
        slow_q=torch.where(slow_admit[...,None,None],slow_q,torch.zeros_like(slow_q))
        slow_mem=affine_scan(slow_p,slow_q)
        slow_mem=last_write_memory(slow_mem,slow_admit)
        slow_read=torch.einsum('btij,btj->bti',slow_mem,key)
        old=F.pad(mem[:,:-1],(0,0,0,0,0,0,1,0))
        native_old=self.native_read_proj(torch.einsum('btij,btj->bti',mem[:,:,0,:self.native_dim],key))
        native_old=native_old+torch.sigmoid(self.slow_read_logit)*slow_read
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
        if return_trace:return logits,dict(slow_memory=slow_mem,slow_read=slow_read,slow_alpha=slow_rho,slow_beta=slow_beta,slow_admission=slow_admit,native_prewrite=old[:,:,0,:self.native_dim],native_memory=mem[:,:,0,:self.native_dim],native_read=native_old,native_query_key=key,native_write_key=write_key,native_value=value,native_alpha=alpha[...,0],native_beta=write[...,0],native_admission=bank_mask[...,0],aligned_prewrite=old[:,:,1:],wave_prewrite=oldwave,concept_prewrite=oldcp,concept_posterior=C,concept_laplacian=L,gate_beta=write,basis=ctx)
        return logits
