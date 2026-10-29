"""LANGSCALE075: source-transformed native associative 8/12/16 geometry.

Reset-window *training* candidate, not qualified carried state for k>8.
Original graph fast/slow projections, upper48 recurrence, FFN455, six CED
concepts, waves and source positions are unchanged. At k8 this module must
produce bitwise original CONTEXT026/LANGSCALE070 logits, weights and grads.

Native active block k x k inside original three-bank 16 x k scan; the
two aligned memories also expand their address width (16 x k).
The newly learned native read adapter k->8 pays additional parameters.
This is NOT a pure native-matrix-only change.
"""
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F

P=Path(__file__).resolve().parents[1]
for name in ("context026","langscale070"):
    path=str(P/name)
    if path not in sys.path:sys.path.insert(0,path)
from context_transition import affine_scan,scalar_affine_scan,last_write_memory,ordinal_code,phi_position
from scaled_recurrence import ScaledRecurrenceTAC070

WIDTHS={"m8":8,"m12":12,"m16":16}
COUNTS={"m8":499623,"m12":500175,"m16":500663}

def model_parameters(k):
    if type(k) is not int or k not in (8,12,16):
        raise ValueError("registered native address dimension only")
    delta=(k-8)*(49*2+16)+(0 if k==8 else 8*k)
    return 499623+delta

class NativeMatrixTAC075(ScaledRecurrenceTAC070):
    def __init__(self,tier="m8",vocab=256):
        if vocab!=256 or tier not in WIDTHS:
            raise ValueError("registered 256-byte native dimension only")
        super().__init__("p500k",vocab=vocab)
        self.tier=tier
        self.native_dim=WIDTHS[tier]
        k=self.native_dim
        if k==8:
            self.native_read_proj=nn.Identity()
        else:
            self.native_key=nn.Linear(48,k)
            self.native_value=nn.Linear(48,k)
            base=torch.eye(8,k)
            self.rotation=nn.Parameter(torch.stack((base,torch.roll(base,1,1))))
            self.native_read_proj=nn.Linear(k,8,bias=False)
        if self.parameter_count()!=COUNTS[tier] or model_parameters(k)!=COUNTS[tier]:
            raise ValueError("source/active model count mismatch")
        self.upper_state_width=48
        self.ffn_width=455
        self.upper_concept_graph=False

    def architecture_record(self):
        return {"study":"LANGSCALE075","tier":self.tier,
                "native_address_dim":self.native_dim,
                "native_matrix_shape":[self.native_dim,self.native_dim],
                "aligned_matrix_shape":[16,self.native_dim],
                "trainable_parameters":self.parameter_count(),
                "deployed_parameters":sum(p.numel() for p in self.parameters()),
                "upper_state_width":48,"ffn_width":455,
                "base_fast_slow_projection_widths":"88_AND_88_UNCHANGED",
                "whole_model_streaming_qualified":False,
                "protected_TEST":"NOT_ACCESSED"}

# The following is a source-mapped copy of original ContextTransition026.forward
# with explicit dimension substitutions and 8D native readout adapter.
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
        old=F.pad(mem[:,:-1],(0,0,0,0,0,0,1,0))
        native_old=self.native_read_proj(torch.einsum('btij,btj->bti',mem[:,:,0,:self.native_dim],key))
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
        if return_trace:return logits,dict(native_prewrite=old[:,:,0,:self.native_dim],native_memory=mem[:,:,0,:self.native_dim],native_read=native_old,native_query_key=key,native_write_key=write_key,native_value=value,native_alpha=alpha[...,0],native_beta=write[...,0],native_admission=bank_mask[...,0],aligned_prewrite=old[:,:,1:],wave_prewrite=oldwave,concept_prewrite=oldcp,concept_posterior=C,concept_laplacian=L,gate_beta=write,basis=ctx)
        return logits
