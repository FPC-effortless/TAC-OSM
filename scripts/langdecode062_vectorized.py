"""LANGDECODE062: experimental ordinal vectorization of trained TAC step.

Copy of pinned original IncrementalTAC043._advance, changing ONLY two
ordinal_code calls to one bitwise-verified vectorized paired computation.
All original network weights, state, source/clock, causal masks, snapshots
and rest of graph operations remain unchanged. Not production-promoted.
"""
import torch
from torch.nn import functional as F
from scripts.langdecode057_validation import BatchedValidationTAC057
from incremental import InferenceState, IncrementalTAC043, phi_position, transform
from scripts.langkernel061_vector_ordinal import vectorized_ordinal


class VectorOrdinalTAC062(BatchedValidationTAC057):
    """Retains original validated step/prefill/snapshot methods. The advance
    method is a reviewed source-level copy and MUST pass exact checkpoint parity.
    """
    def _advance(self,token,state,mask):
        m=self.model;s=state.tensors;b=token.shape[0];t=state.clock
        tt=torch.full((b,),t,dtype=torch.long,device=token.device)
        embed=m.embedding(token);lex=embed.tanh()
        e=(embed+m.pos(transform(phi_position(tt,embed.dtype),self.position_mode))).tanh()
        local=lex
        for lag in range(3): local=local+m.context_lag_weights[lag].tanh()*s['lexical'][:,lag]
        key=F.normalize(m.native_key(local.tanh()),dim=-1,eps=1e-6)
        wk=s['previous_key'];value=m.native_value(e).tanh();ab=m.native_gate(e).sigmoid()
        eye8=torch.eye(8,dtype=e.dtype,device=e.device)
        native_p=.995*ab[:,0,None,None]*(eye8-ab[:,1,None,None]*wk[:,:,None]*wk[:,None,:])
        native_q=ab[:,1,None,None]*value[:,:,None]*wk[:,None,:]
        native=torch.where((mask & (t>0))[:,None,None],s['native']@native_p+native_q,s['native'])
        native_read=(native@key[:,:,None]).squeeze(-1)
        # Same original periodic indices; one vectorized trig call, then
        # retain both EXACT original einsum/normalize execution orders.
        both=vectorized_ordinal(torch.stack((tt,(tt-1).clamp_min(0)),dim=0),e.dtype)
        akey=F.normalize(torch.einsum('bi,hij->bhj',both[0],m.rotation),dim=-1,eps=1e-6)
        pk=F.normalize(torch.einsum('bi,hij->bhj',both[1],m.rotation),dim=-1,eps=1e-6)
        ar=(s['aligned']@pk[:,:,:,None]).squeeze(-1).mean(1) if t else e.new_zeros((b,16))
        ag=m.aligned_gate(e).sigmoid().view(b,2,3);av=m.aligned_value(e).tanh()
        ap=ag[:,:,0,None,None]*(eye8-ag[:,:,1,None,None]*akey[:,:,:,None]*akey[:,:,None,:])
        aq=ag[:,:,2,None,None]*av[:,None,:,None]*akey[:,:,None,:]
        aligned=torch.where(mask[:,None,None,None],s['aligned']@ap+aq,s['aligned'])
        oldwave=s['wave'].flatten(1);ctx=s['basis']/max(1,t)
        fast=m.fast(torch.cat((e,ctx,oldwave,native_read),-1)).tanh()
        slow=m.slow(torch.cat((fast,ctx,native_read),-1)).tanh()
        G=m.evidence(torch.cat((e,fast,slow,ar,oldwave),-1)).tanh().view(b,6,16)
        q=m.concept_q(G);k=m.concept_k(G)
        sm=(q@k.transpose(-1,-2)/4).softmax(-1);eye6=torch.eye(6,dtype=e.dtype,device=e.device)
        A=.5*(sm+sm.transpose(-1,-2))*(1-eye6);L=torch.diag_embed(A.sum(-1))-A
        C=torch.linalg.solve(1.25*eye6+.25*L,G+.25*s['concepts'])
        pooled=(C*G.mean(-1).softmax(-1)[:,:,None]).sum(1)
        z=m.layernorm[0](torch.cat((fast,slow,native_read,ar,oldwave,pooled,e),-1))
        upper=m.layernorm[1]
        refined=upper.read_from_prior(z,s['upper']);next_upper=upper.observe(z,s['upper'])
        logits=m.ffn_out(m.ffn_up(refined)*F.silu(m.ffn_gate(refined)))+m.concept_out(pooled)
        angle=m.wave_angle;c=angle.cos();sn=angle.sin()
        R=torch.stack((torch.stack((c,-sn),-1),torch.stack((sn,c),-1)),-2)
        rho=.97+.029*m.wave_decay(e).sigmoid()
        wave=(s['wave'][:,:,None,:]@(rho[:,:,None,None]*R)).squeeze(-2)+.025*m.wave_input(e).tanh().view(b,8,2)
        cg=.05*m.concept_gate(e).sigmoid().view(b,6,16)
        concepts=(1-cg)*s['concepts']+cg*m.concept_value(e).tanh().view(b,6,16)
        tensors={'native':native,'aligned':aligned,'wave':wave,'concepts':concepts,
                 'basis':s['basis']+m.basis(e).tanh(),
                 'lexical':torch.stack((lex,s['lexical'][:,0],s['lexical'][:,1]),1),
                 'previous_key':key,'upper':next_upper,'last_logits':logits.clone()}
        return logits,InferenceState(state.owners,t+1,self.weights_sha256,self.position_mode,self.implementation_sha256,tensors)

