"""Inference-only incremental execution of frozen040 graph-free48D TAC.

Same growing-prefix semantics as the original batch graph, within qualified
floating-point tolerances. Not rolling-window eviction, retraining, long-memory
qualification or an IEEE/AD proof. Original source/checkpoints are untouched.
"""
from dataclasses import dataclass,replace
import hashlib,json
from pathlib import Path
import sys
import torch
from torch.nn import functional as F

P=Path(__file__).resolve().parents[1]
for name in ('langlayer040','langmemory042'):
    if str(P/name) not in sys.path: sys.path.insert(0,str(P/name))
from factorial_depth import FactorialDepthControl040
from context_transition import phi_position,ordinal_code
from candidate import transform,controlled_position

SHAPES={'native':(8,8),'aligned':(2,16,8),'wave':(8,2),'concepts':(6,16),
        'basis':(88,),'lexical':(3,48),'previous_key':(8,),'upper':(48,),'last_logits':(256,)}
MAX_CLOCK=2**63-1


@dataclass(frozen=True)
class InferenceState:
    owners:tuple[str,...]
    clock:int
    weights_sha256:str
    position_mode:str
    implementation_sha256:str
    tensors:dict[str,torch.Tensor]

    def tensor_bytes(self):
        return sum(v.numel()*v.element_size() for v in self.tensors.values())


class IncrementalTAC043:
    def __init__(self,model,position_mode='native'):
        if type(model) is not FactorialDepthControl040 or model.upper_state_width!=48 or model.upper_concept_graph:
            raise ValueError('only registered040 graph-free48D model supported')
        if position_mode not in ('native','cap_all'): raise ValueError('unregistered position mode')
        if model.embedding.weight.dtype not in (torch.float32,torch.float64): raise ValueError('unqualified precision')
        self.model=model;self.position_mode=position_mode
        source=hashlib.sha256(str(torch.__version__).encode())
        for relative in ('langstream043/incremental.py','context026/context_transition.py','langlayer040/factorial_depth.py','langlayer038/recurrence_only.py','langmemory042/candidate.py'):
            source.update(relative.encode());source.update((P/relative).read_bytes())
        self.implementation_sha256=source.hexdigest()
        self._binding=self._signature();self.weights_sha256=self._weight_hash()
        self._check_model()

    def _signature(self):
        return tuple((n,id(p),p._version,p.dtype,p.device) for n,p in self.model.named_parameters())

    def _weight_hash(self):
        h=hashlib.sha256()
        for name,p in self.model.state_dict().items():
            h.update(name.encode());h.update(str(p.dtype).encode());h.update(str(tuple(p.shape)).encode())
            h.update(p.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    def _check_model(self):
        if any(m.training for m in self.model.modules()): raise ValueError('eval mode required')
        if self._signature()!=self._binding: raise ValueError('model weights/dtype/device changed; reset engine')
        if any(m._forward_pre_hooks or m._forward_hooks for m in self.model.modules()): raise ValueError('unregistered model hook')

    @staticmethod
    def _owners(owners):
        if not isinstance(owners,(list,tuple)) or not 1<=len(owners)<=64 or len(set(owners))!=len(owners):
            raise ValueError('unique explicit stream owners required')
        if any(type(o) is not str or not o or len(o.encode())>128 for o in owners): raise ValueError('invalid stream owner')
        return tuple(owners)

    def start(self,owners):
        self._check_model();owners=self._owners(owners);w=self.model.embedding.weight
        tensors={name:w.new_zeros((len(owners),*shape)) for name,shape in SHAPES.items()}
        return InferenceState(owners,0,self.weights_sha256,self.position_mode,self.implementation_sha256,tensors)

    def validate(self,state,owners):
        self._check_model();owners=self._owners(owners)
        if type(state) is not InferenceState or state.owners!=owners: raise ValueError('state owner mismatch')
        if state.weights_sha256!=self.weights_sha256 or state.position_mode!=self.position_mode or state.implementation_sha256!=self.implementation_sha256: raise ValueError('state model/mode mismatch')
        if type(state.clock) is not int or not 0<=state.clock<MAX_CLOCK: raise ValueError('invalid state clock')
        if set(state.tensors)!=set(SHAPES): raise ValueError('state schema mismatch')
        w=self.model.embedding.weight
        for name,shape in SHAPES.items():
            x=state.tensors[name]
            if not torch.is_tensor(x) or tuple(x.shape)!=(len(owners),*shape) or x.dtype!=w.dtype or x.device!=w.device or x.requires_grad or not torch.isfinite(x).all():
                raise ValueError('state tensor invalid: '+name)

    @torch.no_grad()
    def step(self,token,state,*,owners,write_mask=None):
        self.validate(state,owners)
        if token.dtype!=torch.long or token.shape!=(len(state.owners),) or token.device!=self.model.embedding.weight.device or token.min()<0 or token.max()>255:
            raise ValueError('one observed byte per stream required')
        if write_mask is None: write_mask=torch.ones_like(token,dtype=torch.bool)
        if write_mask.shape!=token.shape or write_mask.dtype!=torch.bool or write_mask.device!=token.device: raise ValueError('invalid write admission')
        logits,next_state=self._advance(token,state,write_mask)
        self.validate(next_state,owners)
        return logits,next_state

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
        akey=F.normalize(torch.einsum('bi,hij->bhj',ordinal_code(tt,e.dtype),m.rotation),dim=-1,eps=1e-6)
        pk=F.normalize(torch.einsum('bi,hij->bhj',ordinal_code((tt-1).clamp_min(0),e.dtype),m.rotation),dim=-1,eps=1e-6)
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

    @torch.no_grad()
    def prefill(self,tokens,*,owners,write_mask=None):
        """Bulk original prefill, then reconstruct exact-state semantics at last byte.
        No entire input history is retained in the returned state.
        """
        initial=self.start(owners)
        if tokens.ndim!=2 or tokens.shape[0]!=len(initial.owners) or not tokens.shape[1] or tokens.dtype!=torch.long or tokens.device!=self.model.embedding.weight.device or tokens.min()<0 or tokens.max()>255:
            raise ValueError('invalid source prefill')
        b,length=tokens.shape
        if length>=MAX_CLOCK: raise ValueError('source clock limit')
        if write_mask is None: write_mask=torch.ones_like(tokens,dtype=torch.bool)
        if write_mask.shape!=tokens.shape or write_mask.dtype!=torch.bool or write_mask.device!=tokens.device: raise ValueError('invalid prefill admission')
        captured=[]
        hook=self.model.layernorm[1].register_forward_pre_hook(lambda module,args:captured.append(args[0]))
        try:
            with controlled_position(self.model,self.position_mode):
                logits,tr=self.model(tokens,write_mask=write_mask,return_trace=True)
        finally: hook.remove()
        _,upper_trace=self.model.layernorm[1](captured[0],return_trace=True)
        m=self.model;tt=torch.arange(length,device=tokens.device).expand(b,length)
        embed=m.embedding(tokens)
        e=(embed+m.pos(transform(phi_position(tt,embed.dtype),self.position_mode))).tanh()
        basis=m.basis(e).tanh().cumsum(1)
        lex=embed.tanh();oldlex=[]
        for lag in (2,3,4): oldlex.append(lex[:,-lag] if length>=lag else embed.new_zeros((b,48)))
        before={**initial.tensors,'native':tr['native_prewrite'][:,-1],
                'aligned':tr['aligned_prewrite'][:,-1],
                'wave':tr['wave_prewrite'][:,-1].view(b,8,2),
                'concepts':tr['concept_prewrite'][:,-1],
                'basis':basis[:,-2] if length>1 else embed.new_zeros((b,88)),
                'lexical':torch.stack(oldlex,1),
                'previous_key':tr['native_query_key'][:,-2] if length>1 else embed.new_zeros((b,8)),
                'upper':upper_trace['prior_state'][:,-1]}
        prior=replace(initial,clock=length-1,tensors=before)
        _,state=self._advance(tokens[:,-1],prior,write_mask[:,-1])
        last=logits[:,-1].clone();state=replace(state,tensors={**state.tensors,'last_logits':last.clone()})
        self.validate(state,owners)
        return last,state

    @staticmethod
    def _snapshot_digest(payload):
        meta={k:v for k,v in payload.items() if k not in ('tensors','digest')}
        h=hashlib.sha256(json.dumps(meta,sort_keys=True,separators=(',',':')).encode())
        for name,value in sorted(payload['tensors'].items()):
            h.update(name.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    def save_state(self,state,path):
        self.validate(state,state.owners);path=Path(path)
        payload={'schema':'langstream043.v1','owners':state.owners,'clock':state.clock,
                 'weights_sha256':state.weights_sha256,'position_mode':state.position_mode,'implementation_sha256':state.implementation_sha256,
                 'tensors':{n:x.detach().cpu().clone() for n,x in state.tensors.items()}}
        payload['digest']=self._snapshot_digest(payload)
        with path.open('xb') as handle: torch.save(payload,handle)

    def load_state(self,path,*,owners):
        path=Path(path)
        if path.stat().st_size>1024**2: raise ValueError('oversize state snapshot')
        payload=torch.load(path,map_location='cpu',weights_only=True)
        if not isinstance(payload,dict) or set(payload)!={'schema','owners','clock','weights_sha256','position_mode','implementation_sha256','tensors','digest'} or payload['schema']!='langstream043.v1' or self._snapshot_digest(payload)!=payload['digest']:
            raise ValueError('state snapshot schema/integrity failure')
        device=self.model.embedding.weight.device
        state=InferenceState(tuple(payload['owners']),payload['clock'],payload['weights_sha256'],payload['position_mode'],payload['implementation_sha256'],{n:v.to(device) for n,v in payload['tensors'].items()})
        self.validate(state,owners)
        return state
