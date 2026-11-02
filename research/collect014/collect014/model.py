"""Ordering-only phase change and loss-only Switch balancing; no new parameters."""
import hashlib,json,math
import torch
from torch.nn import functional as F
from .common import OLD
from collect011.model import CollectiveTAC
from collect011.memory import PhaseMemory,PhaseState,write_step

class OrderedPhaseMemory(PhaseMemory):
    def forward(self,x,state=None,write_mask=None,*,owners=None):
        if not self.read_after:
            return super().forward(x,state=state,write_mask=write_mask,owners=owners)
        if owners is None:
            if state is not None:raise ValueError('explicit continuing owners required')
            owners=tuple(f'fresh-{i}' for i in range(x.shape[0]))
        owners=self.owners(owners);state=self.start(owners) if state is None else state
        mask=torch.ones(x.shape[:2],device=x.device,dtype=torch.bool) if write_mask is None else write_mask
        self.validate(x,state,owners,mask)
        shape=(*x.shape[:2],self.heads,self.key_dim//2)
        theta=math.pi*(self.content(x).reshape(shape).tanh()+self.role(x).reshape(shape).tanh().roll(1,-1))
        key=torch.stack((theta.cos(),theta.sin()),-1).flatten(-2)/math.sqrt(self.key_dim//2)
        values=self.value(x).reshape(*x.shape[:2],self.heads,self.value_dim).tanh()
        route=self.router(x).softmax(-1)
        hard=F.one_hot(route.argmax(-1),self.heads).to(route.dtype);route=hard+(route-route.detach())
        gain=self.admission(x).sigmoid()*mask[...,None]*route
        memory,precision=state.memory,state.precision;reads=[]
        for key_t,value,g,r in zip(key.unbind(1),values.unbind(1),gain.unbind(1),route.unbind(1)):
            memory,precision=write_step(memory,precision,key_t,value,g,kind=self.kind)
            reads.append((memory@key_t.unsqueeze(-1)).squeeze(-1)*r[...,None])
        return torch.stack(reads,1),PhaseState(owners,state.clock+x.shape[1],memory.contiguous(),precision.contiguous(),self.signature)

class StudyTAC(CollectiveTAC):
    def __init__(self,variant,*,read_after,balance):
        super().__init__('hybrid_hard_p',width=16,ffn_dim=57,depth=4)
        self.variant=variant;self.balance_coefficient=balance;self.router_logits=[]
        self.collect_signature=hashlib.sha256((self.collect_signature+'COLLECT014:'+variant).encode()).hexdigest()
        for block in self.core.blocks:
            memory=block.phase_memory;memory.__class__=OrderedPhaseMemory;memory.read_after=read_after
            memory.signature=hashlib.sha256((memory.signature+json.dumps(dict(read_after=read_after))).encode()).hexdigest()
            memory.router.register_forward_hook(self._capture_router)
    def _capture_router(self,module,inputs,output):self.router_logits.append(output)
    def forward(self,*args,**kwargs):
        self.router_logits=[]
        return super().forward(*args,**kwargs)
    def balancing(self):
        losses=[];counts=[]
        for z in self.router_logits:
            p=z.softmax(-1).reshape(-1,z.shape[-1])
            hard=F.one_hot(p.argmax(-1),p.shape[-1]).to(p.dtype)
            counts.append(hard.sum(0).detach());losses.append(p.shape[-1]*(hard.mean(0)*p.mean(0)).sum())
        if len(losses)!=4:raise ValueError('complete layer routing trace required')
        return torch.stack(losses).mean(),torch.stack(counts)
