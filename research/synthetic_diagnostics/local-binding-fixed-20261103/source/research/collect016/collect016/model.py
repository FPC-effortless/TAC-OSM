"""One graph with connected or detached chunk credit; active current-only control."""
import hashlib,math
import torch
from torch.nn import functional as F
from collect014.model import StudyTAC,OrderedPhaseMemory
from collect011.model import CollectiveTAC
from collect011.memory import PhaseMemory,PhaseState

class CurrentValueMemory(PhaseMemory):
    """Remove unused addresses; only the immediate phase value component remains.

    Native fast/slow/rotary/SSA state remains trainable. This control is therefore
    NOT history-free. Removed address parameters are reallocated to active FFNs.
    """
    def start(self,owners):
        owners=self.owners(owners);w=self.value.weight
        return PhaseState(owners,0,w.new_zeros(len(owners),self.heads,self.value_dim,self.key_dim),
            torch.eye(self.key_dim,device=w.device,dtype=w.dtype).repeat(len(owners),self.heads,1,1)/self.ridge,self.signature)
    def forward(self,x,state=None,write_mask=None,*,owners=None):
        if owners is None:
            if state is not None:raise ValueError('explicit continuing owners required')
            owners=tuple(f'fresh-{i}' for i in range(x.shape[0]))
        owners=self.owners(owners);state=self.start(owners) if state is None else state
        if state.signature!=self.signature or state.owners!=owners:raise ValueError('state/owner mismatch')
        mask=torch.ones(x.shape[:2],dtype=torch.bool,device=x.device) if write_mask is None else write_mask
        if mask.dtype!=torch.bool or mask.shape!=x.shape[:2]:raise ValueError('Bool[B,T] mask required')
        p=self.router(x).softmax(-1);hard=F.one_hot(p.argmax(-1),self.heads).to(p.dtype);r=hard+(p-p.detach())
        g=self.admission(x).sigmoid()*mask[...,None]*r
        v=self.value(x).reshape(*x.shape[:2],self.heads,self.value_dim).tanh()
        return r[...,None]*g[...,None]*v,PhaseState(owners,state.clock+x.shape[1],state.memory,state.precision,self.signature)

class CarriedTAC(StudyTAC):
    def __init__(self,variant):
        # Same native family, all real active parameters counted.
        if variant!='current_only':
            super().__init__(variant,read_after=True,balance=.01)
        else:
            seed=torch.initial_seed()
            super().__init__('initializer_source',read_after=True,balance=.01)
            original={n:p.detach().clone() for n,p in self.named_parameters()}
            torch.manual_seed(seed)
            CollectiveTAC.__init__(self,'hybrid_hard_p',width=16,ffn_dim=78,depth=4)
            self.variant=variant;self.balance_coefficient=.01;self.router_logits=[]
            for block in self.core.blocks:
                m=block.phase_memory;m.__class__=CurrentValueMemory;del m.content;del m.role
                m.signature=hashlib.sha256((m.signature+'COLLECT016:current_only').encode()).hexdigest()
                m.router.register_forward_hook(self._capture_router)
            with torch.no_grad():
                for n,p in self.named_parameters():
                    if n in original and p.shape==original[n].shape:p.copy_(original[n])
                    elif n in original and '.ffn_' in n:
                        a=original[n]
                        if p.shape[0]==a.shape[0]:p[:,:a.shape[1]].copy_(a)
                        else:p[:a.shape[0]].copy_(a)
        self.collect_signature=hashlib.sha256((self.collect_signature+'COLLECT016:'+variant).encode()).hexdigest()

def run_chunks(model,x,*,chunk=128,detach=False,state=None,owners=None,collect_balance=True):
    owners=tuple(f'owner-{i}' for i in range(x.shape[0])) if owners is None else tuple(owners)
    logits=[];balances=[];counts=[]
    for lo in range(0,x.shape[1],chunk):
        z,state=model(x[:,lo:lo+chunk],state=state,owners=owners)
        logits.append(z)
        if collect_balance:
            loss,count=model.balancing();balances.append(loss*z.shape[1]/x.shape[1]);counts.append(count)
        if detach and lo+chunk<x.shape[1]:state=state.detach()
    return torch.cat(logits,1),state,(sum(balances) if balances else None),(torch.stack(counts).sum(0) if counts else None)
