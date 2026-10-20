"""ME005: native TAC-OSM unified predictive-executable research architecture.
Wraps hash-verified trained ME004 native two-timescale checkpoint, NOT an SSM.
All new modules are optional and start with output-preserving residual = 0.
Continuous critics never authorize actions; real receipts required for promotion.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from scripts.me004_native_real_forecast import NativeTAC, FEATURES, HORIZONS

D = len(FEATURES)

def sparsemax(x,dim=-1):
    z=x-x.max(dim=dim,keepdim=True).values
    zs=z.sort(dim=dim,descending=True).values
    r=torch.arange(1,z.size(dim)+1,device=z.device,dtype=z.dtype)
    shape=[1]*z.ndim;shape[dim]=z.size(dim)
    r=r.view(shape)
    cs=zs.cumsum(dim)-1
    support=r*zs>cs
    k=support.sum(dim=dim,keepdim=True).clamp_min(1)
    tau=cs.gather(dim,k-1)/k
    return (z-tau).clamp_min(0)

@dataclass
class State:
    fast:torch.Tensor
    slow:torch.Tensor
    slots:torch.Tensor
    assoc:torch.Tensor
    eligibility:torch.Tensor
    phase:torch.Tensor

class AssemblyGraph(nn.Module):
    def __init__(self,width=16,slots=4):
        super().__init__()
        self.width=width;self.count=slots
        self.anchors=nn.Parameter(torch.randn(slots,width)*.08)
        self.sensor_codes=nn.Parameter(torch.randn(D,width)*.04)
        self.token_encoder=nn.Linear(width+2,width)
        self.query=nn.Linear(width*2,width,bias=False)
        self.key=nn.Linear(width,width,bias=False)
        self.null=nn.Linear(width,1)
        self.graph_query=nn.Linear(width,width,bias=False)
        self.graph_key=nn.Linear(width,width,bias=False)
        self.candidate=nn.Linear(width*3,width)
        self.write=nn.Linear(width*2,1)
    def forward(self,slots,values,mask,write_gate):
        B=slots.size(0)
        codes=self.sensor_codes[None].expand(B,-1,-1)
        tokens=torch.tanh(self.token_encoder(torch.cat(
            (codes,values.unsqueeze(-1),mask.unsqueeze(-1)),dim=-1)))
        query=self.query(torch.cat((slots,self.anchors[None].expand(B,-1,-1)),dim=-1))
        similarity=torch.einsum('bsd,bfd->bsf',query,self.key(tokens))/math.sqrt(self.width)
        scores=torch.cat((similarity,self.null(tokens).transpose(1,2)),dim=1)
        alloc=F.softmax(scores,dim=1)[:,:self.count]*mask[:,None,:]
        mass=alloc.sum(-1,keepdim=True)
        pooled=torch.einsum('bsf,bfd->bsd',alloc,tokens)/mass.clamp_min(1e-6)
        edges=sparsemax(torch.einsum('bsd,btd->bst',
           self.graph_query(slots),self.graph_key(slots))/math.sqrt(self.width),dim=-1)
        neighbors=edges@slots
        candidate=torch.tanh(self.candidate(torch.cat((slots,pooled,neighbors),dim=-1)))
        g=torch.sigmoid(self.write(torch.cat((slots,pooled),dim=-1)))
        g=g*write_gate[:,None,:]*(mass>1e-6)
        return (1-g)*slots+g*candidate,alloc,edges

class PredictiveHebbianMemory(nn.Module):
    def __init__(self,width=16):
        super().__init__()
        self.width=width
        self.obs=nn.Linear(2*D,width)
        self.predict=nn.Linear(width,width)
        self.to_read=nn.Linear(width,width)
        self.eta=nn.Parameter(torch.tensor(-4.0))
        self.decay=nn.Parameter(torch.tensor(-5.0))
    def forward(self,A,E,pooled,x,mask,plastic_gate):
        is_seen=(mask.sum(-1,keepdim=True)>0).to(x.dtype)
        target=torch.tanh(self.obs(x))
        pre=torch.tanh(pooled)
        retrieval=(A@pre.unsqueeze(-1)).squeeze(-1)
        prediction=torch.tanh(self.predict(pre)+retrieval)
        error=target-prediction
        gain=.5*(target.square().sum(-1)-error.square().sum(-1))
        E=.9*E+pre*error
        modulation=plastic_gate*is_seen
        lr=.05*torch.sigmoid(self.eta)
        decay=.01*torch.sigmoid(self.decay)
        A=(1-decay)*A+lr*modulation.unsqueeze(-1)*torch.einsum('bd,be->bde',error,pre)/self.width
        scale=(A.flatten(1).norm(dim=-1,keepdim=True)/2).clamp_min(1)
        A=A/scale[:,None,:]
        return A,E,self.to_read(retrieval),gain

class ExecutableOperators(nn.Module):
    def __init__(self,width=16,count=4):
        super().__init__()
        roles=torch.where(torch.randn(2,width)>0,1.0,-1.0)
        self.register_buffer('roles',roles)
        self.operators=nn.ModuleList([nn.Sequential(nn.Linear(width*2,width),
                nn.Tanh(),nn.Linear(width,width),nn.Tanh()) for _ in range(count)])
        self.router=nn.Linear(2*width,count+1) # identity/no-op is last
    def forward(self,memory,slots):
        bind=torch.cat((memory*self.roles[0],slots*self.roles[1]),dim=-1)
        gates=F.softmax(self.router(bind),dim=-1)
        candidates=torch.stack([f(bind) for f in self.operators],dim=1)
        return memory+(candidates*gates[:,:-1,None]).sum(1),gates

class PhaseGates(nn.Module):
    def __init__(self):
        super().__init__()
        self.freq=nn.Parameter(torch.tensor(-3.0))
        self.offset=nn.Parameter(torch.zeros(4))
        self.amplitude=nn.Parameter(torch.zeros(4))
        self.context=nn.Linear(64,4)
    def forward(self,z,phase,use_phase=True):
        phase=torch.remainder(phase+.7*torch.sigmoid(self.freq),2*math.pi)
        rhythm=torch.sin(phase+self.offset)*self.amplitude if use_phase else 0
        return phase,torch.sigmoid(self.context(z)+rhythm)

class AttractorRefinement(nn.Module):
    def __init__(self,steps=4):
        super().__init__()
        self.steps=steps
        self.prototypes=nn.Parameter(torch.randn(6,64)*.04)
        self.q=nn.Linear(64,64)
        self.stop=nn.Linear(64,1)
        self.step_logit=nn.Parameter(torch.tensor(-1.4))
    def forward(self,z,gate):
        p=F.softmax(self.q(z)@self.prototypes.T/math.sqrt(64),dim=-1)
        target=p@self.prototypes
        alpha=.5*torch.sigmoid(self.step_logit)*gate
        cur=z;remain=torch.ones_like(gate);used=[];weights=[]
        for k in range(self.steps):
            cur=(1-alpha)*cur+alpha*target
            stop=torch.sigmoid(self.stop(cur)).clamp(.01,.99)
            w=remain*stop if k<self.steps-1 else remain
            used.append(cur);weights.append(w)
            remain=remain*(1-stop)
        z_out=sum(s*w for s,w in zip(used,weights))
        cost=sum((i+1)*w for i,w in enumerate(weights)).mean()
        return z_out,cost,p

class UnifiedNativeTAC(nn.Module):
    def __init__(self,backbone=None,graph=True,plastic=True,operators=True,
                 rhythms=True,reasoning=True,freeze_base=True):
        super().__init__()
        self.backbone=backbone if backbone is not None else NativeTAC('tac_two_timescale')
        if freeze_base:
            for p in self.backbone.parameters():p.requires_grad_(False)
        self.has_graph=graph;self.has_plastic=plastic
        self.has_operators=operators;self.has_rhythm=rhythms;self.has_reasoning=reasoning
        self.graph=AssemblyGraph()
        self.plastic=PredictiveHebbianMemory()
        self.operator=ExecutableOperators()
        self.phase=PhaseGates()
        self.project=nn.Linear(32,64)
        self.reason=AttractorRefinement()
        self.correction=nn.Linear(64,2*D)
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)
        self.logvar=nn.Linear(64,2*D)
        self.critic=nn.Linear(64,1)
        self.action_embed=nn.Linear(4,64)
        self.outcome=nn.Linear(64,1)
    def initial(self,B,device,dtype):
        return State(torch.zeros(B,32,device=device,dtype=dtype),
            torch.zeros(B,32,device=device,dtype=dtype),
            torch.zeros(B,4,16,device=device,dtype=dtype),
            torch.zeros(B,16,16,device=device,dtype=dtype),
            torch.zeros(B,16,device=device,dtype=dtype),
            torch.zeros(B,1,device=device,dtype=dtype))
    def step(self,x,s):
        mask=x[:,D:].clamp(0,1)
        value=torch.where(mask>0,x[:,:D],torch.zeros_like(mask))
        x=torch.cat((value,mask),dim=-1)
        proposal=torch.tanh(self.backbone.encoder(x)+self.backbone.recur(s.fast))
        g=torch.sigmoid(self.backbone.gate(torch.cat((x,s.fast),dim=-1)))
        fast=(1-g)*s.fast+g*proposal
        pair=torch.cat((s.slow,fast),dim=-1)
        slowgate=torch.sigmoid(self.backbone.slow_gate(pair))
        slow=(1-slowgate)*s.slow+slowgate*torch.tanh(self.backbone.slow(pair))
        latent=torch.cat((fast,slow),dim=-1)
        phi,gates=self.phase(latent,s.phase,self.has_rhythm)
        write,retrieve,reason,plastic=[gates[:,i:i+1] for i in range(4)]
        if not self.has_graph:write=write*0
        slots,allocation,edges=self.graph(s.slots,value,mask,write)
        pooled=slots.mean(1)
        if self.has_plastic:
            A,E,mem,cdl=self.plastic(s.assoc,s.eligibility,pooled,x,mask,plastic)
        else:
            A,E,mem,cdl=s.assoc,s.eligibility,torch.zeros_like(pooled),x.new_zeros(len(x))
        if self.has_operators:
            composition,usage=self.operator(mem*retrieve,pooled)
        else:
            composition,usage=torch.zeros_like(pooled),None
        updated=State(fast,slow,slots,A,E,phi)
        aux={'allocation':allocation,'edges':edges,'cdl_gain':cdl,
             'operator_usage':usage,'reason_gate':reason}
        return updated,composition,aux
    def forward(self,x,recorded_actions=None,details=False):
        if x.ndim!=3 or x.shape[-1]!=2*D:raise ValueError('expected [B,T,24] input')
        s=self.initial(x.size(0),x.device,x.dtype)
        operator=None;aux=None
        for t in range(x.size(1)):
            s,operator,aux=self.step(x[:,t],s)
        latent=torch.cat((s.fast,s.slow),dim=-1)
        baseline=self.backbone.head(latent).reshape(-1,2,D)
        enhanced=latent+.2*torch.tanh(self.project(torch.cat((s.slots.mean(1),operator),dim=-1)))
        if recorded_actions is not None:
            if recorded_actions.shape!=(len(x),4):raise ValueError('recorded action shape [B,4]')
            enhanced=enhanced+.1*self.action_embed(recorded_actions)
        if self.has_reasoning:
            refined,cost,attractors=self.reason(enhanced,aux['reason_gate'])
        else:refined,cost,attractors=enhanced,x.new_zeros(()),None
        prediction=baseline+self.correction(refined).reshape(-1,2,D)
        if not details:return prediction
        return {'prediction':prediction,'baseline':baseline,
                'logvariance':self.logvar(refined).reshape(-1,2,D).clamp(-8,5),
                'energy_risk':F.softplus(self.critic(refined)),
                'outcome_logit':self.outcome(refined) if recorded_actions is not None else None,
                'compute_cost':cost,'state':s,'assembly_assignment':aux['allocation'],
                'graph':aux['edges'],'cdl_gain':aux['cdl_gain'],
                'operator_usage':aux['operator_usage'],'attractor_prob':attractors}

class EvidenceLedger:
    """Never promote provisional executable knowledge without real distinct receipts."""
    def __init__(self,min_contexts=3):
        self.min_contexts=min_contexts;self.provisional={};self.persistent={}
    def propose(self,key,payload):
        if key in self.persistent:return False
        self.provisional.setdefault(key,{'payload':payload,'contexts':set(),'failed':False})
        return True
    def verify(self,key,context,success,receipt):
        if key not in self.provisional or not receipt:
            raise ValueError('real receipt and provisional entry required')
        entry=self.provisional[key]
        if not success:entry['failed']=True;return False
        entry['contexts'].add((str(context),str(receipt)))
        if not entry['failed'] and len({c for c,_ in entry['contexts']})>=self.min_contexts:
            self.persistent[key]=entry['payload'];del self.provisional[key];return True
        return False

def load_me004_backbone(checkpoint,report,seed):
    import hashlib,json
    from pathlib import Path
    d=json.loads(Path(report).read_text())
    digest=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    good={(v['model'],v['seed'],v['checkpoint_sha256']) for v in d['fits']}
    if ('tac_two_timescale',seed,digest) not in good:raise ValueError('unverified checkpoint')
    saved=torch.load(checkpoint,weights_only=True,map_location='cpu')
    if saved['kind']!='tac_two_timescale' or saved['seed']!=seed:raise ValueError('checkpoint metadata')
    model=NativeTAC('tac_two_timescale')
    model.load_state_dict(saved['state_dict'],strict=True)
    return model
