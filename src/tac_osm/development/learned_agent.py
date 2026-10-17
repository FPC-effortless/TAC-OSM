"""Standalone learned control path. Runtime dependency: torch only.

No task-specific rules, entity-key dictionaries, teachers, simulator imports,
transition tables, analytical planner, cost argmin, retry rule or stop rule.
All selection below is over trained neural logits or tensor padding metadata.
"""
from dataclasses import dataclass,asdict
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class AgentConfig:
    event_features:int=22
    action_features:int=11
    query_features:int=8
    slots:int=32
    dimension:int=48
    read_heads:int=2
    read_top_k:int=4
    operators:int=3
    depth:int=2


def mlp(a,b,c):return nn.Sequential(nn.Linear(a,b),nn.GELU(),nn.Linear(b,c))


class LearnedAgent(nn.Module):
    def __init__(self,config=AgentConfig(),seed=0):
        super().__init__();torch.manual_seed(seed);self.config=config
        d,m=config.dimension,config.slots
        self.admission=mlp(config.event_features,32,1)
        self.address=mlp(m+config.event_features,64,m)
        self.encoder=mlp(config.event_features,d,d)
        self.action_encoder=mlp(config.action_features,d,d)
        self.query_encoder=mlp(config.query_features,d,d)
        self.null=nn.Parameter(torch.randn(d)*.02)
        self.read_query=nn.Linear(d,config.read_heads*d)
        self.read_key=nn.Linear(d,config.read_heads*d)
        self.read_value=nn.Linear(d,d)
        self.node=mlp((config.read_heads+1)*d,d,d)
        self.graph_q=nn.ModuleList([nn.Linear(d,d) for _ in range(config.depth)])
        self.graph_k=nn.ModuleList([nn.Linear(d,d) for _ in range(config.depth)])
        self.graph_v=nn.ModuleList([nn.Linear(d,d) for _ in range(config.depth)])
        self.graph_norm=nn.ModuleList([nn.LayerNorm(d) for _ in range(config.depth)])
        self.operator=nn.ModuleList([mlp(2*d,2*d,d) for _ in range(config.operators)])
        self.operator_selector=nn.Linear(2*d,config.operators)
        self.halt=nn.Linear(2*d,1);nn.init.constant_(self.halt.bias,-2.)
        self.pool=nn.Linear(d,1)
        self.transition=mlp(2*d,2*d,d)
        self.current_verifier=mlp(d,d,1)
        self.verifier=mlp(d,d,1)
        self.cost=mlp(d,d,1)
        self.outcome=mlp(2*d,d,1)
        self.future_policy=mlp(2*d,d,1)
        self.policy=mlp(2*d+3,2*d,1)
        self.last_work={}

    def initial(self,batch,device=None):
        ref=next(self.parameters())
        return (torch.zeros(batch,self.config.slots,self.config.dimension,device=device or ref.device),
                torch.zeros(batch,self.config.slots,device=device or ref.device))

    def write_logits(self,event,occupied):
        return self.admission(event).squeeze(-1),self.address(torch.cat((event,occupied),-1))

    def observe(self,state,event,present=None,force_write=False):
        if event.ndim!=2 or event.shape[-1]!=self.config.event_features or not torch.isfinite(event).all():
            raise ValueError('finite public event tensor required')
        memory,occupied=state
        gate,address=self.write_logits(event,occupied)
        hard=(gate>=0).to(event.dtype)
        if force_write:hard=torch.ones_like(hard)
        if present is not None:hard=hard*present.to(hard.dtype)
        choice=address.argmax(-1)
        selected=F.one_hot(choice,self.config.slots).to(event.dtype)*hard[:,None]
        value=self.encoder(event)
        memory=memory*(1-selected[:,:,None])+selected[:,:,None]*value[:,None]
        occupied=torch.maximum(occupied,selected)
        return (memory,occupied),{'writes':int(hard.detach().sum()),'address':choice.detach(),'gate':hard.detach()}

    def encode_bank(self,public_records,occupied):
        return self.encoder(public_records)*occupied[:,:,None],occupied

    def read(self,state,actions,uniform=False):
        memory,occupied=state;b,a,d=actions.shape;h=self.config.read_heads
        bank=torch.cat((memory,self.null[None,None].expand(b,1,-1)),1)
        valid=torch.cat((occupied>0,torch.ones(b,1,dtype=torch.bool,device=memory.device)),1)
        q=self.read_query(actions).reshape(b,a,h,d)
        k=self.read_key(bank).reshape(b,-1,h,d)
        logits=torch.einsum('bahd,bmhd->bahm',q,k)/math.sqrt(d)
        logits=logits.masked_fill(~valid[:,None,None],-1e9)
        soft=logits.softmax(-1)
        if uniform:
            weights=valid[:,None,None].to(soft.dtype).expand_as(soft)
            weights=weights/weights.sum(-1,keepdim=True)
        else:
            ids=logits.topk(min(self.config.read_top_k,bank.shape[1]),-1).indices
            hard=torch.zeros_like(soft).scatter_(-1,ids,soft.gather(-1,ids))
            hard=hard/hard.sum(-1,keepdim=True).clamp_min(1e-9)
            weights=hard.detach()+soft-soft.detach() if self.training else hard
        if not self.training and not uniform:
            selected=bank[:,None,None].expand(-1,a,h,-1,-1).gather(3,ids[:,:,:,:,None].expand(-1,-1,-1,-1,d))
            retained=logits.gather(-1,ids).softmax(-1)
            read=(retained[:,:,:,:,None]*self.read_value(selected)).sum(-2).flatten(-2)
        else:
            read=torch.einsum('bahm,bmd->bahd',weights,self.read_value(bank)).flatten(-2)
        return read,logits

    def forward(self,state,descriptors,valid,queries,query_valid,lesion='none'):
        if descriptors.ndim!=3 or descriptors.shape[-1]!=self.config.action_features:
            raise ValueError('public candidate descriptors required')
        if not torch.isfinite(descriptors).all() or not torch.isfinite(queries).all():raise ValueError('nonfinite public features')
        if not valid.any(-1).all():raise ValueError('at least one public candidate required')
        if lesion=='zero_memory':state=(torch.zeros_like(state[0]),state[1])
        b,a,_=descriptors.shape;d=self.config.dimension
        action=self.action_encoder(descriptors)
        reads,read_logits=self.read(state,action,lesion=='uniform_reads')
        node=self.node(torch.cat((action,reads),-1))
        calls=0;graph_edges=[];selection=[]
        if lesion!='no_casm':
            active=valid.clone()
            for depth in range(self.config.depth):
                edges=torch.einsum('bad,bcd->bac',self.graph_q[depth](node),self.graph_k[depth](node))/math.sqrt(d)
                edges=edges.masked_fill(~valid[:,None],-1e9).softmax(-1)
                node=self.graph_norm[depth](node+torch.einsum('bac,bcd->bad',edges,self.graph_v[depth](node)))
                joint=torch.cat((node,action),-1).reshape(-1,2*d)
                probability=self.operator_selector(joint).softmax(-1)
                chosen=probability.argmax(-1)
                hard=F.one_hot(chosen,self.config.operators).to(node.dtype)
                weights=hard.detach()+probability-probability.detach() if self.training else hard
                alive_probability=1-self.halt(joint).sigmoid().squeeze(-1)
                alive=(alive_probability>.1).to(node.dtype)*active.flatten().to(node.dtype)
                gate=alive.detach()+alive_probability-alive_probability.detach() if self.training else alive
                delta=torch.zeros_like(node).reshape(-1,d)
                for op,operator in enumerate(self.operator):
                    ids=((chosen==op)&(alive>0)).nonzero(as_tuple=True)[0]
                    if len(ids):
                        delta=delta.index_add(0,ids,weights[ids,op,None]*operator(joint[ids]))
                        calls+=len(ids)
                node=node+(gate[:,None]*delta).reshape(b,a,d)
                active=active&(alive.reshape(b,a)>0)
                graph_edges.append(edges);selection.append(chosen.reshape(b,a))
        pool=self.pool(node).squeeze(-1).masked_fill(~valid,-1e9).softmax(-1)
        context=torch.einsum('ba,bad->bd',pool,node)
        initial=self.current_verifier(context).squeeze(-1)
        z1=context[:,None]+self.transition(torch.cat((context[:,None].expand(-1,a,-1),node),-1))
        if lesion=='zero_dynamics':z1=torch.zeros_like(z1)
        future_logits=self.future_policy(torch.cat((z1[:,:,None].expand(-1,-1,a,-1),action[:,None].expand(-1,a,-1,-1)),-1)).squeeze(-1)
        future_logits=future_logits.masked_fill(~valid[:,None],-1e9)
        p=future_logits.softmax(-1);choice=p.argmax(-1)
        hard=F.one_hot(choice,a).to(p.dtype)
        weights=hard.detach()+p-p.detach() if self.training else hard
        proposed=torch.einsum('bac,bcd->bad',weights,action)
        z2=z1+self.transition(torch.cat((z1,proposed),-1))
        if lesion=='zero_dynamics':z2=torch.zeros_like(z2)
        next_done=self.verifier(z1).squeeze(-1);later_done=self.verifier(z2).squeeze(-1)
        cost=self.cost(z1).squeeze(-1)
        q=self.query_encoder(queries)
        outcomes=self.outcome(torch.cat((z1[:,:,None].expand(-1,-1,q.shape[1],-1),q[:,None].expand(-1,a,-1,-1)),-1)).squeeze(-1)
        later=self.outcome(torch.cat((z2[:,:,None].expand(-1,-1,q.shape[1],-1),q[:,None].expand(-1,a,-1,-1)),-1)).squeeze(-1)
        features=torch.cat((z1,z2,initial[:,None,None].expand(-1,a,-1),next_done[:,:,None],cost[:,:,None]),-1)
        logits=self.policy(features).squeeze(-1).masked_fill(~valid,-1e9)
        if lesion=='zero_policy':logits=torch.zeros_like(logits).masked_fill(~valid,-1e9)
        self.last_work={'read_candidates':int(valid.sum())*(self.config.slots+1)*self.config.read_heads,
                        'retained_read_slots':int(valid.sum())*self.config.read_top_k*self.config.read_heads,
                        'operator_calls':calls,'latent_transitions':int(valid.sum())*2}
        return {'logits':logits,'read_logits':read_logits,'completion':initial,
                'next_completion':next_done,'later_completion':later_done,'cost':cost,
                'outcomes':outcomes,'later_outcomes':later,'future_logits':future_logits,
                'future_choice':choice,'graph_edges':graph_edges,'operator_selection':selection}

    def configuration(self):return asdict(self.config)
