"""DEV059 retrained neural controls. No world, teacher or task-policy imports.

The attention reference deliberately stores every raw public event using a
programmed append buffer. GRU processes every event. Neither has admission,
CDL-style candidate-conditioned top-k reads, or CASM. These are controls, not
claims of learned storage/addressing. Full-agent deployment remains unchanged.
"""
from dataclasses import asdict
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence
try:
    from .learned_agent import LearnedAgent, AgentConfig, mlp
except ImportError:  # standalone exported module
    from agent_model import LearnedAgent, AgentConfig, mlp


class NoCASM(LearnedAgent):
    def __init__(self, config, seed=0):
        super().__init__(AgentConfig(**config), seed)
        for name in ('graph_q','graph_k','graph_v','graph_norm','operator','operator_selector','halt'):
            delattr(self,name)

    def forward(self,state,descriptors,valid,queries,query_valid,lesion='none'):
        if lesion!='none':raise ValueError('DEV059 controls are retrained, not runtime lesions')
        return super().forward(state,descriptors,valid,queries,query_valid,lesion='no_casm')


class DenseController(nn.Module):
    def __init__(self,config,seed=0):
        super().__init__();torch.manual_seed(seed);self.spec=config;d=config['dimension']
        self.kind=config['kind'];self.last_work={}
        if self.kind=='gru':
            self.gru=nn.GRU(22,config['hidden'],batch_first=True)
            self.project=mlp(config['hidden'],d,d)
        else:
            self.event_encoder=mlp(22,d,d)
            self.attention=nn.ModuleList([nn.MultiheadAttention(d,4,dropout=0.,batch_first=True) for _ in range(2)])
            self.mix=nn.ModuleList([mlp(d,config['feedforward'],d) for _ in range(2)])
            self.norm=nn.ModuleList([nn.LayerNorm(d) for _ in range(2)])
            self.pool=nn.Linear(d,1);self.null=nn.Parameter(torch.randn(1,1,d)*.02)
        self.action_encoder=mlp(11,d,d);self.query_encoder=mlp(8,d,d)
        self.node=mlp(2*d,d,d);self.transition=mlp(2*d,2*d,d)
        self.current_verifier=mlp(d,d,1);self.verifier=mlp(d,d,1)
        self.cost=mlp(d,d,1);self.outcome=mlp(2*d,d,1)
        self.future_policy=mlp(2*d,d,1);self.policy=mlp(2*d+3,2*d,1)

    def configuration(self):return self.spec.copy()

    def initial(self,batch):
        if self.kind=='gru':return torch.zeros(batch,1,self.spec['hidden']),torch.zeros(batch,1)
        return torch.zeros(batch,self.spec['capacity'],22),torch.zeros(batch,self.spec['capacity'])

    def observe(self,state,event,present=None,force_write=False):
        if force_write:raise ValueError('raw controls already process all observations')
        if not torch.isfinite(event).all():raise ValueError('public event must be finite')
        memory,occupied=state
        keep=torch.ones(len(event),dtype=torch.bool,device=event.device) if present is None else present
        if self.kind=='gru':
            _,h=self.gru(event[:,None],memory.transpose(0,1).contiguous())
            memory=torch.where(keep[:,None,None],h.transpose(0,1),memory)
            occupied=occupied+keep[:,None]
            address=torch.zeros(len(event),dtype=torch.long)
        else:
            address=occupied.sum(-1).long()
            if ((address>=self.spec['capacity'])&keep).any():raise ValueError('raw reference buffer exhausted')
            selected=F.one_hot(address.clamp_max(self.spec['capacity']-1),self.spec['capacity']).to(event.dtype)*keep[:,None]
            memory=memory*(1-selected[:,:,None])+selected[:,:,None]*event[:,None]
            occupied=torch.maximum(occupied,selected)
        return (memory,occupied),{'gate':keep,'writes':int(keep.sum()),'address':address}

    def training_state(self,data):
        events,present=data['events'],data['present']
        if self.kind=='gru':
            packed=pack_padded_sequence(events,present.sum(-1).cpu(),batch_first=True,enforce_sorted=False)
            _,h=self.gru(packed)
            return h.transpose(0,1),present.sum(-1,keepdim=True).float()
        return events,present.float()

    def context(self,state):
        if self.kind=='gru':return self.project(state[0][:,0]),None
        raw,occupied=state
        length=max(1,int(occupied.sum(-1).max()))
        encoded=self.event_encoder(raw[:,:length]);valid=occupied[:,:length]>0
        encoded=torch.cat((encoded,self.null.expand(len(raw),-1,-1)),1)
        valid=torch.cat((valid,torch.ones(len(raw),1,dtype=torch.bool)),1)
        weights=self.pool(encoded).squeeze(-1).masked_fill(~valid,-1e9).softmax(-1)
        return torch.einsum('bl,bld->bd',weights,encoded),(encoded,valid)

    def forward(self,state,descriptors,valid,queries,query_valid,lesion='none'):
        if lesion!='none':raise ValueError('retrained controls do not use runtime lesions')
        b,a,_=descriptors.shape
        context,raw=self.context(state);action=self.action_encoder(descriptors)
        node=self.node(torch.cat((context[:,None].expand(-1,a,-1),action),-1))
        if raw is not None:
            bank,present=raw
            for attention,mix,norm in zip(self.attention,self.mix,self.norm):
                acquired,_=attention(node,bank,bank,key_padding_mask=~present,need_weights=False)
                node=norm(node+acquired);node=node+mix(node)
        initial=self.current_verifier(context).squeeze(-1)
        z1=context[:,None]+self.transition(torch.cat((context[:,None].expand(-1,a,-1),node),-1))
        future=self.future_policy(torch.cat((z1[:,:,None].expand(-1,-1,a,-1),action[:,None].expand(-1,a,-1,-1)),-1)).squeeze(-1)
        future=future.masked_fill(~valid[:,None],-1e9)
        probability=future.softmax(-1);choice=probability.argmax(-1)
        hard=F.one_hot(choice,a).to(probability.dtype)
        weight=hard.detach()+probability-probability.detach() if self.training else hard
        proposed=torch.einsum('bac,bcd->bad',weight,action)
        z2=z1+self.transition(torch.cat((z1,proposed),-1))
        next_done=self.verifier(z1).squeeze(-1);later_done=self.verifier(z2).squeeze(-1)
        cost=self.cost(z1).squeeze(-1);q=self.query_encoder(queries)
        pair=lambda z:torch.cat((z[:,:,None].expand(-1,-1,q.shape[1],-1),q[:,None].expand(-1,a,-1,-1)),-1)
        prediction=self.outcome(pair(z1)).squeeze(-1);later=self.outcome(pair(z2)).squeeze(-1)
        features=torch.cat((z1,z2,initial[:,None,None].expand(-1,a,-1),next_done[:,:,None],cost[:,:,None]),-1)
        logits=self.policy(features).squeeze(-1).masked_fill(~valid,-1e9)
        lengths=state[1].sum(-1) if self.kind=='attention' else torch.zeros(b)
        self.last_work={'read_candidates':int((8*(lengths+1)*valid.sum(-1)+lengths+1).sum()) if self.kind=='attention' else 0,
                        'retained_read_slots':int(lengths.sum()),'operator_calls':0,'latent_transitions':int(valid.sum())*2}
        return {'logits':logits,'completion':initial,'next_completion':next_done,'later_completion':later_done,
                'cost':cost,'outcomes':prediction,'later_outcomes':later,'future_logits':future,'future_choice':choice,
                'graph_edges':[],'operator_selection':[]}


def create(arm,config,seed):
    if arm.startswith('full'):return LearnedAgent(AgentConfig(**config),seed)
    if arm=='no_casm_stop':return NoCASM(config,seed)
    return DenseController(config,seed)


def configurations(target=133580):
    def count(model):return sum(p.numel() for p in model.parameters())
    result={'full_common':asdict(AgentConfig()),'full_stop':asdict(AgentConfig())}
    candidates=[]
    for dimension in range(40,81):
        config=asdict(AgentConfig(dimension=dimension,depth=0))
        candidates.append((abs(count(NoCASM(config))-target),config))
    result['no_casm_stop']=min(candidates,key=lambda row:row[0])[1]
    candidates=[]
    for hidden in range(48,225):
        config={'kind':'gru','dimension':48,'hidden':hidden,'capacity':192}
        candidates.append((abs(count(DenseController(config))-target),config))
    result['gru_stop']=min(candidates,key=lambda row:row[0])[1]
    candidates=[]
    for dimension in range(40,81,4):
        for ff in range(dimension,4*dimension+1,8):
            config={'kind':'attention','dimension':dimension,'feedforward':ff,'capacity':192}
            candidates.append((abs(count(DenseController(config))-target),config))
    result['attention_stop']=min(candidates,key=lambda row:row[0])[1]
    for arm,config in result.items():
        if abs(count(create(arm,config,0))/target-1)>.05:raise AssertionError('parameter match failed')
    return result
