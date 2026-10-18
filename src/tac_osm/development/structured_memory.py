"""DEV061 research learners and explicitly labeled structural controls.

Entity variants impose PUBLIC categorical-ID routing; they do not learn entity
association. GRU updates, admission, CDL, CASM, predictions and action logits are
learned. PublicFeatureMLP deliberately contains a public feature extractor; it
is a comparator, not a fully learned observation-to-state architecture.
"""
from dataclasses import asdict
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
try:
    from .learned_agent import LearnedAgent,AgentConfig,mlp
    from .temporal_agent import SparseTemporalAgent
except ImportError:
    from agent_model import LearnedAgent,AgentConfig,mlp
    from temporal_model import SparseTemporalAgent


class MemoryAgent(SparseTemporalAgent):
    def __init__(self,config,arm='generic',seed=0):
        super().__init__(AgentConfig(**config),seed,temporal='aging' in arm,sparse=True)
        self.arm=arm;d=self.config.dimension;self.entity=arm!='generic';self.retention='retention' in arm
        self.state_probe=mlp(2*d,d,1)
        if self.entity:
            del self.address
            self.local=nn.GRU(22,d,batch_first=True)
        if self.retention:
            self.keep=mlp(22+d,d,1);nn.init.constant_(self.keep[-1].bias,2.)
        self.register_buffer('retention_ready',torch.tensor(False))
        if arm=='structured_gru':
            for name in ('graph_q','graph_k','graph_v','graph_norm','operator','operator_selector','halt'):delattr(self,name)
        self.training_diagnostics={}

    def initial(self,batch,device=None):
        p=next(self.parameters());device=device or p.device
        return torch.zeros(batch,32,self.config.dimension+22,device=device),torch.zeros(batch,33 if self.entity else 32,device=device)

    def observe(self,state,event,present=None,force_write=False):
        memory,occupied=state;d=self.config.dimension
        if not self.entity:
            core,trace=super().observe((memory[:,:,:d],occupied),event,present,force_write)
            raw=memory[:,:,d:];rows=(trace['gate']>0).nonzero(as_tuple=True)[0]
            if len(rows):raw=raw.clone();raw[rows,trace['address'][rows]]=event[rows]
            return (torch.cat((core[0],raw),-1),core[1]),trace
        if not torch.isfinite(event).all():raise ValueError('finite public event required')
        admit=self.admission(event).squeeze(-1)>=0
        if force_write:admit=torch.ones_like(admit)
        if present is not None:admit=admit&present.bool()
        rows=admit.nonzero(as_tuple=True)[0];address=torch.full((len(event),),-1,dtype=torch.long,device=event.device)
        if not len(rows):return state,{'gate':admit,'writes':0,'address':address,'episodic_writes':0}
        e=event[rows];slot=e[:,4:12].argmax(-1)+8*(e[:,:4].argmax(-1)==1).long();address[rows]=slot
        _,hidden=self.local(e[:,None],memory[rows,slot,:d][None].contiguous());hidden=hidden[0]
        updated=memory.clone();used=occupied.clone();updated[rows,slot]=torch.cat((hidden,e),-1);used[rows,slot]=1
        keep=torch.ones(len(rows),dtype=torch.bool,device=event.device)
        if self.retention and bool(self.retention_ready):keep=self.keep(torch.cat((e,hidden),-1)).squeeze(-1)>=0
        chosen=rows[keep];cursor=used[chosen,32].long()%16;where=16+cursor
        updated[chosen,where]=torch.cat((self.encoder(event[chosen]),event[chosen]),-1)
        used[chosen,where]=1;used[chosen,32]+=1
        return (updated,used),{'gate':admit,'writes':len(rows),'address':address,'episodic_writes':len(chosen)}

    def training_state(self,data):
        d=self.config.dimension
        if not self.entity:
            raw=data['bank'];occ=data['occupied'];memory=torch.cat((self.encoder(raw)*occ[:,:,None],raw),-1)
            return memory,occ
        sequences=data['routed'];counts=data['route_counts'];b,m,t,_=sequences.shape
        packed=pack_padded_sequence(sequences.reshape(b*m,t,22),counts.reshape(-1).clamp_min(1).cpu(),batch_first=True,enforce_sorted=False)
        encoded,hidden=self.local(packed);encoded,_=pad_packed_sequence(encoded,batch_first=True,total_length=t)
        hidden=hidden[0].reshape(b,m,d);occ=(counts>0).to(hidden.dtype)
        last=sequences.gather(2,(counts-1).clamp_min(0)[:,:,None,None].expand(-1,-1,1,22)).squeeze(2)
        memory=torch.cat((hidden*occ[:,:,None],last),-1)
        memory=torch.cat((memory,torch.zeros(b,16,d+22,device=memory.device)),1)
        used=torch.cat((occ,torch.zeros(b,17,device=memory.device)),1)
        events=data['signals'];valid=data['signal_present'];route=data['signal_route'];rank=data['signal_rank']
        outputs=encoded.reshape(b,m,t,d)
        prefix=outputs[torch.arange(b)[:,None],route,rank]
        keep_logits=self.keep(torch.cat((events,prefix),-1)).squeeze(-1) if self.retention else torch.ones_like(valid,dtype=hidden.dtype)*10
        if self.retention and bool(self.retention_ready):
            probability=keep_logits.sigmoid();hard=(keep_logits>=0).to(probability.dtype)*valid
            gate=hard.detach()+probability-probability.detach()
            gate=gate*valid
        else:gate=valid.to(hidden.dtype);hard=gate
        values=torch.cat((self.encoder(events),events),-1);counter=torch.zeros(b,dtype=torch.long)
        for index in range(events.shape[1]):
            slot=16+counter%16;select=F.one_hot(slot,32).to(hidden.dtype)*gate[:,index,None]
            memory=memory*(1-select[:,:,None])+select[:,:,None]*values[:,index,None]
            used=torch.cat((torch.maximum(used[:,:32],select.detach()),used[:,32:]),-1)
            counter=counter+(hard[:,index]>0).long()
        used=torch.cat((used[:,:32],counter[:,None].to(hidden.dtype)),1)
        self.training_diagnostics={'keep_logits':keep_logits,'events':events,'valid':valid}
        return memory,used

    def read(self,state,actions,uniform=False):
        memory,occupied=state;d=self.config.dimension
        core=memory[:,:,:d]
        if self.temporal:core=torch.cat((core,memory[:,:,d+21:d+22]),-1)
        return super().read((core,occupied[:,:32]),actions,uniform)

    def forward(self,state,descriptors,valid,queries,query_valid,lesion='none'):
        mode='no_casm' if self.arm=='structured_gru' else lesion
        out=LearnedAgent.forward(self,state,descriptors,valid,queries,query_valid,mode)
        reads,scores=self.read(state,self.query_encoder(queries))
        out['current_logits']=self.state_probe(reads).squeeze(-1);out['state_read_logits']=scores
        return out


class PublicFeatureMLP(nn.Module):
    """Strong reference: programmed public latest-state features, learned action."""
    def __init__(self,width=358,seed=0):
        super().__init__();torch.manual_seed(seed)
        self.network=mlp(11,width,width);self.action=nn.Linear(width,1);self.done=nn.Linear(width,1)
        self.current=mlp(2,16,1);self.width=width;self.last_work={}

    def initial(self,batch):return torch.zeros(batch,16,22),torch.zeros(batch,16)

    def observe(self,state,event,present=None,force_write=False):
        raw,occupied=state;kind=event[:,:4].argmax(-1);keep=kind!=3
        if present is not None:keep=keep&present.bool()
        rows=keep.nonzero(as_tuple=True)[0];slot=event[:,4:12].argmax(-1)+8*(kind==1).long()
        raw=raw.clone();occupied=occupied.clone();raw[rows,slot[rows]]=event[rows];occupied[rows,slot[rows]]=1
        return (raw,occupied),{'gate':keep,'writes':len(rows),'address':slot}

    def training_state(self,data):return data['latest'],data['latest_present']

    def forward(self,state,descriptors,valid,queries,query_valid,lesion='none'):
        raw,occupied=state;b,a,_=descriptors.shape;row=torch.arange(b)[:,None]
        key=descriptors[:,:,3:].argmax(-1);parent=raw[:,8:,12:20].argmax(-1)
        statebits=raw[:,:8,20];target=statebits.gather(1,key);parents=parent.gather(1,key)
        parentbits=statebits.gather(1,parents);exists=occupied[:,8:].gather(1,key)
        known=occupied[:,:8].gather(1,key);parentknown=occupied[:,:8].gather(1,parents)
        total=(statebits*occupied[:,:8]).sum(-1,keepdim=True).expand(-1,a)
        extra=torch.stack((target,parentbits,exists,(total>0).float(),known,parentknown,total/6,occupied[:,:8].any(-1)[:,None].expand(-1,a).float()),-1)
        features=torch.cat((descriptors[:,:,:3],extra),-1);hidden=F.gelu(self.network(features))
        logits=self.action(hidden).squeeze(-1).masked_fill(~valid,-1e9)
        pooled=(hidden*valid[:,:,None]).sum(1)/valid.sum(-1,keepdim=True)
        query=queries.argmax(-1);values=statebits[row,query];present=occupied[:,:8][row,query]
        current=self.current(torch.stack((values,present),-1)).squeeze(-1)
        self.last_work={k:0 for k in ('read_candidates','retained_read_slots','operator_calls','latent_transitions')}
        return {'logits':logits,'completion':self.done(pooled).squeeze(-1),'current_logits':current}


def create(arm,config,seed=0):
    return PublicFeatureMLP(config['width'],seed) if arm=='public_mlp' else MemoryAgent(config,arm,seed)


def configurations():
    def count(m):return sum(p.numel() for p in m.parameters())
    base=asdict(AgentConfig());target=count(create('generic',base));result={'generic':base}
    for arm in ('entity','entity_aging','entity_retention','entity_aging_retention','structured_gru'):
        candidates=[]
        for d in range(36,73):
            config=asdict(AgentConfig(dimension=d));candidates.append((abs(count(create(arm,config))-target),config))
        result[arm]=min(candidates,key=lambda x:x[0])[1]
    widths=[(abs(count(PublicFeatureMLP(w))-target),w) for w in range(280,440)]
    result['public_mlp']={'width':min(widths)[1]}
    assert all(abs(count(create(a,c))/target-1)<=.05 for a,c in result.items())
    return result
