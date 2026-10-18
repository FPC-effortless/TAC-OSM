"""DEV061.5 torch-only policies. No teacher, physics, repair rule or stop override.

Declared priors: public episode entity manifest, cosine binding to ephemeral
manifest slots, typed record admission and separate relationship-key stream.
Content/state updates, CDL strengths, relational computation, predictions and
policy are neural. This is NOT learned entity discovery or symbolic CASM.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence


def mlp(a,b,c):return nn.Sequential(nn.Linear(a,b),nn.GELU(),nn.Linear(b,c))


def cosine(a,b):
    return F.normalize(a,dim=-1,eps=1e-12)@F.normalize(b,dim=-1,eps=1e-12).transpose(-1,-2)


def choose(logits,valid,epsilon=1e-6):
    """First supplied valid choice within a registered numerical score margin."""
    scores=logits.masked_fill(~valid,-1e9);top=scores.max(-1,keepdim=True).values
    return ((top-scores<=epsilon)&valid).long().argmax(-1)


class IdentityAgent(nn.Module):
    def __init__(self,arm='split_robust',width=32,aux_width=64,seed=0):
        super().__init__();torch.manual_seed(seed);self.arm=arm;self.width=width;self.aux_width=aux_width
        self.fixed=arm.startswith('fixed');self.gru=arm.startswith('raw_gru');self.pooled=arm.startswith('raw_mlp');d=width
        self.admission=mlp(4,16,1);self.relation_gate=mlp(4,16,1)
        if self.gru:
            self.sequence=nn.GRU(38,aux_width,batch_first=True);self.query=mlp(aux_width+16,d,d)
        elif self.pooled:
            self.event=mlp(54,aux_width,aux_width);self.query=mlp(2*aux_width,d,d)
        else:
            if self.fixed:self.identity_projection=nn.Linear(32,4)
            self.local=nn.GRU(10 if self.fixed else 6,d,batch_first=True)
        self.match_strength=nn.Parameter(torch.tensor(24.));self.null_bias=nn.Parameter(torch.tensor(4.))
        self.has_casm=arm not in ('no_casm_robust','raw_gru_robust','raw_mlp_robust')
        if self.has_casm:
            self.graph_q=nn.ModuleList([nn.Linear(d,d) for _ in range(2)]);self.graph_k=nn.ModuleList([nn.Linear(d,d) for _ in range(2)])
            self.graph_v=nn.ModuleList([nn.Linear(d,d) for _ in range(2)]);self.norm=nn.ModuleList([nn.LayerNorm(d) for _ in range(2)])
            self.operators=nn.ModuleList([mlp(d,d,d) for _ in range(3)]);self.selector=nn.Linear(d,3)
        else:self.refine=mlp(3*d,aux_width,d)
        self.relate=mlp(3*d,d,d);self.actor=mlp(2*d+3+(4 if self.fixed else 0),d,d)
        self.transition=mlp(d,d,d);self.refuture=mlp(2*d,d,d);self.outcome=mlp(2*d+1,d,1)
        self.current=mlp(d,d,1);self.complete=mlp(d,d,1);self.done=mlp(d,d,1);self.cost=mlp(d,d,1)
        self.policy=mlp(3*d+4,d,1)
        self.register_buffer('temperature',torch.tensor(1.))

    def initial(self,keys,entity_valid):
        """Public identity manifest only; no health/relations/hidden state."""
        return {'keys':keys,'entity_valid':entity_valid,'events':keys.new_zeros(len(keys),64,38),'present':torch.zeros(len(keys),64,dtype=torch.bool,device=keys.device)}

    def observe(self,state,event,present=None):
        admit=self.admission(event[:,:4]).squeeze(-1)>=0
        if present is not None:admit=admit&present
        memory=state['events'].clone();mask=state['present'].clone()
        for row in admit.nonzero(as_tuple=True)[0].tolist():
            count=int(mask[row].sum())
            if count==64:memory[row,:63]=memory[row,1:].clone();count=63
            memory[row,count]=event[row];mask[row,count]=True
        return {**state,'events':memory,'present':mask},{'writes':int(admit.sum()),'gate':admit}

    def encode(self,events,present,keys,entity_valid):
        events=torch.cat((events[:,:,:37],events[:,:,37:38]/256),-1)
        b,t,_=events.shape;n=keys.shape[1];d=self.width
        # Binding is a generic cosine structural prior, never a category decoder.
        sim=cosine(events[:,:,4:20],keys).masked_fill(~entity_valid[:,None],-1e9)
        routed=sim.argmax(-1)
        relation=(self.relation_gate(events[:,:,:4]).squeeze(-1)>=0)&present
        parent=events.new_zeros(b,n,16)
        for step in range(t):
            rows=relation[:,step].nonzero(as_tuple=True)[0]
            parent[rows,routed[rows,step]]=events[rows,step,20:36]
        if self.gru:
            packed=pack_padded_sequence(events,present.sum(-1).clamp_min(1).cpu(),batch_first=True,enforce_sorted=False)
            _,h=self.sequence(packed);content=self.query(torch.cat((h[0][:,None].expand(-1,n,-1),keys),-1))
        elif self.pooled:
            inp=torch.cat((events[:,None].expand(-1,n,-1,-1),keys[:,:,None].expand(-1,-1,t,-1)),-1)
            h=self.event(inp);mask=present[:,None,:,None]
            mean=(h*mask).sum(2)/mask.sum(2).clamp_min(1);maximum=h.masked_fill(~mask,-1e9).max(2).values
            content=self.query(torch.cat((mean,maximum),-1))
        else:
            # Exact route grouping caches raw histories, not current-state facts.
            source=torch.cat((events[:,:,:4],events[:,:,36:38]),-1)
            if self.fixed:source=torch.cat((source,self.identity_projection(events[:,:,4:36])),-1)
            counts=((routed[:,None]==torch.arange(n,device=events.device)[None,:,None])&present[:,None]&entity_valid[:,:,None]).sum(-1)
            length=max(1,int(counts.max()));sequences=events.new_zeros(b,n,length,source.shape[-1])
            for slot in range(n):
                take=(routed==slot)&present&entity_valid[:,slot,None];rank=take.cumsum(-1)-1
                rows,steps=take.nonzero(as_tuple=True);sequences[rows,slot,rank[rows,steps]]=source[rows,steps]
            packed=pack_padded_sequence(sequences.reshape(b*n,length,-1),counts.reshape(-1).clamp_min(1).cpu(),batch_first=True,enforce_sorted=False)
            _,h=self.local(packed);content=h[0].reshape(b,n,d)
        content=content*entity_valid[:,:,None]
        strength=F.softplus(self.match_strength)
        scores=strength*cosine(parent,keys);scores=scores.masked_fill(~entity_valid[:,None],-1e9)
        null=self.null_bias.expand(b,n,1);attention=torch.cat((scores,null),-1).softmax(-1)
        bank=torch.cat((content,content.new_zeros(b,1,d)),1);linked=attention@bank
        global_state=content.sum(1)/entity_valid.sum(-1,keepdim=True).clamp_min(1)
        node=self.relate(torch.cat((content,linked,global_state[:,None].expand(-1,n,-1)),-1))
        if self.has_casm:
            for depth in range(2):
                graph=self.graph_q[depth](node)@self.graph_k[depth](node).transpose(-1,-2)/(d**.5)
                graph=graph+strength*cosine(parent,keys);graph=graph.masked_fill(~entity_valid[:,None],-1e9)
                node=self.norm[depth](node+graph.softmax(-1)@self.graph_v[depth](node))
                weights=self.selector(node).softmax(-1)
                node=node+sum(weights[:,:,i,None]*op(node) for i,op in enumerate(self.operators))
        else:node=node+self.refine(torch.cat((content,linked,global_state[:,None].expand(-1,n,-1)),-1))
        return node,content,attention

    def forward(self,events,present,keys,entity_valid,candidates,valid):
        node,content,read=self.encode(events,present,keys,entity_valid);b,n,d=node.shape
        strength=F.softplus(self.match_strength);score=strength*cosine(candidates[:,:,3:],keys)
        score=score.masked_fill(~entity_valid[:,None],-1e9);weights=score.softmax(-1);local=weights@node
        global_state=(node*entity_valid[:,:,None]).sum(1)/entity_valid.sum(-1,keepdim=True).clamp_min(1)
        parts=[local,global_state[:,None].expand(-1,len(valid[0]),-1),candidates[:,:,:3]]
        if self.fixed:parts.append(self.identity_projection(torch.cat((candidates[:,:,3:],torch.zeros_like(candidates[:,:,3:])),-1)))
        actor=self.actor(torch.cat(parts,-1));future=self.transition(actor);later=self.refuture(torch.cat((future,actor),-1))
        current=self.current(content).squeeze(-1);completion=self.complete(global_state).squeeze(-1)
        pair=torch.cat((future[:,:,None].expand(-1,-1,n,-1),node[:,None].expand(-1,actor.shape[1],-1,-1),cosine(candidates[:,:,3:],keys)[:,:,:,None]),-1)
        outcomes=self.outcome(pair).squeeze(-1);done=self.done(future).squeeze(-1);cost=self.cost(future).squeeze(-1)
        risk=(outcomes.sigmoid()*entity_valid[:,None]).sum(-1)/entity_valid.sum(-1,keepdim=True).clamp_min(1)
        features=torch.cat((actor,future,later,done[:,:,None],cost[:,:,None],risk[:,:,None],completion[:,None,None].expand(-1,actor.shape[1],-1)),-1)
        logits=self.policy(features).squeeze(-1).masked_fill(~valid,-1e9)
        return {'logits':logits,'current':current,'completion':completion,'outcomes':outcomes,'done':done,'cost':cost,'reads':read}

    def decide(self,state,candidates,valid):return self(state['events'],state['present'],state['keys'],state['entity_valid'],candidates,valid)


def configurations():
    arms=('fixed_bc','fixed_robust','split_bc','split_robust','no_casm_robust','raw_gru_robust','raw_mlp_robust')
    target=sum(p.numel() for p in IdentityAgent().parameters());result={}
    for arm in arms:
        choices=(range(16,256) if arm.startswith(('raw_','no_casm')) else (64,))
        best=min(choices,key=lambda w:abs(sum(p.numel() for p in IdentityAgent(arm,32,w).parameters())-target))
        m=IdentityAgent(arm,32,best);count=sum(p.numel() for p in m.parameters());assert abs(count/target-1)<.05,(arm,count,target)
        result[arm]={'width':32,'aux_width':best,'parameters':count}
    return result
