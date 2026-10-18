"""Torch-only DEV062A beliefs/control. No Bayes filter or repair/stopping rule."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence
def mlp(a,b,c):return nn.Sequential(nn.Linear(a,b),nn.GELU(),nn.Linear(b,c))


def cosine(a,b):return F.normalize(a,dim=-1,eps=1e-12)@F.normalize(b,dim=-1,eps=1e-12).transpose(-1,-2)


def choose(logits,valid,epsilon=1e-6):
    scores=logits.masked_fill(~valid,-1e9)
    return ((scores.max(-1,keepdim=True).values-scores<=epsilon)&valid).long().argmax(-1)


class BeliefAgent(nn.Module):
    def __init__(self,arm='tac',width=32,aux=32,seed=0):
        super().__init__();torch.manual_seed(seed);self.arm=arm;self.width=width
        self.gate=mlp(6,12,1)
        if arm in ('tac','no_casm'):
            self.local=nn.GRU(18,width,batch_first=True);self.query=mlp(2*width,width,width)
            self.key=nn.Linear(width,width);self.value=nn.Linear(width,width)
            if arm=='tac':self.ops=nn.ModuleList([mlp(width,width,width) for _ in range(3)]);self.selector=nn.Linear(width,3)
            else:self.refine=mlp(width,aux,width)
        elif arm=='gru':self.sequence=nn.GRU(18,aux,batch_first=True);self.query=mlp(aux,width,width)
        elif arm=='mlp':self.sequence=mlp(64*18,aux,width)
        else:raise ValueError(arm)
        self.belief=mlp(width,width,4);self.actor=mlp(width+4+3+2+2,width,width)
        self.future=mlp(width,width,4);self.cost=mlp(width,width,1);self.policy=mlp(width+4+1,width,1)
        self.register_buffer('temperature',torch.tensor(1.))

    def initial(self,keys):return {'keys':keys,'events':keys.new_zeros(len(keys),64,46),'present':torch.zeros(len(keys),64,dtype=torch.bool,device=keys.device)}

    def observe(self,state,event,present=None):
        take=self.gate(event[:,:6]).squeeze(-1)>=0
        if present is not None:take=take&present
        e=state['events'].clone();p=state['present'].clone()
        for row in take.nonzero(as_tuple=True)[0].tolist():
            count=int(p[row].sum())
            if count==64:e[row,:63]=e[row,1:].clone();count=63
            e[row,count]=event[row];p[row,count]=True
        return {**state,'events':e,'present':p}

    def forward(self,events,present,keys,candidates,prices):
        # Generic association to PUBLIC episode slots; no health feature parser.
        subject=cosine(events[:,:,6:22],keys);other=cosine(events[:,:,22:38],keys)
        x=torch.cat((events[:,:,:6],events[:,:,39:42],subject,other,events[:,:,38:39],events[:,:,42:43]/32,events[:,:,43:46]),-1)
        # Six type + three action + four identity similarities + bit/time/exact/sensor/price=18.
        lengths=present.sum(-1).clamp_min(1).cpu()
        if self.arm in ('tac','no_casm'):
            local=[]
            for slot in range(2):
                # Shared weights, role-relative identity binding; keep all events.
                order=torch.tensor([slot,1-slot],device=x.device)
                xx=torch.cat((x[:,:,:9],subject[:,:,order],other[:,:,order],x[:,:,13:]),-1)
                packed=pack_padded_sequence(xx,lengths,batch_first=True,enforce_sorted=False);_,h=self.local(packed);local.append(h[0])
            bank=torch.stack(local,1);query=self.query(torch.cat(local,-1));weights=(self.key(bank)*query[:,None]).sum(-1).div(self.width**.5).softmax(-1)
            latent=query+(weights[:,:,None]*self.value(bank)).sum(1)
            if self.arm=='tac':
                selector=self.selector(latent).softmax(-1);latent=latent+sum(selector[:,i,None]*op(latent) for i,op in enumerate(self.ops))
            else:latent=latent+self.refine(latent)
        elif self.arm=='gru':
            packed=pack_padded_sequence(x,lengths,batch_first=True,enforce_sorted=False);_,h=self.sequence(packed);latent=self.query(h[0])
        else:
            padded=x.new_zeros(len(x),64,18);padded[:,:x.shape[1]]=x*present[:,:,None];latent=self.sequence(padded.flatten(1))
        posterior=self.belief(latent).softmax(-1)
        binding=cosine(candidates[:,:,3:],keys)
        actor=self.actor(torch.cat((latent[:,None].expand(-1,5,-1),posterior[:,None].expand(-1,5,-1),candidates[:,:,:3],binding,prices[:,None].expand(-1,5,-1)),-1))
        future=self.future(actor);cost=self.cost(actor).squeeze(-1);logits=self.policy(torch.cat((actor,future.softmax(-1),cost[:,:,None]),-1)).squeeze(-1)
        return {'posterior':posterior,'future':future,'cost':cost,'logits':logits,'latent':latent}

    def decide(self,state,candidates,prices):return self(state['events'],state['present'],state['keys'],candidates,prices)

    def choice(self,output):return choose(output['logits'],torch.ones_like(output['logits'],dtype=torch.bool))


def configurations():
    target=sum(p.numel() for p in BeliefAgent().parameters());result={}
    for arm in ('tac','no_casm','gru','mlp'):
        aux=min(range(4,256),key=lambda a:abs(sum(p.numel() for p in BeliefAgent(arm,32,a).parameters())-target)) if arm!='tac' else 32
        m=BeliefAgent(arm,32,aux);count=sum(p.numel() for p in m.parameters());assert abs(count/target-1)<=.05
        result[arm]={'width':32,'aux':aux,'parameters':count}
    return result
