"""Optional temporal inductive bias and genuinely skipped inference operators.

Torch-only learner; no identity dictionary, world rules or stopping heuristic.
Time is supplied public record metadata, not a hidden-state signal. Copy-on-write
preserves earlier states; admitted writes still clone the bank. No DRAM claim.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
try:
    from .learned_agent import LearnedAgent, AgentConfig
except ImportError:
    from agent_model import LearnedAgent, AgentConfig


class SparseTemporalAgent(LearnedAgent):
    def __init__(self,config=AgentConfig(),seed=0,temporal=True,sparse=True,timestamp_feature=-1):
        super().__init__(config,seed)
        self.raw_decay=nn.Parameter(torch.full((config.read_heads,),math.log(math.expm1(1.)))) if temporal else None
        self.temporal=temporal
        self.sparse=sparse
        self.timestamp_feature=timestamp_feature

    def initial(self,batch,device=None):
        memory,occupied=super().initial(batch,device)
        if not self.temporal:return memory,occupied
        return torch.cat((memory,torch.zeros_like(memory[:,:,:1])),-1),occupied

    def encode_bank(self,public_records,occupied):
        encoded=self.encoder(public_records)*occupied[:,:,None]
        if not self.temporal:return encoded,occupied
        return torch.cat((encoded,public_records[:,:,self.timestamp_feature,None]*occupied[:,:,None]),-1),occupied

    def observe(self,state,event,present=None,force_write=False):
        if event.ndim!=2 or event.shape[-1]!=self.config.event_features or not torch.isfinite(event).all():
            raise ValueError('finite public event tensor required')
        memory,occupied=state;d=self.config.dimension
        gate=self.admission(event).squeeze(-1)
        hard=torch.ones_like(gate,dtype=torch.bool) if force_write else gate>=0
        if present is not None:hard=hard&present.bool()
        rows=hard.nonzero(as_tuple=True)[0]
        choice=torch.full((len(event),),-1,dtype=torch.long,device=event.device)
        trace={'writes':len(rows),'address':choice,'gate':hard.to(event.dtype).detach(),
               'address_rows':0,'encoder_rows':0,'bank_clone_bytes':0}
        if self.training or not self.sparse:
            address=self.address(torch.cat((event,occupied),-1));all_choice=address.argmax(-1)
            choice[rows]=all_choice[rows]
            selected=F.one_hot(all_choice,self.config.slots).to(event.dtype)*hard[:,None]
            value=self.encoder(event)
            if self.temporal:value=torch.cat((value,event[:,self.timestamp_feature,None]),-1)
            updated=memory*(1-selected[:,:,None])+selected[:,:,None]*value[:,None]
            trace.update(address_rows=len(event),encoder_rows=len(event))
            return (updated,torch.maximum(occupied,selected)),trace
        if not len(rows):return state,trace
        address=self.address(torch.cat((event[rows],occupied[rows]),-1));selected=address.argmax(-1)
        choice[rows]=selected
        value=self.encoder(event[rows])
        updated=memory.clone();used=occupied.clone()
        updated[rows,selected,:d]=value
        if self.temporal:updated[rows,selected,d]=event[rows,self.timestamp_feature]
        used[rows,selected]=1
        trace.update(address_rows=len(rows),encoder_rows=len(rows),
                     bank_clone_bytes=memory.numel()*memory.element_size()+occupied.numel()*occupied.element_size())
        return (updated,used),trace

    def read(self,state,actions,uniform=False):
        memory,occupied=state;b,a,d=actions.shape;h=self.config.read_heads
        bank=torch.cat((memory[:,:,:d],self.null[None,None].expand(b,1,-1)),1)
        valid=torch.cat((occupied>0,torch.ones(b,1,dtype=torch.bool,device=memory.device)),1)
        q=self.read_query(actions).reshape(b,a,h,d);k=self.read_key(bank).reshape(b,-1,h,d)
        logits=torch.einsum('bahd,bmhd->bahm',q,k)/math.sqrt(d)
        if self.temporal:
            stamp=memory[:,:,d]
            latest=stamp.masked_fill(~(occupied>0),-1e9).amax(-1,keepdim=True)
            latest=torch.where((occupied>0).any(-1,keepdim=True),latest,torch.zeros_like(latest))
            ages=torch.cat(((latest-stamp).clamp_min(0),torch.zeros_like(latest)),-1)
            logits=logits-F.softplus(self.raw_decay)[None,None,:,None]*ages[:,None,None]
        logits=logits.masked_fill(~valid[:,None,None],-1e9);soft=logits.softmax(-1)
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
        else:read=torch.einsum('bahm,bmd->bahd',weights,self.read_value(bank)).flatten(-2)
        return read,logits


def proper_and_decision_losses(output,completion_target,outcome_target,candidate_risk,valid,outcome_mask=None):
    """Proper scores stay unweighted; asymmetric risk belongs to decision loss.

Risk is a training label or estimator, never an execution-time argmin. Passing
privileged labels here during training does not permit deployment hidden inputs.
    Decision gradients can still reach shared representations through policy logits;
    separate proper terms do not guarantee calibration. Test it on held-out data.
"""
    completion=F.binary_cross_entropy_with_logits(output['completion'],completion_target)
    if outcome_mask is None:outcome_mask=valid[:,:,None].expand_as(outcome_target)
    outcomes=F.binary_cross_entropy_with_logits(output['outcomes'][outcome_mask],outcome_target[outcome_mask])
    probability=output['logits'].masked_fill(~valid,-1e9).softmax(-1)
    decision=(probability*candidate_risk.detach().masked_fill(~valid,0)).sum(-1).mean()
    return {'completion_proper':completion,'outcome_proper':outcomes,'decision_risk':decision}
