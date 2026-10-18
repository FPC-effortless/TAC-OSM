"""Batch available public packets; trained state/policy semantics stay unchanged."""
import torch
from torch.nn.utils.rnn import pack_padded_sequence
try:
    from .belief_agent import BeliefAgent
    from .cached_belief_agent import CachedBeliefAgent
except ImportError:
    from agent_model import BeliefAgent
    from cached_model import CachedBeliefAgent


def append_packet(state,events,present,gate):
    take=gate(events[:,:,:6]).squeeze(-1)>=0
    if present is not None:take=take&present
    if not take.any():return state,take
    memory=state['events'].clone();mask=state['present'].clone()
    for row in take.any(-1).nonzero(as_tuple=True)[0].tolist():
        records=torch.cat((memory[row,mask[row]],events[row,take[row]]),0)[-64:]
        memory[row].zero_();mask[row].zero_();memory[row,:len(records)]=records;mask[row,:len(records)]=True
    return {**state,'events':memory,'present':mask},take


class PacketObserver:
    """Single-event compatibility defers only until the next explicit read."""
    def initial(self,keys):return {**super().initial(keys),'pending_events':[],'pending_masks':[]}

    def observe(self,state,event,present=None):
        present=torch.ones(len(event),dtype=torch.bool,device=event.device) if present is None else present
        return {**state,'pending_events':state['pending_events']+[event],'pending_masks':state['pending_masks']+[present]}

    def flush(self,state):
        if state['pending_events']:
            updated=self.observe_many(state,torch.stack(state['pending_events'],1),torch.stack(state['pending_masks'],1));state.update(updated);state['pending_events']=[];state['pending_masks']=[]
        return state

    def decide(self,state,candidates,prices):return super().decide(self.flush(state),candidates,prices)


class BatchedBeliefAgent(PacketObserver,BeliefAgent):
    def observe_many(self,state,events,present=None):return append_packet(state,events,present,self.gate)[0]


class BatchedCachedBeliefAgent(PacketObserver,CachedBeliefAgent):
    def observe_many(self,state,events,present=None):
        updated,take=append_packet(state,events,present,self.gate)
        if self.arm=='mlp' or not take.any():return updated
        counts=state['present'].sum(-1);lengths=take.sum(-1);overflow=counts+lengths>64;ready=state['cache_ready'];active=(lengths>0)&ready&~overflow;hidden=state['cached_hidden'].clone()
        if active.any():
            raw=self.features(events[active],state['keys'][active]);selected=take[active];lens=lengths[active];maximum=int(lens.max());x=raw.new_zeros(len(raw),maximum,18)
            for row in range(len(raw)):x[row,:int(lens[row])]=raw[row,selected[row]]
            old=hidden[active];old=torch.where((counts[active]>0)[:,None,None],old,torch.zeros_like(old))
            if self.arm=='gru':
                packed=pack_padded_sequence(x,lens.cpu(),batch_first=True,enforce_sorted=False);_,h=self.sequence(packed,old[:,0][None]);new=h[0][:,None]
            else:
                b=len(x);relative=torch.stack([self.relative(x,i) for i in range(2)],1).reshape(b*2,maximum,18);packed=pack_padded_sequence(relative,lens[:,None].expand(-1,2).reshape(-1).cpu(),batch_first=True,enforce_sorted=False);_,h=self.local(packed,old.reshape(b*2,self.width)[None]);new=h[0].reshape(b,2,self.width)
            hidden[active]=new
        return {**updated,'cached_hidden':hidden,'cache_ready':ready&~overflow}
