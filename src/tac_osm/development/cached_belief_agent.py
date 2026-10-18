"""Inference-only GRU caching; semantic decisions and predictions remain learned."""
import torch
from torch.nn.utils.rnn import pack_padded_sequence
try:
    from .belief_agent import BeliefAgent,cosine
except ImportError:
    from agent_model import BeliefAgent,cosine


class CachedBeliefAgent(BeliefAgent):
    def initial(self,keys):
        state=super().initial(keys)
        if self.arm!='mlp':
            width=self.sequence.hidden_size if self.arm=='gru' else self.width
            state.update(cached_hidden=keys.new_zeros(len(keys),1 if self.arm=='gru' else 2,width),cache_ready=torch.zeros(len(keys),dtype=torch.bool,device=keys.device))
        return state

    def features(self,events,keys):
        subject=cosine(events[:,:,6:22],keys);other=cosine(events[:,:,22:38],keys)
        return torch.cat((events[:,:,:6],events[:,:,39:42],subject,other,events[:,:,38:39],events[:,:,42:43]/32,events[:,:,43:46]),-1)

    def relative(self,x,slot):
        order=torch.tensor([slot,1-slot],device=x.device)
        return torch.cat((x[:,:,:9],x[:,:,9:11][:,:,order],x[:,:,11:13][:,:,order],x[:,:,13:]),-1)

    def refresh(self,state):
        x=self.features(state['events'],state['keys']);lengths=state['present'].sum(-1).clamp_min(1).cpu()
        if self.arm=='gru':
            _,h=self.sequence(pack_padded_sequence(x,lengths,batch_first=True,enforce_sorted=False));hidden=h[0][:,None]
        else:
            values=[]
            for slot in range(2):
                _,h=self.local(pack_padded_sequence(self.relative(x,slot),lengths,batch_first=True,enforce_sorted=False));values.append(h[0])
            hidden=torch.stack(values,1)
        state['cached_hidden']=hidden;state['cache_ready']=torch.ones_like(state['cache_ready']);return state

    def observe(self,state,event,present=None):
        if self.arm=='mlp':return super().observe(state,event,present)
        # The same trained admission decision controls history and hidden updates.
        take=self.gate(event[:,:6]).squeeze(-1)>=0
        if present is not None:take=take&present
        memory=state['events'].clone();mask=state['present'].clone()
        for row in take.nonzero(as_tuple=True)[0].tolist():
            count=int(mask[row].sum())
            if count==64:memory[row,:63]=memory[row,1:].clone();count=63
            memory[row,count]=event[row];mask[row,count]=True
        updated={**state,'events':memory,'present':mask}
        counts=state['present'].sum(-1);overflow=take&(counts==64);ready=state['cache_ready'];active=take&ready&~overflow
        hidden=state['cached_hidden'].clone()
        if active.any():
            x=self.features(event[active,None],state['keys'][active]);old=hidden[active];old=torch.where((counts[active]>0)[:,None,None],old,torch.zeros_like(old))
            if self.arm=='gru':_,h=self.sequence(x,old[:,0][None]);new=h[0][:,None]
            else:
                batch=len(x);relative=torch.stack([self.relative(x,slot) for slot in range(2)],1).reshape(batch*2,1,18);_,h=self.local(relative,old.reshape(batch*2,self.width)[None]);new=h[0].reshape(batch,2,self.width)
            hidden[active]=new
        return {**updated,'cached_hidden':hidden,'cache_ready':ready&~overflow}

    def decide(self,state,candidates,prices):
        if self.arm=='mlp':return super().decide(state,candidates,prices)
        if not state['cache_ready'].all():self.refresh(state)
        h=state['cached_hidden']
        if self.arm=='gru':latent=self.query(h[:,0])
        else:
            local=[h[:,0],h[:,1]];bank=torch.stack(local,1);query=self.query(torch.cat(local,-1));weights=(self.key(bank)*query[:,None]).sum(-1).div(self.width**.5).softmax(-1);latent=query+(weights[:,:,None]*self.value(bank)).sum(1)
            if self.arm=='tac':
                selector=self.selector(latent).softmax(-1);latent=latent+sum(selector[:,i,None]*op(latent) for i,op in enumerate(self.ops))
            else:latent=latent+self.refine(latent)
        posterior=self.belief(latent).softmax(-1);binding=cosine(candidates[:,:,3:],state['keys']);actor=self.actor(torch.cat((latent[:,None].expand(-1,5,-1),posterior[:,None].expand(-1,5,-1),candidates[:,:,:3],binding,prices[:,None].expand(-1,5,-1)),-1));future=self.future(actor);cost=self.cost(actor).squeeze(-1);logits=self.policy(torch.cat((actor,future.softmax(-1),cost[:,:,None]),-1)).squeeze(-1)
        return {'posterior':posterior,'future':future,'cost':cost,'logits':logits,'latent':latent}
