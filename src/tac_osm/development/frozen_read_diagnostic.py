"""DEV061 public-only read intervention. Diagnostic, NEVER a learned-model export."""
import torch
from torch.nn import functional as F
from .learned_agent import LearnedAgent


class FrozenReadDiagnostic(LearnedAgent):
    def __init__(self,config,seed=0,mode='learned'):
        super().__init__(config,seed);self.mode=mode;self.descriptors=None

    def initial(self,batch,device=None):
        memory,occupied=super().initial(batch,device)
        return torch.cat((memory,torch.zeros(batch,self.config.slots,22,device=memory.device)),-1),occupied

    def observe(self,state,event,present=None,force_write=False):
        memory,occupied=state;d=self.config.dimension
        core,trace=super().observe((memory[:,:,:d],occupied),event,present,force_write)
        selected=F.one_hot(trace['address'],self.config.slots).to(memory.dtype)*trace['gate'][:,None]
        raw=memory[:,:,d:]*(1-selected[:,:,None])+event[:,None]*selected[:,:,None]
        return (torch.cat((core[0],raw),-1),core[1]),trace

    def read(self,state,actions,uniform=False):
        memory,occupied=state;d=self.config.dimension
        if self.mode=='learned':return super().read((memory[:,:,:d],occupied),actions,uniform)
        raw=memory[:,:,d:];descriptors=self.descriptors
        key=descriptors[:,:,3:].argmax(-1);has_key=descriptors[:,:,3:].any(-1)
        match=raw[:,:,4:12].argmax(-1)[:,None,:]==key[:,:,None]
        common=match&(occupied[:,None,:]>0)&has_key[:,:,None]
        status=(raw[:,:,0]>0)|(raw[:,:,2]>0);relation=raw[:,:,1]>0
        selections=[]
        for head,kind in enumerate((status,relation)):
            eligible=common&kind[:,None,:]
            timestamp=raw[:,:,21][:,None,:].expand_as(eligible)
            score=-timestamp if self.mode=='oldest' and head==0 else timestamp
            chosen=score.masked_fill(~eligible,-1e9).argmax(-1)
            selections.append(torch.where(eligible.any(-1),chosen,torch.full_like(chosen,self.config.slots)))
        indices=torch.stack(selections,2)
        bank=torch.cat((memory[:,:,:d],self.null[None,None].expand(len(memory),1,-1)),1)
        values=bank[torch.arange(len(memory))[:,None,None],indices]
        scores=torch.full((*indices.shape,self.config.slots+1),-1e9,device=memory.device)
        scores.scatter_(-1,indices[:,:,:,None],0)
        return self.read_value(values).flatten(-2),scores

    def forward(self,state,descriptors,valid,queries,query_valid,lesion='none'):
        self.descriptors=descriptors
        return super().forward(state,descriptors,valid,queries,query_valid,lesion)
