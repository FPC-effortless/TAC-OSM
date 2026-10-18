"""DEV061 optional same-weight inference optimization; no policy change."""
import torch
try:
    from .structured_memory import MemoryAgent
    from .temporal_agent import SparseTemporalAgent
except ImportError:
    from memory_model import MemoryAgent
    from temporal_model import SparseTemporalAgent


class InferenceMemoryAgent(MemoryAgent):
    def observe(self,state,event,present=None,force_write=False):
        if self.training or self.entity:return super().observe(state,event,present,force_write)
        memory,occupied=state;d=self.config.dimension
        core,trace=SparseTemporalAgent.observe(self,(memory[:,:,:d],occupied),event,present,force_write)
        if not trace['writes']:return state,trace
        raw=memory[:,:,d:].clone();rows=(trace['gate']>0).nonzero(as_tuple=True)[0]
        raw[rows,trace['address'][rows]]=event[rows]
        return (torch.cat((core[0],raw),-1),core[1]),trace
