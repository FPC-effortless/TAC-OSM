"""Pure complete/context-aware physical graph and optional query selection."""
import math
import torch
from .context_filter_world import ContextFilterState
from .median_anchored_execution import execution_quantiles


def capture_clock_features(clock):
    periods=clock.new_tensor((1.,4.,16.),dtype=torch.float64)
    phase=torch.remainder(clock.double(),periods)*(2*math.pi/periods)
    return torch.stack((phase.sin(),phase.cos()),-1).flatten(-2).float().double()


class NormalizedContextGraph(torch.nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,x,a,dt,images,task_ids,clock_context,state_values):
        p,s=self.model.visual_tensor_graph(x,a,dt,images,ContextFilterState(*state_values),last_only=True,task_ids=task_ids,clock_context=clock_context)
        q=execution_quantiles(p)
        return (*q,p['motion_logit'].sigmoid(),*tuple(s))


class PhysicalContextGraph(torch.nn.Module):
    def __init__(self,model):
        super().__init__();self.model=model
        if next(model.parameters()).dtype!=torch.float64:raise ValueError('float64 original reference required')
    def forward(self,x,a,dt,pixels,task_ids,capture_clock,state_values):
        xx=((x.float()-self.model.state_mean.float())/self.model.state_scale.float()).double()
        aa=((a.float()-self.model.action_mean.float())/self.model.action_scale.float()).double()
        context=capture_clock_features(capture_clock)
        p,s=self.model.visual_tensor_graph(xx,aa,dt.float().double(),(pixels.float()/255).double(),ContextFilterState(*state_values),last_only=True,task_ids=task_ids,clock_context=context)
        q=execution_quantiles(p)
        return tuple(v*self.model.state_scale+self.model.state_mean for v in q)+(p['motion_logit'].sigmoid(),*tuple(s))


class ContextInitializer(torch.nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,first_physical_qpos,task_ids):
        x=((first_physical_qpos.float()-self.model.state_mean.float())/self.model.state_scale.float()).double()
        return tuple(self.model.initial_state(x,task_ids))


class PointContextGraph(torch.nn.Module):
    def __init__(self,complete):super().__init__();self.complete=complete
    def forward(self,x,a,dt,pixels,task_ids,capture_clock,state_values):
        full=self.complete(x,a,dt,pixels,task_ids,capture_clock,state_values)
        return (full[1],full[3],*full[4:])
