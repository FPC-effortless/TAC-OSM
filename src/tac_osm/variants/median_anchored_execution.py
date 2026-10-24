"""Three-regime physical input/output wrapper; no learned parameters."""
import torch
from .clock_value_world import ClockValueState

def execution_quantiles(prediction):
    # The learned median is already explicit. Invert only the requested tails.
    from .multimode_distribution import mixture_quantiles
    lower,upper=mixture_quantiles(prediction,(.05,.95))
    return lower,prediction['median'].double(),upper


class PhysicalMedianAnchoredGraph(torch.nn.Module):
    """Pure recorded-units → fixed preprocessing → native graph → recorded units.

Normative original preprocessing is float32, followed by float64 reference
inference. Caller supplies explicit normalized ClockValueState values. This
adds no learned weights, mutable cache, authority or future observations.
"""
    def __init__(self,model):
        super().__init__()
        if next(model.parameters()).dtype!=torch.float64:raise ValueError('normative reference model must be float64')
        self.model=model

    def forward(self,state_qpos,known_reference_qpos,elapsed,pixels,state_values):
        x=((state_qpos.float()-self.model.state_mean.float())/self.model.state_scale.float()).double()
        a=((known_reference_qpos.float()-self.model.action_mean.float())/self.model.action_scale.float()).double()
        images=(pixels.float()/255).double()
        raw,state=self.model.visual_tensor_graph(x,a,elapsed.float().double(),images,ClockValueState(*state_values),last_only=True)
        q=execution_quantiles(raw)
        return tuple(v*self.model.state_scale+self.model.state_mean for v in q)+(raw['motion_logit'].sigmoid(),*tuple(state))
