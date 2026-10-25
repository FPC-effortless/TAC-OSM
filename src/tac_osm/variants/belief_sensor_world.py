"""ME101 source-aware current-state belief with exact affine marginalization.

The response is the Jacobian of the declared conditional affine emission, NOT
an alleged full neural-network Jacobian. The inherited state/scan is unchanged.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .entity_sensor_world import EntitySensorWorld,SensorForecast


class BeliefSensorWorld(EntitySensorWorld):
    def __init__(self):
        super().__init__()
        # Positive variance intercept, variance/day and missing-evidence multiplier.
        self.state_noise=nn.Parameter(torch.tensor([[-3.]*6,[-.5]*6,[0.]*6]))
        self.state_response=nn.Parameter(torch.zeros(6))

    def forward(self,values,observed,ages,clock,owners,hours,*,state=None,return_trace=False):
        base,final,tr=super().forward(values,observed,ages,clock,owners,hours,state=state,return_trace=True)
        # Age is external causal bookkeeping supplied identically at every history
        # length. It is capped256h by the source contract, not guessed from values.
        age_hours=torch.expm1(ages[:,-1,:6]*math.log(257.))
        evidence_mass=final.source_mass[:,:6].clamp(0,1)
        coefficients=F.softplus(self.state_noise)
        state_variance=(coefficients[0]+coefficients[1]*age_hours/24)*(1+coefficients[2]*(1-evidence_mass))
        state_variance=torch.where(observed[:,-1,:6],torch.zeros_like(state_variance),state_variance)
        response=1+.25*self.state_response.tanh()
        propagated=response.square()*state_variance
        future_variance=base.log_scale.square()
        total_variance=future_variance+propagated
        pred=SensorForecast(base.log_location,total_variance.sqrt())
        if return_trace:
            return pred,final,dict(tr,latent_state_variance=state_variance,
              affine_response=response,future_variance=future_variance,
              propagated_state_variance=propagated,total_variance=total_variance)
        return pred,final

    def training_loss(self,pred,raw_targets,mask):
        """Proper likelihood/interval objectives plus a fixed log-point loss."""
        target=(torch.log1p(raw_targets.clamp_min(0))-self.target_mean)/self.target_scale
        threshold=-self.target_mean/self.target_scale
        radius=1.6448536269514722*pred.log_scale
        lower=torch.maximum(pred.log_location-radius,threshold)
        upper=torch.maximum(pred.log_location+radius,threshold)
        score=upper-lower+20*(lower-target).clamp_min(0)+20*(target-upper).clamp_min(0)
        mse=(pred.log_location-target).square()
        def avg(x):return torch.where(mask,x,torch.zeros_like(x)).sum()/mask.sum().clamp_min(1)
        nll=self.nll(pred,raw_targets,mask)
        return nll+.1*avg(mse)+.1*avg(score),dict(nll=nll,mse=avg(mse),interval=avg(score))
