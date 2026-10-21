"""ME043B: full-capacity context-first TAC with scheduled native learning.

Post-ME043A historical-development engineering; no independent qualification.
"""
import torch
from torch import nn
from tac_osm.me038_innovation_forecast import InnovationForecaster
from tac_osm.me043_adaptive_compute import AdaptiveForecaster,objective as dense_objective

ARMS=('wide_context','wide_native','scheduled_free','scheduled_001')


class ScheduledForecaster(AdaptiveForecaster):
    def __init__(self,horizon,cost):
        super().__init__(horizon,cost)
        self.encoder=nn.Sequential(nn.Linear(192,50),nn.SiLU(),nn.Linear(50,128))

    def cheap(self,x):
        base,_,ctx,latent,q,mean,scale=self.features(x)
        delta=self.correction(ctx,latent)
        return self.unpack(base+delta,mean,scale),delta


def build_model(arm,horizon,seed):
    torch.manual_seed(seed)
    if arm=='wide_context':
        net=InnovationForecaster('context_cross',horizon)
        net.encoder=nn.Sequential(nn.Linear(192,80),nn.SiLU(),nn.Linear(80,128))
    elif arm=='wide_native':
        net=InnovationForecaster('innovation_tac',horizon)
        net.context_encoder=nn.Sequential(nn.Linear(192,63),nn.SiLU(),nn.Linear(63,64))
    elif arm in ('scheduled_free','scheduled_001'):
        net=ScheduledForecaster(horizon,0. if arm=='scheduled_free' else .001)
        # Retain every original context-only coefficient, rather than randomly
        # reinitializing the strong cheap path to accommodate the new expert.
        torch.manual_seed(seed)
        old=InnovationForecaster('context_cross',horizon)
        state=net.state_dict()
        for name,value in old.state_dict().items():state[name]=value
        net.load_state_dict(state,strict=True)
    else:raise ValueError('unregistered scheduled model')
    return net


def objective(net,x,y,step):
    if isinstance(net,ScheduledForecaster) and step%8!=0:
        pred,delta=net.cheap(x)
        mse=(pred-y).square().mean()
        return mse+10*net.anchor_loss()+.01*delta.square().mean(),mse
    return dense_objective(net,x,y)
