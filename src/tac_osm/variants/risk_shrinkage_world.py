"""ME079 integrated risk-conditioned smooth physical residual shrinkage.

An established adaptive-shrinkage principle is applied within each complete
native/GRU/Transformer predictor. No oracle switching or test calibration.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from tac_osm.variants.kinematic_world import KinematicWorld

class ShrinkageWorld(KinematicWorld):
    def __init__(self,architecture='native',width=None,identified=None):
        super().__init__(architecture,width,identified)
        self.shrinkage_scale=nn.Parameter(torch.full((3,14),math.log(math.expm1(.25))))
    def forward(self,x,actions):
        forecast=super().forward(x,actions)
        raw_delta=forecast['median']-x[:,:,None]
        probability=forecast['motion_logit'].sigmoid()[:,:,:,None]
        threshold=(self.delta_scale*self.horizons[:,None]*F.softplus(self.shrinkage_scale))[None,None]
        threshold=threshold*(1-probability)/(probability+.05)
        gain=raw_delta.square()/(raw_delta.square()+threshold.square()+1e-12)
        displacement=raw_delta*gain
        shift=displacement-raw_delta
        return {**forecast,'median':forecast['median']+shift,
                'lower':forecast['lower']+shift,'upper':forecast['upper']+shift}
