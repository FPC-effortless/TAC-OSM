"""Learned residual thermal forecasts; no fault labels or maintenance rules."""
import torch
from torch import nn


class ThermalForecast(nn.Module):
    def __init__(self, kind='linear', revin=False, rooms=10, context=8,
                 exogenous=2, horizons=2, width=64, seed=0):
        super().__init__(); torch.manual_seed(seed)
        self.kind=kind; self.revin=revin; self.rooms=rooms; self.context=context
        self.exogenous=exogenous; self.horizons=horizons
        self.register_buffer('location', torch.zeros(rooms+exogenous))
        self.register_buffer('scale', torch.ones(rooms+exogenous))
        inputs=context*(rooms+exogenous+2)
        if kind == 'linear': self.network=nn.Linear(inputs, horizons*rooms)
        elif kind == 'mlp': self.network=nn.Sequential(nn.Linear(inputs,width),nn.GELU(),nn.Linear(width,horizons*rooms))
        else: raise ValueError(kind)
        head=self.network if kind=='linear' else self.network[-1]
        nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)

    def features(self, values, minutes):
        if values.ndim!=3 or values.shape[1:]!=(self.context,self.rooms+self.exogenous):
            raise ValueError('Expected batch x context x public room/exogenous values')
        if minutes.shape != values.shape[:2]: raise ValueError('Past minute-of-day per observation required')
        normalized=(values-self.location)/self.scale
        decoder_scale=self.scale[:self.rooms][None].expand(len(values),-1)
        if self.revin:
            rooms=values[:,:,:self.rooms]
            center=rooms.mean(1,keepdim=True)
            scale=rooms.std(1,keepdim=True,unbiased=False).clamp_min(1e-6)
            normalized=torch.cat(((rooms-center)/scale, normalized[:,:,self.rooms:]),-1)
            decoder_scale=scale[:,0]
        phase=minutes*(2*torch.pi/1440)
        features=torch.cat((normalized,phase.sin()[:,:,None],phase.cos()[:,:,None]),-1)
        return features.flatten(1),decoder_scale

    def change(self,values,minutes):
        features,decoder_scale=self.features(values,minutes)
        delta=self.network(features).reshape(-1,self.horizons,self.rooms)
        return delta*decoder_scale[:,None]

    def forward(self,values,minutes):
        return values[:,-1,None,:self.rooms]+self.change(values,minutes)


def specifications():
    return {f'{kind}_{normalization}':dict(kind=kind,revin=normalization=='revin')
            for kind in ('linear','mlp') for normalization in ('static','revin')}
