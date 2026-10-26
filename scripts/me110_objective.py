"""Availability-balanced point learning and uncertainty-only auxiliary losses."""
import torch
from scripts.me106_gradient_roles import objective_parts


def balanced(values,valid,observed):
    sums=[]
    for c in range(6):
        for present in (False,True):
            mask=valid[:,c] & (observed[:,c]==present)
            if bool(mask.any()): sums.append(values[:,c][mask].mean())
    if not sums: raise ValueError('no genuine future labels')
    return torch.stack(sums).mean()


def losses(model,forecast,trace,batch,rawscale):
    _,parts=objective_parts(model,forecast,trace,batch,rawscale,{'point':.5,'interval':.25})
    observed=batch[1][:,-1,:6];valid=batch[7];truth=batch[6]
    median=model.raw_quantiles(forecast)[...,1]
    mae=balanced((median-truth).abs()/rawscale,valid,observed)
    normalized=(torch.log1p(truth)-model.target_mean)/model.target_scale
    mse=balanced((forecast.log_location-normalized).square(),valid,observed)
    point=mae+.1*mse
    auxiliary=parts['nll']+.25*parts['interval']+.3*parts['reconstruction_nll']
    return point,auxiliary,dict(parts,balanced_point=mae,balanced_location_mse=mse,
        point_objective=point,uncertainty_objective=auxiliary)
