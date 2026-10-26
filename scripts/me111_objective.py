"""Consistently availability-balanced censored forecast and reconstruction losses."""
import math
import torch
from scripts.me110_objective import balanced


def nll_elements(model,forecast,truth):
    mu=forecast.log_location
    z=(torch.log1p(truth)-model.target_mean)/model.target_scale
    side=torch.where(z<mu,forecast.left_scale,forecast.right_scale)
    positive=.5*((z-mu)/side).square()+side.log()+.5*math.log(2*math.pi)+model.target_scale.log()+torch.log1p(truth)
    threshold=-model.target_mean/model.target_scale
    ts=torch.where(threshold<mu,forecast.left_scale,forecast.right_scale)
    return torch.where(truth==0,-torch.special.log_ndtr((threshold-mu)/ts),positive)


def losses(model,forecast,trace,batch,rawscale):
    observed=batch[1][:,-1,:6];valid=batch[7];truth=batch[6]
    median=model.raw_quantiles(forecast)[...,1]
    mae=balanced((median-truth).abs()/rawscale,valid,observed)
    norm=(torch.log1p(truth)-model.target_mean)/model.target_scale
    mse=balanced((forecast.log_location-norm).square(),valid,observed)
    point=mae+.1*mse
    uncertainty=model.uncertainty_forecast(forecast,trace)
    nll=balanced(nll_elements(model,uncertainty,truth),valid,observed)
    lower,_,upper=model.raw_quantiles(uncertainty).unbind(-1)
    score=(upper-lower+20*(lower-truth).clamp_min(0)+20*(truth-upper).clamp_min(0))/rawscale
    interval=balanced(score,valid,observed)
    mean,variance=trace['current_mean'],trace['current_variance']
    e=.5*((batch[8]-mean).square()/variance+variance.log()+math.log(2*math.pi))
    # Reconstruct only actually recorded, deliberately hidden current values;
    # balance present channels, do not fabricate natural-gap truth.
    cells=[e[:,c][batch[9][:,c]].mean() for c in range(6) if bool(batch[9][:,c].any())]
    reconstruction=torch.stack(cells).mean() if cells else (mean.sum()+variance.sum())*0
    auxiliary=nll+.25*interval+.3*reconstruction
    return point,auxiliary,dict(nll=nll,point=mae,location_mse=mse,interval=interval,
        reconstruction_nll=reconstruction,point_objective=point,uncertainty_objective=auxiliary)
