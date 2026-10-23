"""Numerical fixtures only; these tests do not establish learned capability."""
import json
from pathlib import Path
import torch
from tac_osm.variants.selective_physical import SelectivePhysicalModel, mixture_cdf, mixture_quantile, mixture_nll, MotionForecast
from scripts.me077_selective_physical import rollout, integrity


def test_whole_model_budget_and_source_freeze():
    model=SelectivePhysicalModel()
    assert sum(p.numel() for p in model.parameters())==102503
    assert all(p.requires_grad for p in model.parameters())
    assert len(integrity()['models'])==5


def test_quantiles_are_coherent_with_cdf_and_ordered():
    torch.manual_seed(33)
    model=SelectivePhysicalModel()
    x=torch.randn(3,14)
    pred=model(x,torch.zeros_like(x),torch.randn_like(x)*.02,torch.ones(3))
    q=[mixture_quantile(pred,p) for p in (.05,.5,.95)]
    assert (q[0]<=q[1]).all() and (q[1]<=q[2]).all()
    for v,p in zip(q,(.05,.5,.95)):
        torch.testing.assert_close(mixture_cdf(v-x,pred),torch.full_like(x,p),atol=1e-4,rtol=0)
    assert pred.fast.abs().max()<=1


def test_stationary_component_and_moving_component_distribution_limits():
    x=torch.zeros(2,14);one=torch.ones_like(x)
    pred=MotionForecast(x,one,.01*one,.1*one,-30*one,torch.zeros(2),torch.zeros(2,96))
    torch.testing.assert_close(mixture_quantile(pred,.5),x,atol=1e-6,rtol=0)
    pred=pred._replace(mixture_logit=30*one)
    torch.testing.assert_close(mixture_quantile(pred,.5),one,atol=1e-6,rtol=0)


def test_finite_end_to_end_gradients_and_causal_history():
    torch.manual_seed(4);model=SelectivePhysicalModel()
    x=torch.randn(4,8,14);a=torch.randn_like(x)
    pred=rollout(model,x,a)
    loss=mixture_nll(pred,x[:,-1]+.01)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.encoder[0].weight.grad.norm()>0 or model.heads.weight.grad.norm()>0
    # Model receives no future state or future human authority feature.
    assert pred.current.data_ptr()==x[:,-1].data_ptr()
