import sys
from pathlib import Path
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from tac_osm.variants.point_priority_sensor_world import PointPrioritySensorWorld
from scripts.me106_gradient_roles import objective_parts,assign_priority_gradients,project_gradient_lists
from scripts.me106_graph_reference import priority_gradients
from tests.test_protected_sensor_world import sample

torch.set_num_threads(1)


def batch(dtype):
    x=sample(dtype)
    target=torch.arange(12,dtype=dtype).reshape(2,6)/2+1
    normalized=torch.randn(2,6,dtype=dtype)
    return [*x,target,torch.ones(2,6,dtype=torch.bool),normalized,torch.ones(2,6,dtype=torch.bool)]


def test_role_view_preserves_predictions_and_removes_only_direct_median_gradient():
    model=PointPrioritySensorWorld().double();model.target_mean.fill_(3.)
    x=batch(torch.float64);f,_,trace=model(*x[:6],return_trace=True)
    view=model.uncertainty_forecast(f,trace)
    assert sum(p.numel()for p in model.parameters())==100349
    assert len(list(model.parameters()))==33
    assert torch.equal(model.raw_quantiles(f),model.raw_quantiles(view))
    nll=model.nll(view,x[6],x[7]);params=(model.median_head.weight,model.width_head.weight,model.encoder.weight)
    grads=torch.autograd.grad(nll,params,retain_graph=True)
    assert grads[0].abs().sum()==0
    assert grads[1].abs().sum()>0 and grads[2].abs().sum()>0
    point=model.raw_quantiles(f)[...,1].sum()
    pg=torch.autograd.grad(point,params,allow_unused=True)
    assert pg[0].abs().sum()>0
    assert pg[1]is None or pg[1].abs().sum()==0


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_independent_full_source_loss_and_projected_gradient_reference(dtype):
    model=PointPrioritySensorWorld(floor_rate=torch.full((6,),.1)).to(dtype);model.target_mean.fill_(3.)
    with torch.no_grad():
        model.median_head.weight.normal_(0,.01);model.width_head.weight.normal_(0,.01)
    x=batch(dtype);scale=torch.arange(1,7,dtype=dtype);coeff={'point':1.,'interval':.25}
    f,_,tr=model(*x[:6],return_trace=True)
    _,parts=objective_parts(model,f,tr,x,scale,coeff)
    receipt=assign_priority_gradients(model,parts,coeff['point'],coeff['interval'])
    actual=[p.grad.clone()for p in model.parameters()]
    expected,refparts=priority_gradients(model,x,scale,coeff)
    rtol,atol=(2e-5,2e-6)if dtype==torch.float32 else(1e-9,1e-10)
    for name in parts:torch.testing.assert_close(parts[name],refparts[name],rtol=rtol,atol=atol)
    for a,b in zip(actual,expected):
        assert torch.isfinite(a).all();torch.testing.assert_close(a,b,rtol=rtol,atol=atol)
    bound=100*torch.finfo(dtype).eps*(abs(float(receipt['auxiliary_dot_before']))+float(receipt['point_gradient_norm_squared'])+1)
    assert float(receipt['auxiliary_dot_after'])>=-bound


def test_projection_conflict_alignment_and_zero_point_boundary():
    for p,u in [(torch.tensor([1.,0.]),torch.tensor([-2.,3.])),
                (torch.tensor([1.,0.]),torch.tensor([2.,3.])),
                (torch.zeros(2),torch.tensor([2.,3.]))]:
        total,info=project_gradient_lists([p],[u]);projected=total[0]-p
        assert p.dot(projected)>=0
        assert torch.isfinite(total[0]).all()
        if p[1]==0:assert projected[1]==u[1]
    # An FP64 coefficient is applied before rounding; no overflowing FP32 scalar.
    p=torch.tensor([1e-30,0.]);u=torch.tensor([-1e20,1.]);out,_=project_gradient_lists([p],[u]);assert torch.isfinite(out[0]).all()
