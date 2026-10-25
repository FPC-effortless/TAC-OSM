"""Optional planar CBF QP operator only: no action policy, training or safety data."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

import pytest
import torch
from tac_osm.variants.planar_constraint_adapter import project_planar_cbf


def one(position,obstacle,desired,w,**kwargs):
    to_t=lambda v:torch.tensor([v],dtype=torch.float64)
    return project_planar_cbf(to_t(position),to_t(obstacle),to_t(desired),to_t(w),
        radius=1.,gamma=1.,dt=.1,**kwargs)


def test_obstacle_approaching_zero_is_unsafe():
    u,margin=one((1.,0.),(0.,0.),(0.,0.),(1.,0.))
    assert u[0,0] >= 1. - 1e-12
    assert margin[0].item() >= -1e-12


def test_irrelevant_safe_command_stays_unchanged():
    u,margin=one((3.,0.),(0.,0.),(.2,.3),(0.,0.))
    torch.testing.assert_close(u,torch.tensor([[.2,.3]],dtype=torch.float64))
    assert margin.item() >= 0.


def test_uncertainty_margin_increases_outward_control():
    a,_=one((1.01,0.),(0.,0.),(0.,0.),(.5,0.))
    b,_=one((1.01,0.),(0.,0.),(0.,0.),(.5,0.),
            obstacle_velocity_error_bound=.25)
    assert b[0,0] >= a[0,0]+.25-1e-10


def test_initial_unsafe_returns_no_action_not_zero():
    with pytest.raises(ValueError,match="unsafe"):
        one((0.,0.),(0.,0.),(0.,0.),(0.,0.))


def test_invalid_dt_rejected():
    x=torch.tensor([[2.,0.]],dtype=torch.float64)
    with pytest.raises(ValueError,match="contract"):
        project_planar_cbf(x,torch.zeros_like(x),x,torch.zeros_like(x),
                           radius=1.,gamma=4.,dt=.3)


def test_diff_qp_active_set_gradient_and_gradcheck():
    p=torch.tensor([[1.1,.2]],dtype=torch.float64)
    o=torch.zeros_like(p)
    w=torch.tensor([[.8,0.]],dtype=torch.float64)
    d=torch.tensor([[-1.,0.]],dtype=torch.float64,requires_grad=True)
    def layer(x):
        return project_planar_cbf(p,o,x,w,radius=1.,gamma=1.,dt=.1)[0]
    assert torch.autograd.gradcheck(layer,(d,),eps=1e-6,atol=1e-6,rtol=1e-4)
    jac=torch.autograd.functional.jacobian(layer,d)[0,:,0,:]
    n=2*(p-o)[0]
    formula=torch.eye(2,dtype=torch.float64)-torch.outer(n,n)/(n@n)
    torch.testing.assert_close(jac,formula,atol=1e-12,rtol=1e-12)


def test_diff_qp_inactive_gradient_is_identity():
    p=torch.tensor([[3.,0.]],dtype=torch.float64)
    o=torch.zeros_like(p)
    w=o.clone()
    d=torch.tensor([[0.,.25]],dtype=torch.float64,requires_grad=True)
    def layer(x):
        return project_planar_cbf(p,o,x,w,radius=1.,gamma=1.,dt=.1)[0]
    assert torch.autograd.gradcheck(layer,(d,),eps=1e-6,atol=1e-6,rtol=1e-4)
    jac=torch.autograd.functional.jacobian(layer,d)[0,:,0,:]
    torch.testing.assert_close(jac,torch.eye(2,dtype=torch.float64),atol=1e-12,rtol=1e-12)


def test_randomized_fp64_cbf_margin_only_diagnostic():
    generator=torch.Generator().manual_seed(20261024)
    p=torch.randn(64,2,generator=generator,dtype=torch.float64)*.2+torch.tensor([2.0,0.])
    o=torch.zeros_like(p)
    d=torch.randn(64,2,generator=generator,dtype=torch.float64)
    w=torch.randn(64,2,generator=generator,dtype=torch.float64)*.2
    u,margin=project_planar_cbf(p,o,d,w,radius=1.,gamma=.7,dt=.1)
    assert torch.all(margin>=-1e-11).item()
    assert torch.isfinite(u).all().item()
