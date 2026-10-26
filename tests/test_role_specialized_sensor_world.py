from unittest.mock import patch

import pytest
import torch

from tac_osm.variants.role_specialized_sensor_world import RoleSpecializedSensorWorld, RoleFFN
from tac_osm.role_update_contract import protect_displacement
from scripts.me107_graph_reference import serial
from tests.test_protected_sensor_world import sample


def test_live_budget_and_hard_assigned_outputs():
    model = RoleSpecializedSensorWorld()
    assert sum(p.numel() for p in model.parameters()) == 102713
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 102713
    assert len(list(model.parameters())) == 39
    x = torch.randn(2,102)
    assert [h(x).shape[-1] for h in (model.current_head,model.width_head,model.median_head)] == [12,12,6]
    with pytest.raises(ValueError):
        RoleFFN('invented_role')


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_full_serial_graph_and_all_parameter_gradients(dtype):
    torch.manual_seed(107)
    model = RoleSpecializedSensorWorld(torch.full((6,),.1)).to(dtype)
    model.target_mean.fill_(3.)
    x = sample(dtype)
    f, state, trace = model(*x, return_trace=True)
    # Independent reference may use standard native layer weights, but cannot
    # call any new role FFN's implementation or the model's forward.
    with patch.object(model.current_head,'forward',side_effect=AssertionError('not independent')), \
         patch.object(model.width_head,'forward',side_effect=AssertionError('not independent')), \
         patch.object(model.median_head,'forward',side_effect=AssertionError('not independent')):
        mu,left,right,ss,sm,es,memory,rt = serial(model,*x[:5])
    rtol,atol=(2e-5,2e-6) if dtype==torch.float32 else (1e-9,1e-10)
    for a,b in zip((f.log_location,f.left_scale,f.right_scale,state.source_sum,state.source_mass,state.evidence_sum,state.memory),
                   (mu,left,right,ss,sm,es,memory)):
        torch.testing.assert_close(a,b,rtol=rtol,atol=atol)
    actual = f.log_location.square().sum()+f.left_scale.square().sum()+f.right_scale.square().sum()
    expected = mu.square().sum()+left.square().sum()+right.square().sum()
    pa=tuple(model.parameters())
    ga=torch.autograd.grad(actual,pa,retain_graph=True,allow_unused=True)
    gb=torch.autograd.grad(expected,pa,allow_unused=True)
    for a,b,p in zip(ga,gb,pa):
        a=torch.zeros_like(p) if a is None else a
        b=torch.zeros_like(p) if b is None else b
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a,b,rtol=rtol,atol=atol)
    assert (f.left_scale>=trace['protected_floor_scale']).all()
    assert (f.right_scale>=trace['protected_floor_scale']).all()
    assert (trace['median_correction'].abs()<=.5).all()
    assert (trace['width_factors']>=.25).all() and (trace['width_factors']<=4).all()


def test_no_new_recurrent_memory_and_streaming_equivalence():
    model=RoleSpecializedSensorWorld().double()
    x=sample(torch.float64)
    full,fs=model(*x)
    first=[t[:,:8] if t.ndim>=2 else t for t in x]
    rest=[t[:,8:] if t.ndim>=2 else t for t in x]
    _,s=model(*first)
    streamed,ss=model(*rest,state=s)
    for a,b in zip(full,streamed):
        torch.testing.assert_close(a,b,rtol=1e-9,atol=1e-10)
    for a,b in zip((fs.source_sum,fs.source_mass,fs.evidence_sum,fs.memory),
                   (ss.source_sum,ss.source_mass,ss.evidence_sum,ss.memory)):
        torch.testing.assert_close(a,b,rtol=1e-9,atol=1e-10)
    assert sum(t.numel() for t in (ss.source_sum,ss.source_mass,ss.evidence_sum,ss.memory)) == 2*152


def test_width_and_median_parameter_roles_have_structural_gradient_zeros():
    model=RoleSpecializedSensorWorld().double();model.target_mean.fill_(3.)
    f,_,tr=model(*sample(torch.float64),return_trace=True)
    point=model.raw_quantiles(f)[...,1].sum()
    width=tuple(model.width_head.parameters())
    pg=torch.autograd.grad(point,width,allow_unused=True,retain_graph=True)
    assert all(g is None or torch.count_nonzero(g)==0 for g in pg)
    nll=model.nll(model.uncertainty_forecast(f,tr),torch.ones(2,6,dtype=torch.float64),torch.ones(2,6,dtype=torch.bool))
    ng=torch.autograd.grad(nll,tuple(model.median_head.parameters()),allow_unused=True)
    assert all(g is None or torch.count_nonzero(g)==0 for g in ng)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_blockwise_actual_displacement_contract(dtype):
    gen=torch.Generator().manual_seed(107)
    for _ in range(20):
        g=[torch.randn(31,generator=gen,dtype=dtype),torch.randn(19,generator=gen,dtype=dtype),torch.zeros(7,dtype=dtype)]
        d=[torch.randn(x.shape,generator=gen,dtype=dtype) for x in g]
        old=[x.clone() for x in d]
        safe,receipts=protect_displacement(g,d,[[0],[1],[2]])
        assert all(r['after']<=0 for r in receipts)
        assert all(torch.equal(a,b) for a,b in zip(old,d))
        assert torch.equal(safe[2],d[2])


def test_update_guard_fails_closed_on_invalid_partitions_or_nonfinite_data():
    with pytest.raises(ValueError):protect_displacement([torch.ones(2)],[torch.ones(2)],[[0],[0]])
    with pytest.raises(ValueError):protect_displacement([torch.ones(2)],[torch.tensor([float('nan'),1.])],[[0]])
