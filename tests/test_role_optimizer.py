"""Synthetic arithmetic optimizer tests only; no real TAC training."""
import pytest
import torch
from tac_osm.role_optimizer import BlockProtectedAdamW
from scripts.me108_optimizer_reference import step as reference_step


def test_zero_point_role_recovers_unopposed_torch_adamw_moments_and_weights():
    a=torch.nn.Parameter(torch.tensor([.5,-.7],dtype=torch.float64))
    b=torch.nn.Parameter(a.detach().clone())
    guard=BlockProtectedAdamW([a],blocks=[[0]],lr=.01)
    ref=torch.optim.AdamW([b],lr=.01,weight_decay=.001)
    for k in range(4):
        g=torch.tensor([.2+k*.03,-.1],dtype=torch.float64)
        a.grad=g.clone();b.grad=g.clone()
        guard.set_point_gradients([torch.zeros_like(a)])
        guard.step();ref.step()
        torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-14)
        for key in ('exp_avg','exp_avg_sq'):
            torch.testing.assert_close(guard.state[a][key],ref.state[b][key],rtol=1e-12,atol=1e-14)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_actual_weight_changes_respect_each_block_after_rounding_and_keep_free_coordinates(dtype):
    a=torch.nn.Parameter(torch.tensor([1.,1.,1.],dtype=dtype))
    b=torch.nn.Parameter(torch.tensor([.2,-.4],dtype=dtype))
    opt=BlockProtectedAdamW([{'params':[a],'lr':3e-6},{'params':[b],'lr':3e-4}],blocks=[[0],[1]])
    for _ in range(12):
        old=[a.detach().clone(),b.detach().clone()]
        point=[torch.tensor([1.,1.,0.],dtype=dtype),torch.tensor([1.,-1.],dtype=dtype)]
        a.grad=torch.tensor([-2.,1.,3.],dtype=dtype);b.grad=torch.tensor([-1.,2.],dtype=dtype)
        opt.set_point_gradients(point);opt.step()
        for p,g,w in zip((a,b),point,old):
            assert float((g.double()*(p.detach().double()-w.double())).sum())<=0
        assert all(row['applied_dot']<=0 for row in opt.last_guard_receipt)
    with pytest.raises(ValueError):opt.step()


def test_invalid_input_cannot_partially_change_weights_or_moments():
    a=torch.nn.Parameter(torch.ones(2));b=torch.nn.Parameter(torch.ones(2))
    opt=BlockProtectedAdamW([a,b],blocks=[[0],[1]])
    opt.set_point_gradients([torch.ones_like(a),torch.ones_like(b)])
    a.grad=torch.ones_like(a);b.grad=torch.full_like(b,float('nan'))
    with pytest.raises(ValueError):opt.step()
    assert torch.equal(a,torch.ones_like(a)) and torch.equal(b,torch.ones_like(b))
    assert not opt.state
    with pytest.raises(ValueError):
        BlockProtectedAdamW([{'params':[a],'lr':float('nan')}],blocks=[[0]])


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_independent_equations_match_actual_moments_and_applied_parameters(dtype):
    gen=torch.Generator().manual_seed(108)
    ps=[torch.nn.Parameter(torch.randn(17,generator=gen,dtype=dtype)),
        torch.nn.Parameter(torch.randn(11,generator=gen,dtype=dtype))]
    opt=BlockProtectedAdamW([{'params':[ps[0]],'lr':3e-5},{'params':[ps[1]],'lr':3e-4}],blocks=[[0],[1]])
    for _ in range(8):
        gs=[torch.randn(p.shape,generator=gen,dtype=dtype) for p in ps]
        point=[torch.randn(p.shape,generator=gen,dtype=dtype) for p in ps]
        old=[p.detach().clone() for p in ps]
        states=[{k:v.clone() for k,v in opt.state.get(p,{}).items()} for p in ps]
        configs=[{k:g[k] for k in ('lr','betas','eps','weight_decay')} for g in opt.param_groups]
        expected,expected_state=reference_step(old,gs,point,states,configs,[[0],[1]])
        for p,g in zip(ps,gs):p.grad=g
        opt.set_point_gradients(point);opt.step()
        for p,w,s in zip(ps,expected,expected_state):
            torch.testing.assert_close(p,w,rtol=0,atol=0)
            for key in ('step','exp_avg','exp_avg_sq'):
                torch.testing.assert_close(opt.state[p][key],s[key],rtol=0,atol=0)
