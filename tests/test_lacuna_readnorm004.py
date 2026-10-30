import importlib.util
import pytest
import torch
from scripts.lacuna_readnorm004 import REG,dual_norm,controlled_reads


def test_random_two_norms_projection_and_independent_reference():
    spec=importlib.util.spec_from_file_location('independent004',REG/'pnds_reference.py')
    ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    gen=torch.Generator().manual_seed(4)
    for scale in (1e-5,.1,10.):
        for _ in range(30):
            v,h,c=[scale*torch.randn(8,generator=gen,dtype=torch.float64) for _ in range(3)]
            out,status=dual_norm(v,h,c);expected,es=ref.dual_norm(v.tolist(),h.tolist(),c.tolist())
            assert status==es=='REGULAR'
            assert torch.allclose(out,torch.tensor(expected,dtype=torch.float64),atol=1e-12,rtol=1e-12)
            assert torch.allclose(out.norm(),h.norm(),atol=1e-12,rtol=1e-12)
            assert torch.allclose((out+c).norm(),(h+c).norm(),atol=1e-12,rtol=1e-12)
            assert torch.allclose(out@c,h@c,atol=1e-12,rtol=1e-12)
            same,_=dual_norm(h,h,c)
            assert torch.allclose(same,h,atol=1e-12,rtol=1e-12)


@pytest.mark.parametrize('kind',('zero_current','zero_candidate','tiny_current','parallel_candidate','no_freedom'))
def test_degenerate_paths_preserve_norms_and_flag_no_direction(kind):
    x=torch.tensor([1.,0,0,0,0,0,0,0],dtype=torch.float64)
    y=x.roll(1);h=x+y;v=x+2*y;c=x.clone()
    expected='REGULAR_ZERO_CURRENT'
    if kind=='zero_current':c=torch.zeros_like(c)
    if kind=='zero_candidate':c=torch.zeros_like(c);v=torch.zeros_like(v);expected='INCONCLUSIVE_ZERO_CANDIDATE'
    if kind=='tiny_current':c*=1e-13;expected='INCONCLUSIVE_NEAR_ZERO_CURRENT'
    if kind=='parallel_candidate':v=x.clone();expected='INCONCLUSIVE_CANDIDATE_PARALLEL'
    if kind=='no_freedom':h=x.clone();expected='NO_ORTHOGONAL_FREEDOM'
    out,status=dual_norm(v,h,c)
    assert status==expected
    assert torch.allclose(out.norm(),h.norm(),atol=1e-12)
    assert torch.allclose((out+c).norm(),(h+c).norm(),atol=1e-12)
    if expected!='REGULAR_ZERO_CURRENT':assert torch.equal(out,h)


@pytest.mark.parametrize('bad',(torch.ones(7),torch.ones(1,8),torch.ones(8,dtype=torch.long),torch.tensor([float('nan')]*8),torch.tensor([float('inf')]*8)))
def test_invalid_input_rejected_before_prediction(bad):
    good=torch.ones(8,dtype=torch.float64)
    with pytest.raises(ValueError):dual_norm(bad,good,good)


def test_regular_projection_gradient_matches_finite_difference():
    gen=torch.Generator().manual_seed(45)
    v=torch.randn(8,generator=gen,dtype=torch.float64,requires_grad=True)
    h,c,w=[torch.randn(8,generator=gen,dtype=torch.float64) for _ in range(3)]
    out,_=dual_norm(v,h,c);grad=torch.autograd.grad(out@w,v)[0]
    direction=torch.randn(8,generator=gen,dtype=torch.float64);step=1e-6
    plus,_=dual_norm(v.detach()+step*direction,h,c);minus,_=dual_norm(v.detach()-step*direction,h,c)
    assert abs(float(grad@direction)-float((plus-minus)@w/(2*step)))<1e-8


def test_two_slow_norms_do_not_fix_the_fast_slow_mixture_norm():
    h=torch.tensor([1.,1.,0,0,0,0,0,0],dtype=torch.float64)
    c=torch.tensor([1.,0,0,0,0,0,0,0],dtype=torch.float64)
    v=torch.tensor([0.,0,1.,0,0,0,0,0],dtype=torch.float64)
    rf=torch.tensor([0.,1.,0,0,0,0,0,0],dtype=torch.float64)
    out,status=dual_norm(v,h,c)
    assert status=='REGULAR'
    assert torch.equal((out+c).norm(),(h+c).norm())
    assert not torch.isclose((rf+out+c).norm(),(rf+h+c).norm())
