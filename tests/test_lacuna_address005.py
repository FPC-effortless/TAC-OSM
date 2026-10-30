import importlib.util
import pytest
import torch
from scripts.lacuna_address005 import REG,project,random_vector,choice,group_means,read_control

def test_projection_matches_independent_reference_and_all_three_norms():
    s=importlib.util.spec_from_file_location('reference005',REG/'pnds_reference.py');ref=importlib.util.module_from_spec(s);s.loader.exec_module(ref)
    gen=torch.Generator().manual_seed(5005)
    for _ in range(100):
        v,h,c,f=[torch.randn(8,dtype=torch.float64,generator=gen) for _ in range(4)]
        out,status,rank=project(v,h,c,f);expected,es,er=ref.project(v.tolist(),h.tolist(),c.tolist(),f.tolist())
        assert (status,rank)==(es,er)==('REGULAR',2)
        assert torch.allclose(out,torch.tensor(expected,dtype=torch.float64),atol=1e-12,rtol=1e-12)
        for x,y in [(out,h),(out+c,h+c),(.7*f+.3*(out+c),.7*f+.3*(h+c))]:assert torch.allclose(x.norm(),y.norm(),atol=1e-12,rtol=1e-12)
        identity,_,_=project(h,h,c,f);assert torch.allclose(identity,h,atol=1e-12,rtol=1e-12)

@pytest.mark.parametrize('kind',('rank0','rank1','near_axis','zero_radius','parallel_candidate'))
def test_degenerate_rank_and_fallback_are_explicit(kind):
    x=torch.eye(8,dtype=torch.float64);h=x[0]+x[2];v=x[3];c=x[0];f=x[1]
    if kind=='rank0':c=torch.zeros(8);f=torch.zeros(8)
    if kind=='rank1':f=c
    if kind=='near_axis':c=c*1e-13
    if kind=='zero_radius':h=x[0]+x[1]
    if kind=='parallel_candidate':v=x[0]
    out,status,rank=project(v,h,c,f)
    assert torch.allclose(out.norm(),h.norm(),atol=1e-12)
    assert torch.allclose((out+c).norm(),(h+c).norm(),atol=1e-12)
    if kind in ('near_axis','zero_radius','parallel_candidate'):assert status!='REGULAR' and torch.equal(out,h)
    else:assert status=='REGULAR' and rank==(0 if kind=='rank0' else 1)

@pytest.mark.parametrize('bad',(torch.ones(7),torch.ones(8,dtype=torch.long),torch.full((8,),float('nan')),torch.full((8,),float('inf'))))
def test_invalid_input_rejected(bad):
    with pytest.raises(ValueError):project(bad,torch.ones(8),torch.ones(8),torch.ones(8))

def test_projection_gradient_and_native_mixed_norm():
    gen=torch.Generator().manual_seed(51);v=torch.randn(8,dtype=torch.float64,generator=gen,requires_grad=True)
    h,c,f,w=[torch.randn(8,dtype=torch.float64,generator=gen) for _ in range(4)]
    out,_,_=project(v,h,c,f);grad=torch.autograd.grad(out@w,v)[0];direction=torch.ones(8,dtype=torch.float64);step=1e-6
    plus,_,_=project(v.detach()+step*direction,h,c,f);minus,_,_=project(v.detach()-step*direction,h,c,f)
    assert abs(float(grad@direction)-float((plus-minus)@w/(2*step)))<1e-8
    trace={'mix':torch.tensor([[.7,.3]]),'read_fast':f.float()[None],'read_slow':(h+c).float()[None]}
    total,_,info=read_control(v.detach(),h,c,trace)
    assert info['errors']['mixed_native']<3e-6 and total.dtype==torch.float32

def test_random_controls_are_prefix_deterministic_and_group_means_do_not_count_seeds_as_sources():
    assert torch.equal(random_vector('owner',256,0),random_vector('owner',256,0))
    assert not torch.equal(random_vector('owner',256,0),random_vector('owner',256,1))
    assert choice('owner',256,'irrelevant',[4,6,9]) in (4,6,9)
    assert choice('owner',256,'irrelevant',[]) is None
    rows=[{'owner':'a:x','seed':1701}]*100+[{'owner':'b:y','seed':1701}]
    result=group_means(rows,[1.]*100+[-1.],['a:x','b:y'])
    assert result['macro']==0 and len(result['owners'])==2
