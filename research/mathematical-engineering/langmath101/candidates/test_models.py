"""Candidate source/gradient/causal checks; no neural training or capability score."""
import hashlib
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

# Resolve this exact source file independently of the checkout's directory name
# and pytest's import mode. Treat models.py as a package for its sibling imports.
_path=Path(__file__).resolve().with_name('models.py')
_name='_langmath101_candidate_tests_'+hashlib.sha256(str(_path).encode()).hexdigest()[:12]
_spec=importlib.util.spec_from_file_location(_name,_path,submodule_search_locations=[str(_path.parent)])
_models=importlib.util.module_from_spec(_spec);sys.modules[_name]=_models;_spec.loader.exec_module(_models)
FAMILY_NAMES,TARGET=_models.FAMILY_NAMES,_models.TARGET
catalog,make_model,intervene=_models.catalog,_models.make_model,_models.intervene

torch.set_num_threads(1)


def test_native_file_is_unchanged_original_source():
    raw=Path(__file__).with_name('native098.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest()=='e59c0aec80a39d84a6f01a7ca662d0c8baf3e2c2d33901057f017cb23c999498'


def test_all_live_counts_and_pairwise_parity():
    counts=[make_model(n).counts()['trainable'] for n in catalog()]
    assert min(counts)>=95000 and max(counts)<=105000
    assert (max(counts)-min(counts))/max(counts)<=.01
    assert all(make_model(n).counts()['trainable']==make_model(n).counts()['deployed'] for n in catalog())


@pytest.mark.parametrize('name',FAMILY_NAMES)
def test_all_parameter_tensors_are_connected_finite_and_nonzero(name):
    torch.manual_seed(31001);m=make_model(name).double()
    x=torch.tensor([[3,9,1,7,2,5,11,4,13],[19,21,17,23,31,14,27,12,15]])
    logits,_=m(x,owners=('a','b'))
    loss=(logits*torch.linspace(-.7,.9,logits.numel(),dtype=logits.dtype).reshape_as(logits)).square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and torch.count_nonzero(p.grad)>0
               for p in m.parameters()),[(n,None if p.grad is None else float(p.grad.abs().sum())) for n,p in m.named_parameters()]


@pytest.mark.parametrize('name',FAMILY_NAMES)
def test_no_future_input_can_change_previous_prediction(name):
    torch.manual_seed(31002);m=make_model(name).eval()
    x=torch.tensor([[3,4,7,8,11,15,21,30]]);y=x.clone();y[:,5:]=255
    with torch.inference_mode():
        a,_=m(x,owners=('a',));b,_=m(y,owners=('a',))
    assert torch.equal(a[:,:5],b[:,:5])


@pytest.mark.parametrize('name',[n for n in FAMILY_NAMES if not n.startswith('stateless')])
def test_serial_parallel_and_carried_chunk_correspondence(name):
    torch.manual_seed(31003);m=make_model(name).double().eval()
    x=torch.tensor([[3,4,7,8,11,15,21,30,41,45,50]])
    with torch.inference_mode():
        a,s=m(x,owners=('a',),parallel=True)
        b,t=m(x,owners=('a',),parallel=False)
        first,c=m(x[:,:4],owners=('a',));second,u=m(x[:,4:],state=c,owners=('a',))
    assert torch.allclose(a,b,atol=1e-10,rtol=1e-9)
    assert torch.allclose(a,torch.cat((first,second),1),atol=1e-10,rtol=1e-9)
    assert s.clock==t.clock==u.clock==11
    assert all(torch.allclose(v,u.tensors[k],atol=1e-10,rtol=1e-9) for k,v in s.tensors.items())


@pytest.mark.parametrize('name',FAMILY_NAMES)
def test_batch_owners_are_isolated(name):
    torch.manual_seed(31004);m=make_model(name).double().eval()
    x=torch.tensor([[3,4,7,8],[11,15,21,30]])
    with torch.inference_mode():
        together,_=m(x,owners=('a','b'))
        a,_=m(x[:1],owners=('a',));b,_=m(x[1:],owners=('b',))
    assert torch.allclose(together,torch.cat((a,b)),atol=1e-10,rtol=1e-9)


@pytest.mark.parametrize('name',[n for n in FAMILY_NAMES if n.startswith('stateless')])
def test_stateless_has_no_hidden_mutable_stream_state(name):
    torch.manual_seed(31005);m=make_model(name).eval()
    with torch.inference_mode():
        a,s=m(torch.tensor([[63]]),owners=('a',));m(torch.tensor([[1,2,3,4]]),owners=('b',));b,t=m(torch.tensor([[63]]),owners=('a',))
    assert s.tensors==t.tensors=={} and s.clock==t.clock==0 and torch.equal(a,b)


def test_interventions_copy_without_mutating_original():
    m=make_model('hierarchy_assoc_3')
    with torch.inference_mode():_,s=m(torch.tensor([[1,2,3,4]]),owners=('a',))
    before={k:v.clone() for k,v in s.tensors.items()}
    changed=intervene(s,('0',))
    assert changed.clock==s.clock and changed.owners==s.owners
    assert all(torch.count_nonzero(v)==0 for k,v in changed.tensors.items() if k.startswith('0.'))
    assert all(torch.equal(v,s.tensors[k]) for k,v in before.items())


@pytest.mark.parametrize('name',FAMILY_NAMES)
def test_model_boundary_rejects_unregistered_extra_fields(name):
    m=make_model(name)
    with pytest.raises((ValueError,TypeError)):
        m(torch.tensor([[1,2,3]]),owners=('a',),future_targets=torch.tensor([4]))


def test_fast_and_slow_reset_is_union_not_last_selection():
    m=make_model('native098')
    with torch.inference_mode():_,s=m(torch.tensor([[1,2,3,4]]),owners=('a',))
    a=intervene(s,('fast','slow'));b=intervene(s,('banks',))
    assert all(torch.equal(a.tensors[k],b.tensors[k]) for k in a.tensors)
