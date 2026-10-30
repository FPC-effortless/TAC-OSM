"""No-fit exact algebraic identity and source-side input boundaries."""
import importlib.util
from pathlib import Path
import pytest
import torch
from torch.nn import functional as F

spec=importlib.util.spec_from_file_location('hybgeom_math092',Path(__file__).with_name('math092.py'))
math=importlib.util.module_from_spec(spec);spec.loader.exec_module(math)

@pytest.mark.parametrize('dtype', [torch.float32,torch.float64])
def test_exact_tied_decomposition(dtype):
    torch.manual_seed(92)
    embeddings=torch.randn(256,32,dtype=dtype)
    current=torch.tensor([1,18,255,0,18])
    strength=torch.tensor(.013,dtype=dtype)
    terms=math.corrections(embeddings,current,strength)
    assert math.assert_equivalence(terms, tolerance=(1e-6 if dtype==torch.float64 else 6e-6)) <= 6e-6
    assert torch.allclose(terms['similarity']+terms['vocab_norm'],terms['algebraic'],atol=1e-10)


def test_softmax_probabilities_unchanged_by_candidate_mean_shift():
    torch.manual_seed(3)
    e=torch.randn(256,32)
    current=torch.tensor([7,9,17])
    s=torch.tensor(.009)
    out=math.corrections(e,current,s)
    raw=(e[None]-e[current,None]).square().sum(-1)
    logits=torch.randn(3,256)
    assert torch.allclose(F.log_softmax(logits-s*raw,-1),F.log_softmax(logits+out['algebraic'],-1),atol=3e-6)


def test_backprop_to_original_embeds_and_strength_identical():
    torch.manual_seed(41)
    base=torch.randn(4,256)
    labels=torch.tensor([1,7,3,10])
    current=torch.tensor([32,18,255,0])
    embedding=torch.randn(256,32,requires_grad=True)
    weight=torch.tensor(.012,requires_grad=True)
    pieces=math.corrections(embedding,current,weight)
    loss_a=F.cross_entropy(base+pieces['original_distance'],labels)
    grad_a=torch.autograd.grad(loss_a,(embedding,weight),retain_graph=True)
    loss_b=F.cross_entropy(base+pieces['algebraic'],labels)
    grad_b=torch.autograd.grad(loss_b,(embedding,weight))
    assert torch.allclose(loss_a,loss_b,atol=1e-6)
    assert max(float((x-y).abs().max()) for x,y in zip(grad_a,grad_b))<1e-5
    assert all(bool(x.isfinite().all()) for x in grad_b)


def test_same_observed_prefix_future_poison_noninterference():
    torch.manual_seed(8)
    e=torch.randn(256,32)
    current=torch.tensor([11,33,51,2,100,90])
    altered=current.clone();altered[3:]=(altered[3:]+19)%256
    x=math.corrections(e,current,torch.tensor(.014))
    y=math.corrections(e,altered,torch.tensor(.014))
    for k in x: assert torch.equal(x[k][:3],y[k][:3]), k


def test_shuffled_similarity_is_genuine_change():
    torch.manual_seed(92)
    e=torch.randn(256,32)
    pieces=math.corrections(e,torch.tensor([7,11]),torch.tensor(.011))
    assert (pieces['similarity']-pieces['shuffled_similarity']).abs().max()>1e-3

@pytest.mark.parametrize('invalid',[-.01,.03,float('nan')])
def test_strength_invalid_fails_closed(invalid):
    with pytest.raises(ValueError): math.corrections(torch.zeros(256,32),torch.tensor([2]),torch.tensor(invalid))


def test_illegal_current_indices_and_permutation():
    e=torch.ones(256,32);s=torch.tensor(.01)
    with pytest.raises(ValueError): math.corrections(e,torch.tensor([256]),s)
    with pytest.raises(ValueError): math.corrections(e,torch.tensor([-1]),s)
    with pytest.raises(ValueError): math.corrections(e,torch.tensor([0]),s,permutation=torch.zeros(256,dtype=torch.long))
