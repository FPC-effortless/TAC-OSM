"""Source-causal, counted, trained-path tests for factorial controls; NO data fit."""
import sys
from pathlib import Path
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"research/mathematical-engineering/langlayer040"))
from factorial_depth import FactorialDepthControl040

@pytest.mark.parametrize("dim,graph,count,ffn",[(48,False,499623,455),(59,True,499848,426)])
def test_count_causal_and_gradients(dim,graph,count,ffn):
    torch.manual_seed(40000+dim)
    m=FactorialDepthControl040(state_width=dim,upper_graph=graph)
    assert m.parameter_count()==count
    assert sum(p.numel() for p in m.parameters())==count
    assert m.ffn_width==ffn
    assert m.upper_concept_graph==graph
    names=dict(m.layernorm[1].named_parameters())
    assert ("concept_q.weight" in names)==graph
    assert ("concept_gate.weight" in names)==graph
    x=torch.tensor([[1,2,3,4,5,6,7],[11,10,9,8,7,6,5]])
    y=x.clone(); y[:,5:]=torch.tensor([[27,31],[37,41]])
    with torch.no_grad():
        out,tr=m(x,return_trace=True)
        changed=m(y)
    assert out.shape==(2,7,256)
    assert torch.isfinite(out).all()
    assert torch.allclose(out[:,:5],changed[:,:5],atol=1e-6)
    assert torch.equal(tr["native_memory"][:,0],torch.zeros_like(tr["native_memory"][:,0]))
    logits=m(x)
    logits.square().mean().backward()
    required=("input_proj.weight","forget.weight","offer.weight","read_gate.weight","output.weight","residual_gain")
    if graph:required+=("concept_q.weight","concept_k.weight","concept_to_state.weight")
    for name in required:
        p=names[name]
        assert p.grad is not None and torch.isfinite(p.grad).all()
        assert p.grad.abs().max()>0
    assert m.ffn_out.weight.grad.abs().max()>0

def test_illegal_combinations_rejected():
    for d,g in [(48,True),(59,False),(20,False)]:
        with pytest.raises(ValueError):
            FactorialDepthControl040(state_width=d,upper_graph=g)
