"""LANGLAYER037 constructed software diagnostics only; no training or TEST."""
import sys
from pathlib import Path

import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"research/mathematical-engineering/langlayer037"))
from feedforward_transition import TokenwiseResidualFFNBlock, FeedforwardDepthControl037


def test_counts_and_active_total_sizes():
    assert {d:FeedforwardDepthControl037(depth=d).parameter_count()
            for d in (2,3)}=={2:499661,3:499684}
    for d in (2,3):
        m=FeedforwardDepthControl037(depth=d)
        assert len(m.layernorm)==d
        assert m.architecture_record()["deployed_parameters"]==m.parameter_count()
        assert all(p.requires_grad for p in m.parameters())
        assert sum(p.numel() for p in m.parameters())>499000
    with pytest.raises(ValueError):
        FeedforwardDepthControl037(depth=4)


@pytest.mark.parametrize("width", [24,48])
def test_tokenwise_block_has_no_temporal_dependency(width):
    torch.manual_seed(3700+width)
    b=TokenwiseResidualFFNBlock(hidden=width).double().eval()
    x=torch.randn(2,9,280,dtype=torch.double)
    permutation=torch.tensor([2,1,6,0,7,8,4,5,3])
    with torch.no_grad():
        actual=b(x)
        permuted=b(x[:,permutation])
        corrupted=x.clone()
        corrupted[:,5:]+=10.0
        poisoned=b(corrupted)
    assert torch.allclose(permuted,actual[:,permutation],atol=1e-12,rtol=1e-12)
    assert torch.allclose(actual[:,:5],poisoned[:,:5],atol=1e-12,rtol=1e-12)


@pytest.mark.parametrize("depth", [2,3])
def test_whole_model_prefix_causal_and_native_mask(depth):
    torch.manual_seed(3760+depth)
    m=FeedforwardDepthControl037(depth=depth).double().eval()
    x=torch.tensor([[11,12,13,14,15,16,17,18]])
    y=x.clone();y[:,5:]=torch.tensor([[21,22,23]])
    with torch.no_grad():
        logits,trace=m(x,return_trace=True)
        changed=m(y)
    assert logits.shape==(1,8,256)
    assert torch.isfinite(logits).all()
    assert torch.allclose(logits[:,:5],changed[:,:5],atol=1e-12,rtol=1e-12)
    assert not trace["native_admission"][:,0].any()
    assert torch.equal(trace["native_memory"][:,0],torch.zeros_like(trace["native_memory"][:,0]))
    with pytest.raises(ValueError):
        m(x,start=12)


@pytest.mark.parametrize("depth", [2,3])
def test_useful_gradient_paths_all_ffn_blocks(depth):
    torch.manual_seed(3790+depth)
    m=FeedforwardDepthControl037(depth=depth).double()
    x=torch.randint(0,256,(2,9))
    m(x).square().mean().backward()
    for i in range(1,depth):
        block=m.layernorm[i]
        for name in ("up.weight","gate.weight","down.weight","residual_gain"):
            p=dict(block.named_parameters())[name]
            assert p.grad is not None and torch.isfinite(p.grad).all()
            assert bool((p.grad.abs()>0).any()),f"disconnected block {i}: {name}"
    assert bool((m.context_lag_weights.grad.abs()>0).any())
    assert bool((m.ffn_out.weight.grad.abs()>0).any())
