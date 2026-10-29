"""LANGMEM079 constructed true-stream original 078 slow-native semantics checks.

These are synthetic input/constructed engineering tests only, NOT scientific
evidence that the slow lane improves real publisher next-byte accuracy.
"""
from dataclasses import replace
from pathlib import Path
import sys
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1]
for name in ("langmem079","langmem078","langscale075","context026","langscale070","langlayer038","langmemory042"):
    p=str(ROOT/"research/mathematical-engineering"/name)
    if p not in sys.path:sys.path.insert(0,p)
from dual_streaming import DualNativeStreamingTAC079,state_shapes
from dual_native import DualNativeTimescaleTAC078

def test_ten_field_original_source_growing_prefix_and_snapshot(tmp_path):
    torch.manual_seed(79010)
    model=DualNativeTimescaleTAC078().eval()
    e=DualNativeStreamingTAC079(model,"native")
    owners=("first-owner","second-owner")
    t=torch.randint(0,256,(2,24),dtype=torch.long)
    assert e.start(owners).tensor_bytes()==2*4160
    with torch.no_grad():
        logits,state=e.prefill(t[:,:16],owners=owners)
        assert state.clock==16 and state.tensor_bytes()==2*4160
        assert state.tensors["native"].shape==(2,8,8)
        assert state.tensors["slow_native"].shape==(2,8,8)
        assert (logits-model(t[:,:16])[:,-1]).abs().max().item()<=3e-5
        for idx in range(16,24):
            output,state=e.step(t[:,idx],state,owners=owners)
            true=model(t[:,:idx+1])[:,-1]
            assert (output-true).abs().max().item()<=3e-5
        p=tmp_path/"original_dual_state.pt"
        e.save_state(state,p)
        recovered=e.load_state(p,owners=owners)
        assert recovered.clock==state.clock and recovered.tensor_bytes()==2*4160
        assert all(torch.equal(state.tensors[n],recovered.tensors[n]) for n in state.tensors)
    with pytest.raises(ValueError):e.load_state(p,owners=tuple(reversed(owners)))


def test_masked_write_preserves_both_native_matrices():
    torch.manual_seed(79011)
    model=DualNativeTimescaleTAC078().eval()
    e=DualNativeStreamingTAC079(model,"cap_all")
    owner=("owner",)
    with torch.no_grad():
        _,state=e.prefill(torch.arange(32,dtype=torch.long).view(1,-1),owners=owner)
        new_token=torch.tensor([17])
        _,masked=e.step(new_token,state,owners=owner,write_mask=torch.tensor([False]))
        assert torch.equal(masked.tensors["slow_native"],state.tensors["slow_native"])
        assert torch.equal(masked.tensors["native"],state.tensors["native"])
        assert masked.clock==33
        _,written=e.step(new_token,state,owners=owner)
        assert not torch.equal(written.tensors["slow_native"],state.tensors["slow_native"])


def test_old_slow_contribution_does_not_require_changing_fast_source():
    torch.manual_seed(79012)
    m=DualNativeTimescaleTAC078().eval()
    e=DualNativeStreamingTAC079(m)
    owner=("one",)
    with torch.no_grad():
        _,s=e.prefill(torch.arange(24,dtype=torch.long).view(1,24),owners=owner)
        zero=replace(s,tensors={**s.tensors,"slow_native":torch.zeros_like(s.tensors["slow_native"])})
        y,actual=e.step(torch.tensor([31]),s,owners=owner)
        z,ablated=e.step(torch.tensor([31]),zero,owners=owner)
        assert y.shape==z.shape==(1,256)
        assert torch.isfinite(y).all() and torch.isfinite(z).all()
        assert actual.clock==ablated.clock==25
        assert torch.equal(actual.tensors["native"],ablated.tensors["native"])
        assert torch.equal(actual.tensors["aligned"],ablated.tensors["aligned"])


def test_registered_dual_state_rejects_other_sizes():
    assert state_shapes(8)["slow_native"]==(8,8)
    for k in (0,12,16,True,8.0,None):
        with pytest.raises(ValueError):state_shapes(k)
