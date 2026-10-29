"""LANGMEM077 source native gate identity: synthetic ENGINEERING checks only."""
import sys
from dataclasses import replace
from pathlib import Path
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
for name in ("langstream076","langscale075","langscale070","langmemory042","context026","langlayer040","langstream043"):
    path=str(ROOT/"research/mathematical-engineering"/name)
    if path not in sys.path: sys.path.insert(0,path)
from native_matrix import NativeMatrixTAC075
from native_streaming import NativeStreamingTAC076
from scripts.langmem077_retention_probe import native_transition_factors


@pytest.mark.parametrize("tier",["m8","m12","m16"])
def test_exact_native_difference_dynamics_and_retention_bound(tier):
    torch.manual_seed(7702)
    model=NativeMatrixTAC075(tier).eval()
    e=NativeStreamingTAC076(model,"cap_all")
    owner=("strict-original-fixture",)
    # Constructed bytes; these must NEVER count toward real capability.
    tokens=torch.tensor([[1,2,4,8,16,32,64,127,72,73,91]],dtype=torch.long)
    with torch.inference_mode():
        _,initial=e.prefill(tokens[:,:8],owners=owner)
        altered=replace(initial,tensors={**initial.tensors,
                   "native":torch.zeros_like(initial.tensors["native"])})
        initial_norm=torch.linalg.matrix_norm(initial.tensors["native"].double()).item()
        assert initial_norm>0
        prod=1.0
        for t in range(8,11):
            token=tokens[:,t]
            f=native_transition_factors(model,initial,token,"cap_all")
            assert f["write_key_norm"]<=1.00001
            assert 0<f["alpha"]<1 and 0<f["erase"]<1
            before=(initial.tensors["native"]-altered.tensors["native"]).double()
            expected=(before@f["P"].double()).squeeze(0)
            _,initial_next=e.step(token,initial,owners=owner)
            _,altered_next=e.step(token,altered,owners=owner)
            actual=(initial_next.tensors["native"]-altered_next.tensors["native"]).double().squeeze(0)
            assert torch.max(torch.abs(actual-expected)).item()<1e-5
            prod*=f["alpha"]
            assert torch.linalg.matrix_norm(actual).item()/initial_norm<=prod+1e-4
            initial,altered=initial_next,altered_next


def test_masked_native_write_preserves_difference():
    torch.manual_seed(7703)
    m=NativeMatrixTAC075("m16").eval()
    engine=NativeStreamingTAC076(m)
    owner=("owner",)
    with torch.inference_mode():
        _,state=engine.prefill(torch.arange(32,dtype=torch.long).view(1,32),owners=owner)
        changed=replace(state,tensors={**state.tensors,
                   "native":torch.zeros_like(state.tensors["native"])})
        before=state.tensors["native"]-changed.tensors["native"]
        _,a=engine.step(torch.tensor([67]),state,owners=owner,
                        write_mask=torch.tensor([False]))
        _,b=engine.step(torch.tensor([67]),changed,owners=owner,
                        write_mask=torch.tensor([False]))
        assert torch.equal(a.tensors["native"]-b.tensors["native"],before)


def test_current_query_write_key_cosine_is_geometric_not_outcome():
    torch.manual_seed(7704)
    m=NativeMatrixTAC075("m8").eval()
    engine=NativeStreamingTAC076(m)
    with torch.inference_mode():
        _,s=engine.prefill(torch.arange(6,dtype=torch.long).view(1,6),owners=("a",))
        f=native_transition_factors(m,s,torch.tensor([7]),"native")
        assert f["query"].shape==(1,8)
        assert f["old_query_write_key"].shape==(1,8)
        assert abs(f["query_write_cosine"])<=1.00001
        assert f["P"].shape==(1,8,8)
